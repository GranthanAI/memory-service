"""
app/services/summary_service.py

Incremental conversation summarization (the existing "conversation summary" memory).

Watermark semantics
-------------------
Messages are totally ordered by (created_at, message_id), the clustering order of the durable
`conversation_recent_messages` table (message ids are NOT assumed to be chronological on their own).
A summary row records the last message it incorporates (`through_message_id` + `through_created_at`).
"New" messages for the next update are exactly those strictly after that watermark, so a rolling
window that still contains already-summarized messages never double counts them. Which messages are
new is decided here, deterministically; the model never decides that from message text.

Concurrency and retries
-----------------------
Summary text, version and watermark live in one row and are written by one conditional update
(`IF summary_version = <version read>`; `IF NOT EXISTS` for the first). A stale job therefore cannot overwrite
a newer summary or move the watermark backward: it re-reads the committed summary and recomputes what is
new (bounded retries), and if it still cannot commit it fails visibly so the existing retry/DLQ path runs.
A failed model call, empty output or crash before the conditional write leaves the previous
summary untouched; a redelivered event finds no messages after the watermark and creates no new version.

Fitting the summary (validated in TOKENS, never silently exceeded)
-----------------------------------------------------------------
The summary is a DERIVED representation: the source messages stay in the durable store, and recent/older exact messages
remain retrievable. A model does not reliably obey a length request, so the result is counted with the real tokenizer:
  1. over the limit  -> up to SUMMARY_MAX_REFINEMENTS shorter attempts (the same summarization capability, a
                        consolidation prompt), recounted each time;
  2. still over      -> deterministic fallback: cut at a section/sentence boundary. The prompt orders the summary by
                        importance, so the tail is the least important part. The new messages are still covered by the
                        watermark (the update is NOT lost), and the degradation is counted in metrics;
  3. close to the limit (>= SUMMARY_CONSOLIDATE_RATIO) -> one consolidation pass that merges duplicates and drops what is
                        no longer needed, so repeated incremental updates cannot drift or only ever grow.
"""

import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from app.core.config import settings
from app.core.metrics import (
    COMPACTION_RESULTS,
    COMPACTION_SECONDS,
    COMPACTION_SOURCE_MESSAGES,
    SUMMARY_TOKENS,
)
from app.repositories.cassandra_repository import CassandraRepository
from app.repositories.memory_repository import MemoryRepository
from app.services.llm_service import LLMService
from app.schemas.llm import SummarizeRequest, LLMMessage
from app.utils.token_counter import TokenCounter

logger = logging.getLogger("memory_service.services.summary_service")

FETCH_LIMIT = 200            # durable messages considered per attempt
CATCH_UP_FETCH_LIMIT = 1000  # one deeper read when the watermark is older than everything fetched (summary far behind)
BATCH_SIZE = 40              # new messages summarized per model call
MAX_BATCHES = 5              # model calls per job (leftovers are picked up by the next event)
MAX_CAS_ATTEMPTS = 3         # re-reads after losing a concurrent update
SUMMARY_MAX_CHARS = 6000     # secondary guard next to the token limit (a summary is compact)


class SummaryOutputError(RuntimeError):
    """The model returned an unusable summary (empty). Nothing was written."""


class ConcurrentSummaryUpdate(RuntimeError):
    """Another worker kept committing newer summaries; this job gave up without writing."""


def _as_utc(value: Any) -> Optional[datetime]:
    if isinstance(value, str):                    # the Redis window serializes timestamps as ISO strings
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    if not isinstance(value, datetime):
        return None
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def message_key(message: Dict[str, Any]) -> Tuple[datetime, str]:
    """Total order of messages: (created_at, message_id)."""
    created = _as_utc(message.get("created_at")) or datetime.min.replace(tzinfo=timezone.utc)
    return created, str(message.get("message_id") or "")


def select_new_messages(
    messages: List[Dict[str, Any]],
    through_message_id: Optional[str],
    through_created_at: Optional[datetime],
) -> Tuple[List[Dict[str, Any]], bool]:
    """Messages strictly after the watermark, oldest first, plus whether older messages may be missing
    (the watermark is older than everything that was fetched, so a gap cannot be ruled out)."""
    ordered = sorted(messages, key=message_key)
    if not through_message_id:
        return ordered, False

    mark_created = _as_utc(through_created_at)
    if mark_created is None:                                   # legacy row: locate the watermark by id
        for index, m in enumerate(ordered):
            if str(m.get("message_id")) == str(through_message_id):
                return ordered[index + 1:], False
        return ordered, True                                   # watermark not retained: summarize what is there

    mark = (mark_created, str(through_message_id))
    new = [m for m in ordered if message_key(m) > mark]
    gap = bool(ordered) and len(ordered) >= FETCH_LIMIT and message_key(ordered[0]) > mark
    return new, gap


def validate_summary(text: Optional[str]) -> str:
    """Empty output is unusable. Over-long output is NOT rejected here any more: SummaryService refines it or trims it
    deterministically, so a verbose model can never make an update (and its watermark) get lost."""
    cleaned = (text or "").strip()
    if not cleaned:
        raise SummaryOutputError("summary model returned an empty result")
    return cleaned


class SummaryService:
    """
    Coordinates versioned conversation summaries incrementally.
    """

    def __init__(
        self,
        memory_repo: MemoryRepository,
        cassandra_repo: CassandraRepository,
        llm_service: LLMService,
        scheduler=None,
        counter: Optional[TokenCounter] = None,
    ):
        self.memory_repo = memory_repo
        self.cassandra_repo = cassandra_repo
        self.llm_service = llm_service
        self.scheduler = scheduler                 # optional CompactionScheduler: told when a summary was committed
        self.counter = counter or TokenCounter()

    async def process_incremental_summary(
        self,
        conversation_id: str,
        instructions: Optional[str] = None
    ) -> Dict[str, Any]:
        """Summarizes the messages that arrived after the stored watermark. Returns the snapshot
        (unchanged when there was nothing new)."""
        snapshot = await self.memory_repo.get_snapshot(conversation_id)
        if not snapshot:
            raise ValueError(f"Snapshot not found for conversation '{conversation_id}'.")

        started = time.monotonic()
        committed = 0
        try:
            for _batch in range(MAX_BATCHES):
                outcome = await self._summarize_one_batch(conversation_id)
                if outcome is None:
                    break
                committed += 1
                snapshot["summary_version"] = outcome["summary_version"]
                snapshot["last_summary_msg_id"] = outcome["through_message_id"]
                if not outcome["more"]:
                    break
        except ConcurrentSummaryUpdate:
            COMPACTION_RESULTS.labels(outcome="lost_race").inc()
            raise
        except Exception:
            COMPACTION_RESULTS.labels(outcome="failed").inc()
            raise
        finally:
            COMPACTION_SECONDS.observe(time.monotonic() - started)

        if committed:
            snapshot["updated_at"] = datetime.now(timezone.utc)
            await self.memory_repo.save_snapshot(snapshot)
            await self.memory_repo.invalidate_conversation(conversation_id)
            if self.scheduler is not None:
                await self.scheduler.reset(conversation_id)       # pressure restarts from the new watermark
        else:
            COMPACTION_RESULTS.labels(outcome="nothing_new").inc()
            logger.info(f"No new messages after the summary watermark for '{conversation_id}'; no new version.")
            if self.scheduler is not None:
                await self.scheduler.reset(conversation_id)       # a duplicate/late request must not keep the guard set
        return snapshot

    # ─── fitting ─────────────────────────────────────────────────────────────────────────────────────────────────────

    def _over_limit(self, text: str) -> bool:
        return self.counter.count(text) > settings.SUMMARY_MAX_TOKENS or len(text) > SUMMARY_MAX_CHARS

    async def _consolidate(self, text: str, target_tokens: int) -> Optional[str]:
        """One shorter rewrite through the existing summarization capability. Never raises: if it is unavailable or
        fails, the caller keeps what it has (and falls back deterministically if that is over the limit)."""
        consolidate = getattr(self.llm_service, "consolidate_summary", None)
        if consolidate is None:
            return None
        try:
            from app.schemas.llm import ConsolidateSummaryRequest
            response = await consolidate(ConsolidateSummaryRequest(summary=text, target_tokens=target_tokens))
            shorter = (response.summary or "").strip()
            return shorter or None
        except Exception as exc:
            logger.warning("summary consolidation unavailable (%s)", type(exc).__name__)
            return None

    def _trim(self, text: str) -> str:
        """Deterministic fallback. Importance-ordered text loses its least important tail first."""
        clipped = self.counter.clip(text, settings.SUMMARY_MAX_TOKENS)
        if len(clipped) > SUMMARY_MAX_CHARS:
            clipped = clipped[: SUMMARY_MAX_CHARS - 1].rstrip() + "…"
        return clipped

    async def fit_summary(self, raw: Optional[str]) -> Tuple[str, str]:
        """Returns (summary within the limit, outcome). Raises SummaryOutputError only for EMPTY output."""
        text = validate_summary(raw)
        cap = settings.SUMMARY_MAX_TOKENS
        outcome = "committed"
        if self._over_limit(text):
            for _ in range(settings.SUMMARY_MAX_REFINEMENTS):
                shorter = await self._consolidate(text, int(cap * 0.8))
                if shorter:
                    text, outcome = shorter, "refined"
                if not self._over_limit(text):
                    break
            if self._over_limit(text):
                text, outcome = self._trim(text), "fallback_trimmed"
        elif self.counter.count(text) >= cap * settings.SUMMARY_CONSOLIDATE_RATIO:
            shorter = await self._consolidate(text, int(cap * settings.SUMMARY_CONSOLIDATE_TARGET_RATIO))
            if shorter and not self._over_limit(shorter) and self.counter.count(shorter) < self.counter.count(text):
                text, outcome = shorter, "consolidated"
        return validate_summary(text), outcome

    # ─── one model call, one conditional write ───────────────────────────────────────────────────────────────────────

    async def _summarize_one_batch(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """One read -> model -> conditional write cycle. None = nothing new."""
        for attempt in range(1, MAX_CAS_ATTEMPTS + 1):
            row = self.cassandra_repo.get_summary(conversation_id) or {}
            expected_version = row.get("summary_version") if row else None
            previous_text = (row.get("summary_text") or "") if row else ""

            fetched = self.cassandra_repo.get_recent_messages(conversation_id, limit=FETCH_LIMIT)
            new, gap = select_new_messages(fetched, row.get("through_message_id"), row.get("through_created_at"))
            if gap:
                # The summary is far behind: read deeper once, so messages older than the newest FETCH_LIMIT are not lost.
                fetched = self.cassandra_repo.get_recent_messages(conversation_id, limit=CATCH_UP_FETCH_LIMIT)
                new, gap = select_new_messages(fetched, row.get("through_message_id"), row.get("through_created_at"))
                if gap or len(fetched) >= CATCH_UP_FETCH_LIMIT:
                    logger.warning(f"Possible message gap before the summary watermark for '{conversation_id}'.")
            if not new:
                return None
            batch = new[:BATCH_SIZE]

            response = await self.llm_service.summarize(SummarizeRequest(
                previous_summary=previous_text,
                new_messages=[LLMMessage(role=m["role"], content=m["content"]) for m in batch],
            ))
            text, fit_outcome = await self.fit_summary(response.summary)

            last = batch[-1]
            record = {
                "conversation_id": conversation_id,
                "summary_text": text,
                "summary_version": (expected_version or 0) + 1,
                "model_name": settings.LLM_MODEL,
                "model_version": settings.LLM_PROVIDER,
                "generated_at": datetime.now(timezone.utc),
                "through_message_id": str(last["message_id"]),
                "through_created_at": _as_utc(last.get("created_at")),
            }
            if self.cassandra_repo.write_summary_if_current(record, expected_version):
                COMPACTION_RESULTS.labels(outcome=fit_outcome).inc()
                COMPACTION_SOURCE_MESSAGES.observe(len(batch))
                SUMMARY_TOKENS.observe(self.counter.count(text))
                logger.info(
                    f"Committed summary v{record['summary_version']} for '{conversation_id}' "
                    f"({len(batch)} new message(s), fit={fit_outcome})."
                )
                return {
                    "summary_version": record["summary_version"],
                    "through_message_id": record["through_message_id"],
                    "more": len(new) > len(batch),
                }
            logger.info(f"Summary update for '{conversation_id}' lost a concurrent write (attempt {attempt}); re-reading.")

        raise ConcurrentSummaryUpdate(
            f"summary for '{conversation_id}' kept changing concurrently ({MAX_CAS_ATTEMPTS} attempts)"
        )
