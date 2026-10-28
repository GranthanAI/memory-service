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
import re
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
from app.services import anchors as anchor_ledger
from app.services.branch_validity import summary_valid_at_branch
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


class SummaryValidationError(RuntimeError):
    """The new summary lost a pinned fact or invented an entity, even after one corrective retry. Nothing was written: the
    previous summary stays and the watermark does not advance, so the batch is summarized again later."""


# Share of the token cap each section of the summary may use when the deterministic fallback has to cut (the prompt's headings,
# most important first). Identifiers also live in the anchor ledger, so KEY FACTS can be cut without losing them.
SECTION_WEIGHTS = (3.0, 3.0, 1.5, 2.0, 3.0, 2.0, 2.0, 1.0)
_HEADING = re.compile(r"^\s*(?:#{1,3}\s*)?(?:\d+[.)]\s*)?([A-Z][A-Z &/'-]{3,})\s*:?\s*$")


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
        instructions: Optional[str] = None,
        upto_message_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Summarizes the messages that arrived after the stored watermark. Returns the snapshot
        (unchanged when there was nothing new). With `upto_message_id` (a branch point) only messages at or before it are
        summarized, so a child branch can inherit a summary that ends exactly there."""
        snapshot = await self.memory_repo.get_snapshot(conversation_id)
        if not snapshot:
            raise ValueError(f"Snapshot not found for conversation '{conversation_id}'.")

        started = time.monotonic()
        committed = 0
        consumed_ids: List[str] = []
        try:
            for _batch in range(MAX_BATCHES):
                outcome = await self._summarize_one_batch(conversation_id, upto_message_id)
                if outcome is None:
                    break
                committed += 1
                consumed_ids.extend(outcome.get("message_ids", []))
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

        # Exactly the messages this job turned into summary: lets fact extraction work from those, not from "the last 10".
        snapshot["consumed_message_ids"] = consumed_ids
        if committed:
            snapshot["updated_at"] = datetime.now(timezone.utc)
            # Only the summary columns are written. The snapshot row was read before the model call; writing it back
            # whole would reset message_count / snapshot_version that the dispatcher bumped in the meantime.
            await self.memory_repo.update_snapshot_summary_fields(
                conversation_id, snapshot["summary_version"], snapshot["last_summary_msg_id"])
            await self.memory_repo.invalidate_conversation(conversation_id)
        else:
            COMPACTION_RESULTS.labels(outcome="nothing_new").inc()
            logger.info(f"No new messages after the summary watermark for '{conversation_id}'; no new version.")
        if self.scheduler is not None:
            # Pressure restarts from the NEW watermark: messages that arrived while the job ran are still pending.
            pending_messages, pending_tokens = self._pending_after_watermark(conversation_id)
            await self.scheduler.reset(conversation_id, pending_messages, pending_tokens)
        return snapshot

    def _pending_after_watermark(self, conversation_id: str) -> Tuple[int, int]:
        """(messages, tokens) stored after the committed watermark, counted from the durable rows."""
        try:
            row = self.cassandra_repo.get_summary(conversation_id) or {}
            fetched = self.cassandra_repo.get_recent_messages(conversation_id, limit=FETCH_LIMIT)
            new, _gap = select_new_messages(fetched, row.get("through_message_id"), row.get("through_created_at"))
        except Exception as exc:
            logger.warning("could not recount pending messages (%s)", type(exc).__name__)
            return 0, 0
        return len(new), sum(self.counter.count(m.get("content") or "") + 4 for m in new)

    async def _renew_inflight(self, conversation_id: str) -> None:
        renew = getattr(self.scheduler, "renew", None)
        if renew is None:
            return
        try:
            await renew(conversation_id)
        except Exception as exc:                                  # best effort: the TTL is already generous
            logger.warning("could not renew the compaction guard (%s)", type(exc).__name__)

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

    def _split_sections(self, text: str):
        """[(heading line or None, [body lines])] following the prompt's headings; a preamble has heading None."""
        sections: list = [[None, []]]
        for line in text.split("\n"):
            if _HEADING.match(line):
                sections.append([line, []])
            else:
                sections[-1][1].append(line)
        return [(h, b) for h, b in sections if h is not None or any(l.strip() for l in b)]

    def _trim_by_sections(self, text: str) -> Optional[str]:
        """Cut each section to its own budget at whole lines (bullets), instead of cutting the END of the text, which is
        where TIMELINE and KEY FACTS live. Budget a section does not use is shared with the others. None when the text has
        no recognisable headings."""
        sections = self._split_sections(text)
        headed = [i for i, (h, _b) in enumerate(sections) if h is not None]
        if len(headed) < 2:
            return None
        cap = settings.SUMMARY_MAX_TOKENS
        weights = {i: SECTION_WEIGHTS[min(n, len(SECTION_WEIGHTS) - 1)] for n, i in enumerate(headed)}
        cost = {i: self.counter.count("\n".join(([sections[i][0]] if sections[i][0] else []) + sections[i][1])) for i in range(len(sections))}
        budget = {i: 0.0 for i in range(len(sections))}
        pool = float(cap - sum(cost[i] for i in range(len(sections)) if i not in weights))      # preamble is kept as is
        open_ids = set(weights)
        while open_ids and pool > 0:
            total_w = sum(weights[i] for i in open_ids)
            settled = [i for i in open_ids if cost[i] <= pool * weights[i] / total_w]
            if not settled:
                for i in open_ids:
                    budget[i] = pool * weights[i] / total_w
                break
            for i in settled:
                budget[i] = cost[i]
                pool -= cost[i]
                open_ids.discard(i)
        out: list = []
        for i, (heading, body) in enumerate(sections):
            if i not in weights or cost[i] <= budget[i]:
                out.extend(([heading] if heading else []) + body)
                continue
            kept, used = [], self.counter.count(heading or "")
            for line in body:
                c = self.counter.count(line) + 1
                if used + c > budget[i]:
                    break
                kept.append(line)
                used += c
            if any(l.strip() for l in kept):                       # a heading with nothing under it is not kept
                out.extend(([heading] if heading else []) + kept)
        return "\n".join(out).strip()

    def _trim(self, text: str) -> str:
        """Deterministic fallback: per-section budgets (see _trim_by_sections); only text without the expected headings is
        cut from its tail."""
        by_section = self._trim_by_sections(text)
        if by_section and not self._over_limit(by_section):
            return by_section
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

    def _load_ledger(self, conversation_id: str) -> List[Dict[str, Any]]:
        lister = getattr(self.cassandra_repo, "list_anchors", None)
        if lister is None:
            return []
        try:
            return list(lister(conversation_id) or [])
        except Exception as exc:                                  # the ledger is additive: a read failure must not block summaries
            logger.warning("anchor ledger unavailable (%s)", type(exc).__name__)
            return []

    async def _summarize_and_validate(self, conversation_id: str, previous_text: str, batch: List[Dict[str, Any]]):
        """One model call (plus at most SUMMARY_VALIDATION_RETRIES corrective retries) producing a summary that is checked
        against the anchor ledger. Returns (summary, fit outcome, ledger entries to write). Raises SummaryValidationError
        when the summary still loses a pinned fact or invents an entity: nothing is written then."""
        def request(feedback: Optional[str] = None) -> SummarizeRequest:
            return SummarizeRequest(
                previous_summary=previous_text,
                new_messages=[LLMMessage(role=m["role"], content=m["content"]) for m in batch],
                feedback=feedback)

        response = await self.llm_service.summarize(request())
        text, fit_outcome = await self.fit_summary(response.summary)
        if not settings.ANCHOR_LEDGER_ENABLED:
            return text, fit_outcome, []

        prior = self._load_ledger(conversation_id)
        if not prior and previous_text.strip():                   # bootstrap: a summary written before the ledger existed
            prior = [{"kind": "identifier", "text": t, "source_message_id": "", "status": "active"}
                     for t in anchor_ledger.hard_entities(previous_text)]
        new_anchors = anchor_ledger.extract_identifiers(batch)
        done_texts: List[str] = []
        extractor = getattr(self.llm_service, "extract_anchors", None)
        if settings.ANCHOR_LLM_EXTRACTION and extractor is not None:
            open_items = [a["text"] for a in prior if a.get("kind") == "open_item" and a.get("status") == "open"]
            try:
                judged, done_texts = await extractor(batch, open_items)
                new_anchors = new_anchors + list(judged)
            except Exception as exc:                              # identifiers still protect the numbers and names
                logger.warning("anchor extraction unavailable (%s)", type(exc).__name__)
        merged = anchor_ledger.merge_ledger(prior, new_anchors, done_texts)
        new_text = " ".join(str(m.get("content") or "") for m in batch)

        def check(summary: str):
            return anchor_ledger.validate_summary(
                new_summary=summary, previous_summary=previous_text, new_messages_text=new_text,
                prior_ledger=prior, merged_ledger=merged)

        report = check(text)
        for _ in range(settings.SUMMARY_VALIDATION_RETRIES):
            if report.ok:
                break
            response = await self.llm_service.summarize(request(report.feedback()))
            text, fit_outcome = await self.fit_summary(response.summary)
            report = check(text)
        if not report.ok:
            lost = bool(report.lost_anchors or report.dropped_open_items)
            COMPACTION_RESULTS.labels(outcome="anchor_lost" if lost else "unsupported_entity").inc()
            raise SummaryValidationError(
                f"summary for '{conversation_id}' failed validation (lost={len(report.lost_anchors)}, "
                f"open_items_dropped={len(report.dropped_open_items)}, unsupported={len(report.unsupported_entities)}); "
                "the previous summary is kept and the batch stays unsummarized")

        stored = {r["anchor_id"]: r.get("status") for r in prior if r.get("anchor_id")}     # bootstrap rows are not stored yet
        changes = [a for a in merged if a.anchor_id not in stored or stored[a.anchor_id] != a.status]
        return text, fit_outcome, changes

    async def _summarize_one_batch(self, conversation_id: str, upto_message_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
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
            if upto_message_id:
                new = [m for m in new if summary_valid_at_branch(m.get("message_id"), upto_message_id, m.get("created_at"))]
            if not new:
                return None
            batch = new[:BATCH_SIZE]

            await self._renew_inflight(conversation_id)               # the in-flight guard must outlive a slow model call
            text, fit_outcome, ledger_changes = await self._summarize_and_validate(conversation_id, previous_text, batch)

            last = batch[-1]
            record = {
                "conversation_id": conversation_id,
                "summary_text": text,
                "summary_version": (expected_version or 0) + 1,
                "model_name": settings.LLM_MODEL,
                "model_version": settings.LLM_MODEL_VERSION or settings.LLM_MODEL,
                "llm_provider": settings.LLM_PROVIDER,
                "prompt_version": settings.SUMMARY_PROMPT_VERSION,
                "generated_at": datetime.now(timezone.utc),
                "through_message_id": str(last["message_id"]),
                "through_created_at": _as_utc(last.get("created_at")),
            }
            # Archive what is about to be replaced. Cassandra cannot make this one conditional batch with the compare-and-set
            # below (different tables), so the order is: archive (idempotent) -> compare-and-set. A crash in between leaves
            # an extra archived copy of the still-current row, never a lost version; an archive failure aborts the commit.
            archive = getattr(self.cassandra_repo, "archive_summary_version", None)
            if archive is not None and row and row.get("summary_version") is not None:
                archive(row)
            # The ledger is written BEFORE the summary commit: a committed summary always has its anchors. If the commit then
            # loses a race, the anchors are still true facts of real messages (idempotent upsert), so they are harmless.
            upsert_anchors = getattr(self.cassandra_repo, "upsert_anchors", None)
            if ledger_changes and upsert_anchors is not None:
                upsert_anchors(conversation_id, ledger_changes)
            if self.cassandra_repo.write_summary_if_current(record, expected_version):
                await self._renew_inflight(conversation_id)
                COMPACTION_RESULTS.labels(outcome=fit_outcome).inc()
                COMPACTION_SOURCE_MESSAGES.observe(len(batch))
                SUMMARY_TOKENS.observe(self.counter.count(text))
                logger.info(
                    f"Committed summary v{record['summary_version']} for '{conversation_id}' "
                    f"({len(batch)} new message(s), fit={fit_outcome})."
                )
                return {
                    "message_ids": [str(m["message_id"]) for m in batch],
                    "summary_version": record["summary_version"],
                    "through_message_id": record["through_message_id"],
                    "more": len(new) > len(batch),
                }
            logger.info(f"Summary update for '{conversation_id}' lost a concurrent write (attempt {attempt}); re-reading.")

        raise ConcurrentSummaryUpdate(
            f"summary for '{conversation_id}' kept changing concurrently ({MAX_CAS_ATTEMPTS} attempts)"
        )
