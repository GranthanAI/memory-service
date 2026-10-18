"""
app/services/summary_service.py

Incremental conversation summarization.

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
A failed model call, empty/oversized output or crash before the conditional write leaves the previous
summary untouched; a redelivered event finds no messages after the watermark and creates no new version.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from app.core.config import settings
from app.repositories.cassandra_repository import CassandraRepository
from app.repositories.memory_repository import MemoryRepository
from app.services.llm_service import LLMService
from app.schemas.llm import SummarizeRequest, LLMMessage

logger = logging.getLogger("memory_service.services.summary_service")

FETCH_LIMIT = 200            # durable messages considered per attempt
BATCH_SIZE = 40              # new messages summarized per model call
MAX_BATCHES = 5              # model calls per job (leftovers are picked up by the next event)
MAX_CAS_ATTEMPTS = 3         # re-reads after losing a concurrent update
SUMMARY_MAX_CHARS = 6000     # a summary is compact: beyond this the output is rejected, not truncated


class SummaryOutputError(RuntimeError):
    """The model returned an unusable summary (empty or oversized). Nothing was written."""


class ConcurrentSummaryUpdate(RuntimeError):
    """Another worker kept committing newer summaries; this job gave up without writing."""


def _as_utc(value: Any) -> Optional[datetime]:
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
    cleaned = (text or "").strip()
    if not cleaned:
        raise SummaryOutputError("summary model returned an empty result")
    if len(cleaned) > SUMMARY_MAX_CHARS:
        raise SummaryOutputError(f"summary model returned {len(cleaned)} characters (limit {SUMMARY_MAX_CHARS})")
    return cleaned


class SummaryService:
    """
    Coordinates versioned conversation summaries incrementally.
    """

    def __init__(
        self,
        memory_repo: MemoryRepository,
        cassandra_repo: CassandraRepository,
        llm_service: LLMService
    ):
        self.memory_repo = memory_repo
        self.cassandra_repo = cassandra_repo
        self.llm_service = llm_service

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

        committed = 0
        for _batch in range(MAX_BATCHES):
            outcome = await self._summarize_one_batch(conversation_id)
            if outcome is None:
                break
            committed += 1
            snapshot["summary_version"] = outcome["summary_version"]
            snapshot["last_summary_msg_id"] = outcome["through_message_id"]
            if not outcome["more"]:
                break

        if committed:
            snapshot["updated_at"] = datetime.now(timezone.utc)
            await self.memory_repo.save_snapshot(snapshot)
            await self.memory_repo.invalidate_conversation(conversation_id)
        else:
            logger.info(f"No new messages after the summary watermark for '{conversation_id}'; no new version.")
        return snapshot

    async def _summarize_one_batch(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """One read -> model -> conditional write cycle. None = nothing new."""
        for attempt in range(1, MAX_CAS_ATTEMPTS + 1):
            row = self.cassandra_repo.get_summary(conversation_id) or {}
            expected_version = row.get("summary_version") if row else None
            previous_text = (row.get("summary_text") or "") if row else ""

            fetched = self.cassandra_repo.get_recent_messages(conversation_id, limit=FETCH_LIMIT)
            new, gap = select_new_messages(fetched, row.get("through_message_id"), row.get("through_created_at"))
            if gap:
                logger.warning(f"Possible message gap before the summary watermark for '{conversation_id}'.")
            if not new:
                return None
            batch = new[:BATCH_SIZE]

            response = await self.llm_service.summarize(SummarizeRequest(
                previous_summary=previous_text,
                new_messages=[LLMMessage(role=m["role"], content=m["content"]) for m in batch],
            ))
            text = validate_summary(response.summary)

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
                logger.info(
                    f"Committed summary v{record['summary_version']} for '{conversation_id}' "
                    f"({len(batch)} new message(s))."
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
