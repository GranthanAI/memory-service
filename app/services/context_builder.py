"""
app/services/context_builder.py

Structured Context Builder gathers short-term sliding message windows,
conversation summaries, ancestor lineages, and scored long-term facts concurrently.
"""

import asyncio
import logging
import uuid
from typing import Any, Dict, List, Optional

from app.core.config import settings
from app.core.metrics import CONTEXT_COVERED_SKIPPED, CONTEXT_GAP_CLOSED, SUMMARY_STALE
from app.services.summary_service import _as_utc, message_key
from app.clients.graph_client import GraphClient
from app.services.retrieval_service import RetrievalService

logger = logging.getLogger("memory_service.services.context_builder")


def summary_valid_at_branch(through_message_id: Optional[str], upto_message_id: Optional[str]) -> bool:
    """True when a summary incorporates no message after the branch point.

    Message ids are UUIDv7 (time-ordered, compared as 128-bit numbers). Anything unknown or unparsable is
    treated as NOT valid: a summary whose coverage cannot be proven to end at the branch point must not be
    shown to a child, because it could contain what the parent said afterwards."""
    try:
        return uuid.UUID(str(through_message_id)).int <= uuid.UUID(str(upto_message_id)).int
    except (ValueError, TypeError, AttributeError):
        return False


class ContextBuilder:
    """
    Assembles contextual memory payloads to enrich LLM prompts.
    Coordinates concurrent fetching and handles Graph Service timeouts gracefully.
    """

    def __init__(self, retrieval_service: RetrievalService, graph_client: GraphClient, scheduler=None, history_search=None):
        self.retrieval_service = retrieval_service
        self.graph_client = graph_client
        # Optional collaborators (older callers/tests build the context without them):
        self.scheduler = scheduler            # CompactionScheduler: asked for a summary when the existing one is stale
        self.history_search = history_search  # HistorySearch: recovers older original messages for the question

    async def get_parent_summaries(self, conversation_id: str) -> tuple[List[Dict[str, Any]], bool]:
        """
        Calls Graph Service to fetch ancestor summaries, wrapped in a strict timeout.
        Returns a tuple of (ancestor_summaries_list, is_available_flag).
        """
        timeout_seconds = settings.GRAPH_SERVICE_TIMEOUT_MS / 1000.0
        try:
            async with asyncio.timeout(timeout_seconds):
                ancestors = await self.graph_client.get_ancestors(conversation_id)
                return ancestors, True
        except (asyncio.TimeoutError, Exception) as e:
            logger.warning(
                f"Graph Service unavailable for {conversation_id}: {e}. "
                f"Falling back to current summary only."
            )
            return [], False

    def _summary_meta(self, conversation_id: str) -> Dict[str, Any]:
        try:
            row = self.retrieval_service.memory_repo.cassandra_repo.get_summary(conversation_id) or {}
        except Exception:
            return {}
        return {"summary_version": row.get("summary_version"), "through_message_id": row.get("through_message_id"),
                "through_created_at": row.get("through_created_at")}

    async def ancestor_summaries(
        self, user_id: str, ancestors: Optional[List[Dict[str, str]]]
    ) -> List[Dict[str, Any]]:
        """Branch-valid summaries of the ancestors (root first). `ancestors` is [{conversation_id,
        upto_message_id}] from conversation-service. An ancestor is skipped (never guessed) when it belongs
        to another user, has no summary, or its summary may include messages after the branch point."""
        out: List[Dict[str, Any]] = []
        repo = self.retrieval_service.memory_repo
        for ref in ancestors or []:
            cid, upto = str(ref.get("conversation_id") or ""), ref.get("upto_message_id")
            if not cid or not upto:
                continue
            try:
                snapshot = await repo.get_snapshot(cid)
                if not snapshot or not user_id or snapshot.get("user_id") != user_id:
                    continue
                row = repo.cassandra_repo.get_summary(cid)
            except Exception as exc:
                logger.warning(f"Ancestor summary unavailable for {cid}: {exc}")
                continue
            text = (row or {}).get("summary_text") or ""
            if not text.strip():
                continue
            if not summary_valid_at_branch(row.get("through_message_id"), upto):
                logger.info(f"Ancestor summary for {cid} omitted: not provably valid at the branch point.")
                continue
            out.append({
                "conversation_id": cid,
                "summary": text,
                "covers_through_message_id": row["through_message_id"],
                "summary_version": int(row.get("summary_version") or 0),
            })
        return out

    async def build_context(
        self,
        user_id: str,
        conversation_id: str,
        query_vector: Optional[List[float]] = None,
        ancestors: Optional[List[Dict[str, str]]] = None,
        use_graph_parents: bool = True,
        query_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Concurrently aggregates short-term recent messages, the active summary,
        graph ancestor summaries, and relevant long-term semantic facts.
        """
        # Define tasks for concurrent execution
        snapshot_task = self.retrieval_service.get_or_hydrate_snapshot(conversation_id)
        summary_task = self.retrieval_service.get_or_hydrate_summary(conversation_id)
        messages_task = self.retrieval_service.get_or_hydrate_recent_messages(
            conversation_id,
            limit=settings.SHORT_TERM_MESSAGE_LIMIT
        )
        # Graph summaries describe the parent as it is NOW, not at the branch point, so callers that supply the
        # branch lineage (gRPC) do not use them; they get branch-valid ancestor summaries instead.
        parents_task = self.get_parent_summaries(conversation_id) if use_graph_parents else asyncio.sleep(0, result=([], False))

        if query_vector:
            facts_task = self.retrieval_service.retrieve_relevant_facts(
                user_id=user_id,
                query_vector=query_vector,
                limit=settings.RETRIEVAL_TOP_K_FACTS
            )
        else:
            facts_task = asyncio.sleep(0, result=[])

        # Execute concurrently
        snapshot, summary, messages, parents_result, facts = await asyncio.gather(
            snapshot_task,
            summary_task,
            messages_task,
            parents_task,
            facts_task
        )

        parent_summaries, parents_available = parents_result
        branch_summaries = await self.ancestor_summaries(user_id, ancestors) if ancestors else []

        # Resolve user_id if not passed initially
        resolved_user_id = user_id
        if snapshot:
            if not resolved_user_id:
                resolved_user_id = snapshot.get("user_id")
            elif snapshot.get("user_id") and snapshot.get("user_id") != resolved_user_id:
                # The conversation belongs to someone else: return nothing private
                # (previously this was only logged and the other user's messages,
                # summary and facts were returned).
                logger.warning(
                    f"User ID mismatch for conversation {conversation_id}: "
                    f"snapshot user_id={snapshot.get('user_id')}, requested user_id={resolved_user_id}; denying"
                )
                return {
                    "conversation_id": conversation_id,
                    "user_id": resolved_user_id,
                    "current_summary": "",
                    "short_term_messages": [],
                    "parent_summaries": [],
                    "relevant_facts": [],
                    "metadata": {"parent_summaries_available": False, "facts_retrieved_count": 0, "denied": True},
                }

        # Recent turns: chronological, without what the summary already covers, with no hole before the unsummarised part.
        summary_meta = self._summary_meta(conversation_id) if summary else {}
        messages, window_info = await self._assemble_messages(
            user_id=resolved_user_id or user_id, conversation_id=conversation_id, window=messages,
            summary=summary or "", summary_meta=summary_meta)
        retrieved = await self._older_messages(conversation_id, query_text, messages)

        # Assemble final context response payload
        context_payload = {
            "conversation_id": conversation_id,
            "user_id": resolved_user_id,
            "current_summary": summary or "",
            "short_term_messages": messages,
            "retrieved_history": retrieved,
            "parent_summaries": parent_summaries,
            "ancestor_summaries": branch_summaries,
            "summary_meta": summary_meta,
            "relevant_facts": facts,
            "metadata": {
                "parent_summaries_available": parents_available,
                "facts_retrieved_count": len(facts),
                **window_info,
            }
        }

        return context_payload

    # ─── conversation-memory assembly ─────────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _watermark(summary_meta: Dict[str, Any], ordered: List[Dict[str, Any]]):
        """The summary's coverage boundary as a message key, or None when there is no provable one."""
        through_id = summary_meta.get("through_message_id")
        if not through_id:
            return None
        created = _as_utc(summary_meta.get("through_created_at"))
        if created is None:                                    # legacy row: find the boundary message in the window
            for m in ordered:
                if str(m.get("message_id")) == str(through_id):
                    return message_key(m)
            return None
        return created, str(through_id)

    async def _assemble_messages(self, *, user_id: str, conversation_id: str, window: List[Dict[str, Any]],
                                 summary: str, summary_meta: Dict[str, Any]):
        """Returns (messages oldest-first, info).

        The stores keep the sliding window NEWEST first; consumers read history oldest-first, so it is ordered here, once.
        With a summary, messages it already covers are not sent again (the summary represents them) except a short verbatim
        tail for continuity. Messages after the watermark are always included; if the window starts after the watermark,
        the gap is read back from the durable rows, so nothing falls between the summary and the window."""
        ordered = sorted(window, key=message_key)
        info = {"summary_covers_through": summary_meta.get("through_message_id") or "", "window_messages": len(ordered),
                "gap_closed": False, "unsummarized_messages": 0, "stale": False, "covered_skipped": 0}
        mark = self._watermark(summary_meta, ordered) if summary else None
        if mark is None:
            return ordered, info

        uncovered = [m for m in ordered if message_key(m) > mark]
        window_full = len(ordered) >= settings.SHORT_TERM_MESSAGE_LIMIT
        if window_full and message_key(ordered[0]) > mark:
            # Neither the summary nor the window has what happened in between: read it from the durable rows (bounded).
            try:
                durable = self.retrieval_service.memory_repo.cassandra_repo.get_recent_messages(
                    conversation_id, limit=settings.CONTEXT_MAX_UNSUMMARIZED_MESSAGES)
            except Exception as exc:
                logger.warning(f"Could not read the unsummarised messages back ({type(exc).__name__})")
                durable = []
            after = sorted((m for m in durable if message_key(m) > mark), key=message_key)
            if len(after) > len(uncovered):
                uncovered = after
                info["gap_closed"] = True
                CONTEXT_GAP_CLOSED.inc()
        tail = ordered[-settings.CONTEXT_MIN_VERBATIM_MESSAGES:]
        keep: Dict[str, Dict[str, Any]] = {str(m.get("message_id")): m for m in [*uncovered, *tail]}
        messages = sorted(keep.values(), key=message_key)
        window_ids = {str(m.get("message_id")) for m in ordered}
        info["unsummarized_messages"] = len(uncovered)
        info["covered_skipped"] = len(window_ids - set(keep))
        if info["covered_skipped"]:
            CONTEXT_COVERED_SKIPPED.inc(info["covered_skipped"])
        if len(uncovered) >= settings.CONTEXT_STALE_MESSAGES:
            info["stale"] = True
            SUMMARY_STALE.inc()
            if self.scheduler is not None:
                try:
                    await self.scheduler.request_from_read(
                        conversation_id, user_id, int(summary_meta.get("summary_version") or 0))
                except Exception as exc:
                    logger.warning(f"Could not request a summary update ({type(exc).__name__})")
        return messages, info

    async def _older_messages(self, conversation_id: str, query_text: Optional[str],
                              shown: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Older ORIGINAL messages of this conversation that the question needs and the summary may have lost."""
        if self.history_search is None or not query_text:
            return []
        older_than = message_key(shown[0]) if shown else None
        exclude = {str(m.get("message_id")) for m in shown}
        return await self.history_search.find(conversation_id, query_text, exclude, older_than)
