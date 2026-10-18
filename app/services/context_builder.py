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

    def __init__(self, retrieval_service: RetrievalService, graph_client: GraphClient):
        self.retrieval_service = retrieval_service
        self.graph_client = graph_client

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
        return {"summary_version": row.get("summary_version"), "through_message_id": row.get("through_message_id")}

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

        # Assemble final context response payload
        context_payload = {
            "conversation_id": conversation_id,
            "user_id": resolved_user_id,
            "current_summary": summary or "",
            "short_term_messages": messages,
            "parent_summaries": parent_summaries,
            "ancestor_summaries": branch_summaries,
            "summary_meta": self._summary_meta(conversation_id) if summary else {},
            "relevant_facts": facts,
            "metadata": {
                "parent_summaries_available": parents_available,
                "facts_retrieved_count": len(facts)
            }
        }

        return context_payload
