"""
app/services/context_builder.py

Structured Context Builder gathers short-term sliding message windows,
conversation summaries, ancestor lineages, and scored long-term facts concurrently.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

from app.core.config import settings
from app.clients.graph_client import GraphClient
from app.services.retrieval_service import RetrievalService

logger = logging.getLogger("memory_service.services.context_builder")


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

    async def build_context(
        self,
        user_id: str,
        conversation_id: str,
        query_vector: Optional[List[float]] = None
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
        parents_task = self.get_parent_summaries(conversation_id)

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

        # Resolve user_id if not passed initially
        resolved_user_id = user_id
        if snapshot:
            if not resolved_user_id:
                resolved_user_id = snapshot.get("user_id")
            elif snapshot.get("user_id") != resolved_user_id:
                logger.warning(
                    f"User ID mismatch for conversation {conversation_id}: "
                    f"snapshot user_id={snapshot.get('user_id')}, requested user_id={resolved_user_id}"
                )

        # Assemble final context response payload
        context_payload = {
            "conversation_id": conversation_id,
            "user_id": resolved_user_id,
            "current_summary": summary or "",
            "short_term_messages": messages,
            "parent_summaries": parent_summaries,
            "relevant_facts": facts,
            "metadata": {
                "parent_summaries_available": parents_available,
                "facts_retrieved_count": len(facts)
            }
        }

        return context_payload
