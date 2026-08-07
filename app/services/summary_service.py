"""
app/services/summary_service.py

Incremental Summarization Service coordinates versioned conversation summarization
using LLM gRPC calls bounded by the previous summary plus the latest message window.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.core.config import settings
from app.repositories.cassandra_repository import CassandraRepository
from app.repositories.memory_repository import MemoryRepository
from app.services.llm_service import LLMService
from app.schemas.llm import SummarizeRequest, LLMMessage

logger = logging.getLogger("memory_service.services.summary_service")


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
        """
        Runs the Incremental Summarization Algorithm:
        1. Fetches previous summary and the latest 20 message window.
        2. Re-orders messages to chronological (oldest first).
        3. Invokes LLM gRPC service under a circuit breaker to produce the new summary.
        4. Writes the updated version to Cassandra and evicts the summary cache in Redis.
        """
        # 1. Fetch current snapshot metadata
        snapshot = await self.memory_repo.get_snapshot(conversation_id)
        if not snapshot:
            raise ValueError(f"Snapshot not found for conversation '{conversation_id}'.")

        # 2. Fetch previous summary (defaults to empty if none exists yet)
        prev_summary = await self.memory_repo.get_summary(conversation_id) or ""

        # 3. Fetch latest sliding window of messages
        messages = await self.memory_repo.get_recent_messages(conversation_id, limit=20)
        if not messages:
            logger.warning(f"No recent messages found for conversation '{conversation_id}'. Skipping summarization.")
            return snapshot

        # Reverse the window list to be in chronological order (oldest first)
        chronological_messages = list(reversed(messages))

        # 4. Invoke LLM Service
        llm_messages = [
            LLMMessage(role=msg["role"], content=msg["content"])
            for msg in chronological_messages
        ]
        summarize_request = SummarizeRequest(
            previous_summary=prev_summary,
            new_messages=llm_messages
        )

        logger.info(f"Invoking internal LLM service for incremental summary of conversation '{conversation_id}'.")
        summarize_response = await self.llm_service.summarize(summarize_request)
        new_summary_text = summarize_response.summary

        # 5. Write the versioned summary record back to Cassandra
        next_version = snapshot["summary_version"] + 1
        summary_record = {
            "conversation_id": conversation_id,
            "summary_text": new_summary_text,
            "summary_version": next_version,
            "model_name": settings.EMBEDDING_MODEL_NAME,
            "model_version": settings.EMBEDDING_MODEL_VERSION,
            "generated_at": datetime.now(timezone.utc)
        }
        self.cassandra_repo.upsert_summary(summary_record)
        logger.info(f"Saved new summary version {next_version} for conversation '{conversation_id}' in Cassandra.")

        # 6. Update the snapshot summary version watermark
        snapshot["summary_version"] = next_version
        snapshot["last_summary_msg_id"] = chronological_messages[-1]["message_id"]
        snapshot["updated_at"] = datetime.now(timezone.utc)
        await self.memory_repo.save_snapshot(snapshot)

        # 7. Evict cache to force read-through hydration on the next retrieve call
        await self.memory_repo.invalidate_conversation(conversation_id)

        return snapshot
