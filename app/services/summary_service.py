"""
app/services/summary_service.py

Incremental Summarization Service coordinates versioned conversation summarization
using LLM gRPC calls bounded by the previous summary plus the latest message window.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.clients.llm_client import LLMClient
from app.core.config import settings
from app.proto import llm_pb2, llm_pb2_grpc
from app.repositories.cassandra_repository import CassandraRepository
from app.repositories.memory_repository import MemoryRepository

logger = logging.getLogger("memory_service.services.summary_service")


class SummaryService:
    """
    Coordinates versioned conversation summaries incrementally.
    """

    def __init__(
        self,
        memory_repo: MemoryRepository,
        cassandra_repo: CassandraRepository,
        llm_client: LLMClient
    ):
        self.memory_repo = memory_repo
        self.cassandra_repo = cassandra_repo
        self.llm_client = llm_client

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

        # 4. Invoke LLM gRPC Service via Circuit Breaker
        default_instructions = (
            "Integrate the new messages into an updated summary. "
            "Preserve key facts from the previous summary. Be concise."
        )
        final_instructions = instructions or default_instructions

        # Format message payloads for json serialization
        messages_payload = [
            {
                "message_id": msg["message_id"],
                "role": msg["role"],
                "content": msg["content"],
                "created_at": msg.get("created_at").isoformat() if isinstance(msg.get("created_at"), datetime) else str(msg.get("created_at"))
            }
            for msg in chronological_messages
        ]

        async def summary_stub(channel) -> str:
            stub = llm_pb2_grpc.LLMServiceStub(channel)
            request = llm_pb2.SummaryRequest(
                previous_summary=prev_summary,
                new_messages_json=json.dumps(messages_payload),
                instructions=final_instructions
            )
            response = await stub.GenerateSummary(
                request,
                timeout=settings.GRPC_TIMEOUT_SECONDS
            )
            return response.summary_text

        logger.info(f"Invoking LLM gRPC service for incremental summary of conversation '{conversation_id}'.")
        new_summary_text = await self.llm_client.call_with_circuit_breaker(summary_stub)

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
