"""
app/events/dispatcher.py

EventDispatcher parses raw incoming event payloads, checks event idempotency,
and orchestrates the update of conversation snapshots and message logs.
"""

import logging
from typing import Any, Dict
from pydantic import ValidationError

from app.models.memory import MemoryState
from app.schemas.events import MemoryEventEnvelope
from app.repositories.processed_event_repository import ProcessedEventRepository
from app.repositories.memory_repository import MemoryRepository
from app.services.snapshot_service import SnapshotService
from app.core.config import settings

logger = logging.getLogger("memory_service.events.dispatcher")


class EventDispatcher:
    """
    Validates received Kafka messages, enforces idempotency checks, and initiates
    atomic Cassandra batch transactions to append messages and evolve snapshot states.
    """

    def __init__(
        self,
        processed_event_repo: ProcessedEventRepository,
        memory_repo: MemoryRepository,
        snapshot_service: SnapshotService,
        summary_threshold: int = 20
    ):
        self.processed_event_repo = processed_event_repo
        self.memory_repo = memory_repo
        self.snapshot_service = snapshot_service
        self.summary_threshold = summary_threshold

    async def dispatch(self, raw_event: Dict[str, Any]) -> None:
        """
        Processes a raw incoming event: validates schemas, applies deduplication filters,
        updates states, and executes the Cassandra Logged Batch.
        """
        try:
            envelope = MemoryEventEnvelope.model_validate(raw_event)
        except ValidationError as e:
            logger.error(f"Event payload validation failed: {e}")
            raise e

        event_id = envelope.event_id
        conversation_id = envelope.conversation_id

        # 1. Idempotency Check
        if self.processed_event_repo.is_event_processed(event_id):
            logger.info(f"Duplicate event skipped: {event_id}")
            return

        # 2. Retrieve existing snapshot or initialize active defaults
        snapshot = await self.memory_repo.get_snapshot(conversation_id)
        if not snapshot:
            snapshot = {
                "conversation_id": conversation_id,
                "user_id": envelope.user_id,
                "message_count": 0,
                "state": MemoryState.ACTIVE.value,
                "summary_version": 0,
                "fact_version": 0,
                "snapshot_version": 0,
                "last_summary_msg_id": None
            }

        # Increment counters
        snapshot["message_count"] += 1
        snapshot["snapshot_version"] += 1

        # Extract message metadata
        message = {
            "message_id": envelope.payload.message_id,
            "role": envelope.payload.role,
            "content": envelope.payload.content,
            "created_at": envelope.payload.created_at
        }

        # Determine transition state and target outbox job configurations
        outbox_topic = None
        outbox_payload = None

        if snapshot["message_count"] % self.summary_threshold == 0:
            snapshot["state"] = MemoryState.SUMMARY_PENDING.value
            outbox_topic = settings.KAFKA_SUMMARY_TOPIC
            outbox_payload = {
                "conversation_id": conversation_id,
                "user_id": envelope.user_id,
                "version": snapshot["summary_version"]
            }
        else:
            # Fall back to ACTIVE if it was READY before a new message arrives
            if snapshot["state"] == MemoryState.READY.value:
                snapshot["state"] = MemoryState.ACTIVE.value

        # 3. Atomically commit the updates inside a Cassandra logged batch
        self.snapshot_service.commit_snapshot_and_outbox(
            snapshot=snapshot,
            event_id=event_id,
            outbox_topic=outbox_topic,
            outbox_payload=outbox_payload,
            message=message
        )

        # 4. Post-commit Redis cache eviction to force read-through refresh
        await self.snapshot_service.post_commit_invalidation(conversation_id)
