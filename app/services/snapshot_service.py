"""
app/services/snapshot_service.py

Snapshot Builder Service handles atomic snapshot commits and outbox job dispatches.
All writes are sequenced in a single Cassandra Logged Batch to guarantee atomicity.
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from cassandra.cluster import Session
from cassandra.query import BatchStatement, BatchType

from app.repositories.redis_repository import RedisRepository

logger = logging.getLogger("memory_service.services.snapshot_service")


class SnapshotService:
    """
    Coordinates atomic batch updates for conversation metadata, recent message windows,
    event idempotency registration, and outbox job scheduling in Cassandra.
    After successful commit, invalidates the hot cache in Redis.
    """

    def __init__(self, cassandra_session: Session, redis_repo: RedisRepository):
        self.session = cassandra_session
        self.redis_repo = redis_repo
        self._prepare_statements()

    def _prepare_statements(self) -> None:
        # 1. Snapshot metadata upsert
        self._snap_upsert = self.session.prepare("""
            INSERT INTO conversation_snapshots (
                conversation_id, user_id, message_count, state,
                summary_version, fact_version, snapshot_version,
                last_summary_msg_id, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """)

        # 2. Message append to sliding window
        self._recent_msg_append = self.session.prepare("""
            INSERT INTO conversation_recent_messages (
                conversation_id, created_at, message_id, role, content
            ) VALUES (?, ?, ?, ?, ?)
        """)

        # 3. Idempotency record insertion (7-day TTL is handled by default table TTL)
        self._idemp_insert = self.session.prepare("""
            INSERT INTO processed_events (
                event_id, conversation_id, processed_at
            ) VALUES (?, ?, ?)
        """)

        # 4. Outbox job scheduling
        self._outbox_insert = self.session.prepare("""
            INSERT INTO outbox_jobs (
                status, created_at, job_id, topic, conversation_id,
                payload, attempt_count, last_error, claimed_at
            ) VALUES ('PENDING', ?, ?, ?, ?, ?, 0, NULL, NULL)
        """)

    def commit_snapshot_and_outbox(
        self,
        snapshot: Dict[str, Any],
        event_id: str,
        outbox_topic: Optional[str] = None,
        outbox_payload: Optional[Dict[str, Any]] = None,
        message: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Executes a Cassandra Logged Batch containing snapshot metadata, recent message,
        idempotency marker, and outbox job. Guaranteed to be atomic.
        """
        now = datetime.now(timezone.utc)
        batch = BatchStatement(batch_type=BatchType.LOGGED)

        # 1. Add Snapshot Metadata Upsert
        batch.add(self._snap_upsert, (
            snapshot["conversation_id"],
            snapshot["user_id"],
            snapshot["message_count"],
            snapshot["state"],
            snapshot["summary_version"],
            snapshot["fact_version"],
            snapshot["snapshot_version"],
            snapshot.get("last_summary_msg_id"),
            now
        ))

        # 2. Add Recent Message Append (if present)
        if message:
            # Parse message fields
            msg_id = message["message_id"]
            role = message["role"]
            content = message["content"]
            # Enforce timezone-aware or default datetime
            msg_created_at = message.get("created_at") or now
            if isinstance(msg_created_at, str):
                msg_created_at = datetime.fromisoformat(msg_created_at)
            
            batch.add(self._recent_msg_append, (
                snapshot["conversation_id"],
                msg_created_at,
                msg_id,
                role,
                content
            ))

        # 3. Add Idempotency Marker
        batch.add(self._idemp_insert, (
            event_id,
            snapshot["conversation_id"],
            now
        ))

        # 4. Add Outbox Job Creation
        if outbox_topic and outbox_payload is not None:
            job_id = uuid.uuid4()
            serialized_payload = json.dumps(outbox_payload)
            batch.add(self._outbox_insert, (
                now,
                job_id,
                outbox_topic,
                snapshot["conversation_id"],
                serialized_payload
            ))

        # Execute the logged batch synchronously
        self.session.execute(batch)
        logger.info(
            f"Successfully committed snapshot batch for conversation {snapshot['conversation_id']}."
        )


    async def post_commit_invalidation(self, conversation_id: str) -> None:
        """
        Deletes the conversation's hot cache keys from Redis.
        Invoked after commit_snapshot_and_outbox to enforce read-through cache consistency.
        """
        try:
            await self.redis_repo.invalidate_conversation(conversation_id)
            logger.info(f"Cache invalidated for conversation: {conversation_id}")
        except Exception as e:
            logger.error(f"Error invalidating cache for conversation {conversation_id} post-commit: {e}")
