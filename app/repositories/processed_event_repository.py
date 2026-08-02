"""
app/repositories/processed_event_repository.py

Repository for checking and recording processed events in Cassandra for deduplication.
"""

import logging
from datetime import datetime, timezone
from cassandra.cluster import Session

logger = logging.getLogger("memory_service.repositories.processed_event_repository")


class ProcessedEventRepository:
    """
    Manages persistence and query access to the processed_events table in Cassandra.
    Ensures idempotency checks across event consumers.
    """

    def __init__(self, session: Session):
        self.session = session
        self._prepare_statements()

    def _prepare_statements(self):
        self._select_stmt = self.session.prepare("""
            SELECT event_id FROM processed_events WHERE event_id = ?
        """)
        self._insert_stmt = self.session.prepare("""
            INSERT INTO processed_events (event_id, conversation_id, processed_at)
            VALUES (?, ?, ?)
        """)

    def is_event_processed(self, event_id: str) -> bool:
        """
        Checks if the given event_id has already been processed.
        """
        try:
            rows = self.session.execute(self._select_stmt, (event_id,))
            return bool(rows.one())
        except Exception as e:
            logger.error(f"Error checking processed event {event_id} in Cassandra: {e}")
            # In production, we fail safe or raise. Under at-least-once ingestion,
            # raising allows Kafka retries.
            raise

    def register_event(self, event_id: str, conversation_id: str) -> None:
        """
        Registers a processed event to prevent duplicate execution.
        Cassandra keyspace default TTL of 7 days will automatically apply.
        """
        try:
            now = datetime.now(timezone.utc)
            self.session.execute(self._insert_stmt, (event_id, conversation_id, now))
            logger.debug(f"Registered processed event '{event_id}' for conversation '{conversation_id}'.")
        except Exception as e:
            logger.error(f"Error registering processed event {event_id} in Cassandra: {e}")
            raise
