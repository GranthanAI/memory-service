"""
app/services/idempotency_service.py

Service to enforce event deduplication and ensure exactly-once processing guarantees.
"""

import logging
from app.repositories.processed_event_repository import ProcessedEventRepository
from app.core.exceptions import DeduplicationException

logger = logging.getLogger("memory_service.services.idempotency_service")


class IdempotencyService:
    """
    Coordinates event idempotency checks.
    Throws DeduplicationException if an event was already processed.
    """

    def __init__(self, processed_event_repo: ProcessedEventRepository):
        self.processed_event_repo = processed_event_repo

    def check_and_register(self, event_id: str, conversation_id: str) -> None:
        """
        Queries Cassandra for the event_id. If it exists, raises DeduplicationException.
        If it does not exist, registers the event to declare it as processed.

        NOTE: For atomic state updates, the registration should ideally be bundled
        in the Cassandra logged batch. This method serves as the standalone gating service.
        """
        if self.processed_event_repo.is_event_processed(event_id):
            logger.warning(f"Duplicate event detected and dropped: {event_id}")
            raise DeduplicationException(event_id)

        self.processed_event_repo.register_event(event_id, conversation_id)
