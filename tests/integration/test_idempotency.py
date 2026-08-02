"""
tests/integration/test_idempotency.py

Integration tests for Phase 6: Event Idempotency Service.
Uses real connection pools to verify Cassandra processed_events table and
deduplication exception assertions.
"""

import asyncio
import uuid
import pytest
from app.db.session import initialize_db_sessions, close_db_sessions
from app.db.cassandra import get_session
from app.repositories.processed_event_repository import ProcessedEventRepository
from app.services.idempotency_service import IdempotencyService
from app.core.exceptions import DeduplicationException


@pytest.fixture(scope="module", autouse=True)
def setup_integration_db():
    """
    Initializes real database connection sessions on startup and closes them on teardown.
    """
    loop = asyncio.get_event_loop_policy().new_event_loop()
    loop.run_until_complete(initialize_db_sessions())
    yield
    loop.run_until_complete(close_db_sessions())
    loop.close()


@pytest.fixture
def clean_idempotency_table():
    """
    Truncates the processed_events table before each test run.
    """
    session = get_session()
    session.execute("TRUNCATE processed_events")
    yield


def test_idempotency_service_flow(clean_idempotency_table):
    """
    Verifies that checking a new event registers it successfully,
    and attempting to register the same event_id a second time throws a DeduplicationException.
    """
    session = get_session()
    processed_event_repo = ProcessedEventRepository(session)
    idempotency_service = IdempotencyService(processed_event_repo)

    event_id = f"event-{uuid.uuid4()}"
    conversation_id = "conv-999"

    # 1. First invocation should process and register cleanly
    idempotency_service.check_and_register(event_id, conversation_id)

    # Verify that it is written to the database
    rows = list(session.execute(
        "SELECT event_id, conversation_id FROM processed_events WHERE event_id = %s",
        (event_id,)
    ))
    assert len(rows) == 1
    assert rows[0].event_id == event_id
    assert rows[0].conversation_id == conversation_id

    # 2. Second invocation with same event_id must raise DeduplicationException
    with pytest.raises(DeduplicationException) as exc_info:
        idempotency_service.check_and_register(event_id, conversation_id)

    assert event_id in str(exc_info.value)
