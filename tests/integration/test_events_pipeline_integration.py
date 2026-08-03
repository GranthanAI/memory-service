"""
tests/integration/test_events_pipeline_integration.py

Integration tests for Phase 18 Event Ingestor and Retry Scheduler.
Verifies real Cassandra database mutations, idempotency gates,
recent message serialization, and DLQ execution.
"""

import asyncio
import json
import uuid
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock

from app.db.cassandra import get_session
from app.db.session import initialize_db_sessions, close_db_sessions
from app.models.memory import MemoryState
from app.repositories.cassandra_repository import CassandraRepository
from app.repositories.processed_event_repository import ProcessedEventRepository
from app.repositories.redis_repository import RedisRepository
from app.repositories.memory_repository import MemoryRepository
from app.services.snapshot_service import SnapshotService
from app.events.dispatcher import EventDispatcher
from app.events.retry_scheduler import RetryScheduler


def run_async(coro):
    """Helper to run async coroutines in a consistent event loop."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


@pytest.fixture(scope="module", autouse=True)
def setup_integration_db():
    """Initializes and tears down database sessions."""
    run_async(initialize_db_sessions())
    yield
    run_async(close_db_sessions())


@pytest.fixture
def clean_pipeline_tables():
    """Clears relevant Cassandra tables before and after tests."""
    session = get_session()
    
    tables = [
        "processed_events",
        "conversation_snapshots",
        "conversation_recent_messages",
        "outbox_jobs",
        "retry_jobs"
    ]
    for table in tables:
        session.execute(f"TRUNCATE {table}")
        
    yield
    
    for table in tables:
        session.execute(f"TRUNCATE {table}")


@pytest.mark.asyncio
async def test_event_dispatcher_and_idempotency_integration(clean_pipeline_tables):
    """
    Verifies that EventDispatcher creates snapshots, updates sliding messages,
    and deduplicates replayed events in a live Cassandra database.
    """
    session = get_session()
    cassandra_repo = CassandraRepository(session)
    processed_event_repo = ProcessedEventRepository(session)
    # Using mock redis for test integration simplicity
    redis_repo = MagicMock(spec=RedisRepository)
    redis_repo.get_snapshot = AsyncMock(return_value=None)
    redis_repo.set_snapshot = AsyncMock()
    redis_repo.get_recent_messages = AsyncMock(return_value=None)
    redis_repo.set_recent_messages = AsyncMock()
    redis_repo.invalidate_conversation = AsyncMock()
    
    memory_repo = MemoryRepository(cassandra_repo, redis_repo)
    snapshot_service = SnapshotService(session, redis_repo)

    # Set threshold to 2 to verify outbox triggers
    dispatcher = EventDispatcher(
        processed_event_repo=processed_event_repo,
        memory_repo=memory_repo,
        snapshot_service=snapshot_service,
        summary_threshold=2
    )

    conversation_id = f"conv-{uuid.uuid4()}"
    event_id = f"evt-{uuid.uuid4()}"

    raw_event = {
        "event_id": event_id,
        "event_type": "chat.message.created",
        "conversation_id": conversation_id,
        "user_id": "user-789",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": {
            "message_id": f"msg-{uuid.uuid4()}",
            "role": "user",
            "content": "First integration message",
            "created_at": datetime.now(timezone.utc).isoformat()
        }
    }

    # 1. Dispatch first event
    await dispatcher.dispatch(raw_event)

    # Verify Snapshot creation
    snap = await memory_repo.get_snapshot(conversation_id)
    assert snap is not None
    assert snap["message_count"] == 1
    assert snap["state"] == MemoryState.ACTIVE.value

    # Verify message appended
    messages = await memory_repo.get_recent_messages(conversation_id, limit=5)
    assert len(messages) == 1
    assert messages[0]["content"] == "First integration message"

    # Verify event ID registered in Cassandra
    assert processed_event_repo.is_event_processed(event_id) is True

    # 2. Dispatch the exact same event again (duplicate replay)
    await dispatcher.dispatch(raw_event)

    # Message count and snapshot version must remain unchanged
    snap_after = await memory_repo.get_snapshot(conversation_id)
    assert snap_after["message_count"] == 1
    assert snap_after["snapshot_version"] == snap["snapshot_version"]

    # Recent messages list size must remain 1
    messages_after = await memory_repo.get_recent_messages(conversation_id, limit=5)
    assert len(messages_after) == 1

    # 3. Dispatch second event to trigger SUMMARY_PENDING
    event_id_2 = f"evt-{uuid.uuid4()}"
    raw_event_2 = {
        "event_id": event_id_2,
        "event_type": "chat.response.completed",
        "conversation_id": conversation_id,
        "user_id": "user-789",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": {
            "message_id": f"msg-{uuid.uuid4()}",
            "role": "assistant",
            "content": "Assistant replies",
            "created_at": datetime.now(timezone.utc).isoformat()
        }
    }

    await dispatcher.dispatch(raw_event_2)

    # Verify count is 2 and state transitioned to SUMMARY_PENDING
    snap_final = await memory_repo.get_snapshot(conversation_id)
    assert snap_final["message_count"] == 2
    assert snap_final["state"] == MemoryState.SUMMARY_PENDING.value

    # Verify outbox job was created
    outbox_jobs = cassandra_repo.get_pending_outbox_jobs(limit=10)
    assert len(outbox_jobs) == 1
    assert outbox_jobs[0]["topic"] == "memory.summary.request"


@pytest.mark.asyncio
async def test_retry_scheduler_dlq_integration(clean_pipeline_tables):
    """
    Verifies that RetryScheduler correctly processes retry limits,
    forwards tasks to DLQ, and commits FAILED rows in Cassandra.
    """
    session = get_session()
    cassandra_repo = CassandraRepository(session)
    producer = AsyncMock()

    scheduler = RetryScheduler(session, producer)

    # Seed a job with retry_count = max_retry
    job_id = uuid.uuid4()
    next_retry = datetime.now(timezone.utc) - timedelta(seconds=1)
    
    job = {
        "status": "PENDING",
        "next_retry": next_retry,
        "job_id": job_id,
        "job_type": "fact",
        "payload": json.dumps({"conversation_id": "conv-dlq", "text": "expired retry"}),
        "retry_count": 3,
        "max_retry": 3,
        "last_error": "Milvus disconnected",
        "created_at": datetime.now(timezone.utc) - timedelta(minutes=5)
    }

    cassandra_repo.insert_retry_job(job)

    # Poll and run scheduler
    await scheduler._process_retries()

    # Verify Kafka producer called with DLQ topic
    producer.publish_task.assert_called_once()
    args, kwargs = producer.publish_task.call_args
    assert kwargs["topic"] == "memory.dlq"
    assert kwargs["conversation_id"] == "conv-dlq"

    # Verify original PROCESSING job is deleted
    processing_rows = list(session.execute(
        "SELECT job_id FROM retry_jobs WHERE status = 'PROCESSING'"
    ))
    assert len(processing_rows) == 0

    # Verify FAILED row is written
    failed_rows = list(session.execute(
        "SELECT job_id, last_error FROM retry_jobs WHERE status = 'FAILED'"
    ))
    assert len(failed_rows) == 1
    assert failed_rows[0].job_id == job_id
    assert "Max retries exhausted" in failed_rows[0].last_error
