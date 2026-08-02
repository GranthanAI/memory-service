"""
tests/integration/test_snapshot_service_integration.py

Integration tests for Phase 9: Snapshot Builder Service.
Verifies Cassandra Logged Batch atomicity and post-commit cache invalidation on live containers.
"""

import asyncio
import uuid
from datetime import datetime, timezone, timedelta
import pytest

from app.db.session import initialize_db_sessions, close_db_sessions
from app.db.cassandra import get_session
from app.db.redis import get_redis_client
from app.repositories.redis_repository import RedisRepository
from app.repositories.cassandra_repository import CassandraRepository
from app.services.snapshot_service import SnapshotService


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
    """Initializes and tears down real database sessions for this test module."""
    run_async(initialize_db_sessions())
    yield
    run_async(close_db_sessions())


@pytest.fixture
def clean_tables():
    """Truncates Cassandra tables and flushes Redis before running tests."""
    session = get_session()
    tables = [
        "conversation_snapshots",
        "conversation_recent_messages",
        "processed_events",
        "outbox_jobs"
    ]
    for t in tables:
        session.execute(f"TRUNCATE {t}")
    
    redis_client = get_redis_client()
    run_async(redis_client.flushdb())
    yield


def test_commit_snapshot_and_outbox_integration(clean_tables):
    """
    Verifies that SnapshotService executes the Cassandra Logged Batch atomic mutation
    containing snapshot, recent message, idempotency record, and outbox job.
    """
    session = get_session()
    redis_client = get_redis_client()
    redis_repo = RedisRepository(redis_client)
    cassandra_repo = CassandraRepository(session)
    service = SnapshotService(session, redis_repo)

    conv_id = "conv-snap-service-test"
    user_id = "user-123"
    event_id = f"event-{uuid.uuid4()}"
    
    # 1. Snapshot metadata
    snapshot = {
        "conversation_id": conv_id,
        "user_id": user_id,
        "message_count": 1,
        "state": "ACTIVE",
        "summary_version": 1,
        "fact_version": 1,
        "snapshot_version": 1,
        "last_summary_msg_id": "msg-0"
    }

    # 2. Recent Message
    message = {
        "message_id": "msg-1",
        "role": "user",
        "content": "Hello, this is a test message.",
        "created_at": datetime.now(timezone.utc)
    }

    outbox_topic = "events.topic"
    outbox_payload = {"conversation_id": conv_id, "action": "test"}

    # 3. Commit the batch
    service.commit_snapshot_and_outbox(
        snapshot=snapshot,
        event_id=event_id,
        outbox_topic=outbox_topic,
        outbox_payload=outbox_payload,
        message=message
    )

    # 4. Assert Snapshot metadata was persisted
    snap_db = cassandra_repo.get_snapshot(conv_id)
    assert snap_db is not None
    assert snap_db["conversation_id"] == conv_id
    assert snap_db["user_id"] == user_id
    assert snap_db["message_count"] == 1

    # 5. Assert Recent Message was appended
    recent_msgs = cassandra_repo.get_recent_messages(conv_id, limit=5)
    assert len(recent_msgs) == 1
    assert recent_msgs[0]["message_id"] == "msg-1"
    assert recent_msgs[0]["content"] == "Hello, this is a test message."

    # 6. Assert Idempotency registration row was written
    rows = list(session.execute(
        "SELECT event_id, conversation_id FROM processed_events WHERE event_id = %s",
        (event_id,)
    ))
    assert len(rows) == 1
    assert rows[0].event_id == event_id
    assert rows[0].conversation_id == conv_id

    # 7. Assert Outbox job row was written (status PENDING)
    pending_jobs = cassandra_repo.get_pending_outbox_jobs(limit=5)
    assert len(pending_jobs) == 1
    assert pending_jobs[0]["conversation_id"] == conv_id
    assert pending_jobs[0]["topic"] == outbox_topic
    assert "test" in pending_jobs[0]["payload"]


def test_post_commit_invalidation_integration(clean_tables):
    """Verifies that post_commit_invalidation removes conversation hot cache keys in Redis."""
    session = get_session()
    redis_client = get_redis_client()
    redis_repo = RedisRepository(redis_client)
    service = SnapshotService(session, redis_repo)

    conv_id = "conv-invalidate-test"

    async def _test():
        # Pre-populate Redis
        await redis_repo.set_snapshot({
            "conversation_id": conv_id,
            "user_id": "user-999",
            "message_count": 5,
            "state": "ACTIVE",
            "summary_version": 1,
            "fact_version": 1,
            "snapshot_version": 1,
            "updated_at": datetime.now(timezone.utc)
        })
        await redis_repo.set_summary(conv_id, "Summary text")
        await redis_repo.push_recent_message(conv_id, {"message_id": "msg-1", "role": "user", "content": "hello"})

        # Keys should exist in Redis
        assert await redis_client.exists(f"snapshot:{conv_id}") == 1
        assert await redis_client.exists(f"summary:{conv_id}") == 1
        assert await redis_client.exists(f"recent:{conv_id}") == 1

        # Trigger post-commit invalidation
        await service.post_commit_invalidation(conv_id)

        # Keys should be deleted
        assert await redis_client.exists(f"snapshot:{conv_id}") == 0
        assert await redis_client.exists(f"summary:{conv_id}") == 0
        assert await redis_client.exists(f"recent:{conv_id}") == 0

    run_async(_test())
