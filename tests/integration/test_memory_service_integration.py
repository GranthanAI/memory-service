"""
tests/integration/test_memory_service_integration.py

Integration tests for Phase 12 Memory State Machine and Unified Memory Repository.
Tests read-through snapshot cache hydration, write-through state transitions,
and error-recovery retry/DLQ scheduling against live Cassandra and Redis instances.
"""

import asyncio
import time
import uuid
import pytest
from datetime import datetime, timezone, timedelta

from app.db.cassandra import get_session
from app.db.redis import get_redis_client
from app.db.session import initialize_db_sessions, close_db_sessions
from app.models.memory import MemoryState
from app.repositories.cassandra_repository import CassandraRepository
from app.repositories.redis_repository import RedisRepository
from app.repositories.memory_repository import MemoryRepository
from app.services.memory_service import MemoryService


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
def clean_databases():
    """Deletes test snapshot, summary, and retry records before and after tests."""
    session = get_session()
    redis_client = get_redis_client()
    
    conversation_id = "test-integration-conv-state"
    user_id = "test-integration-user-state"
    
    # Clean up function
    def do_cleanup():
        # Clean Cassandra
        session.execute(
            "DELETE FROM conversation_snapshots WHERE conversation_id = %s",
            (conversation_id,)
        )
        session.execute(
            "DELETE FROM conversation_summaries WHERE conversation_id = %s",
            (conversation_id,)
        )
        # Clear retry_jobs (retry_jobs is partitioned by status. Query status values to clean)
        for status in ["PENDING", "PROCESSING", "FAILED"]:
            session.execute("DELETE FROM retry_jobs WHERE status = %s", (status,))

        # Clean Redis
        run_async(redis_client.delete(
            f"snapshot:{conversation_id}",
            f"summary:{conversation_id}",
            f"recent:{conversation_id}"
        ))

    do_cleanup()
    yield conversation_id, user_id
    do_cleanup()


def test_memory_state_machine_and_cache_hydration_integration(clean_databases):
    """
    Validates end-to-end integration of state transitions, read-through cache hydration,
    and outbox-failure retry scheduling.
    """
    conversation_id, user_id = clean_databases
    
    # 1. Instantiate repositories and services
    cassandra_session = get_session()
    redis_client = get_redis_client()

    cassandra_repo = CassandraRepository(cassandra_session)
    redis_repo = RedisRepository(redis_client)
    memory_repo = MemoryRepository(cassandra_repo, redis_repo)
    service = MemoryService(memory_repo, cassandra_repo)

    # 2. Initialize conversation snapshot in ACTIVE state
    snap = run_async(service.transition_state(
        conversation_id=conversation_id,
        new_state=MemoryState.ACTIVE,
        user_id=user_id
    ))
    assert snap["state"] == MemoryState.ACTIVE
    assert snap["snapshot_version"] == 1

    # Verify write-through cache populated Redis
    cached = run_async(redis_repo.get_snapshot(conversation_id))
    assert cached is not None
    assert cached["state"] == "ACTIVE"
    assert cached["user_id"] == user_id

    # 3. Perform a valid state transition: ACTIVE -> SUMMARY_PENDING
    snap_updated = run_async(service.transition_state(
        conversation_id=conversation_id,
        new_state=MemoryState.SUMMARY_PENDING
    ))
    assert snap_updated["state"] == MemoryState.SUMMARY_PENDING
    assert snap_updated["snapshot_version"] == 2

    # Verify cache got updated
    cached_updated = run_async(redis_repo.get_snapshot(conversation_id))
    assert cached_updated["state"] == "SUMMARY_PENDING"

    # 4. Test Read-Through Cache Hydration
    # Manually delete cache key to trigger a miss
    run_async(redis_client.delete(f"snapshot:{conversation_id}"))

    # Verify cache is empty
    assert run_async(redis_repo.get_snapshot(conversation_id)) is None

    # Call get_or_hydrate_snapshot which falls back to Cassandra and backfills Redis
    hydrated = run_async(service.get_or_hydrate_snapshot(conversation_id))
    assert hydrated is not None
    assert hydrated["state"] == "SUMMARY_PENDING"
    assert hydrated["snapshot_version"] == 2

    # Verify cache was repopulated
    cached_repopulated = run_async(redis_repo.get_snapshot(conversation_id))
    assert cached_repopulated is not None
    assert cached_repopulated["state"] == "SUMMARY_PENDING"

    # 5. Test Invalid Transition (SUMMARY_PENDING -> READY is invalid)
    with pytest.raises(ValueError, match="Invalid state transition"):
        run_async(service.transition_state(conversation_id, MemoryState.READY))

    # 6. Test handle_failure under retry limit (creates PENDING retry job)
    run_async(service.handle_failure(
        conversation_id=conversation_id,
        failed_state=MemoryState.SUMMARIZING,
        job_type="summary",
        payload={"conv_id": conversation_id},
        error_msg="LLM Connection Refused",
        attempt_count=2,
        max_retries=5
    ))

    # Assert retry job was scheduled in Cassandra
    pending_jobs = cassandra_repo.get_pending_retry_jobs(
        next_retry_before=datetime.now(timezone.utc) + timedelta(minutes=5),
        limit=10
    )
    assert len(pending_jobs) == 1
    assert pending_jobs[0]["status"] == "PENDING"
    assert pending_jobs[0]["job_type"] == "summary"
    assert pending_jobs[0]["retry_count"] == 2
    assert "LLM Connection Refused" in pending_jobs[0]["last_error"]

    # Verify conversation state was NOT marked as failed yet
    current_snap = run_async(service.get_or_hydrate_snapshot(conversation_id))
    assert current_snap["state"] == "SUMMARY_PENDING"

    # 7. Test handle_failure reaching retry threshold (trips state to FAILED)
    run_async(service.handle_failure(
        conversation_id=conversation_id,
        failed_state=MemoryState.SUMMARIZING,
        job_type="summary",
        payload={"conv_id": conversation_id},
        error_msg="LLM Timeout",
        attempt_count=5,
        max_retries=5
    ))

    # Verify conversation snapshot state is now FAILED
    failed_snap = run_async(service.get_or_hydrate_snapshot(conversation_id))
    assert failed_snap["state"] == "FAILED"

    # Verify a FAILED retry job (DLQ metadata row) was inserted
    # Query Cassandra directly for FAILED partition
    rows = cassandra_session.execute("SELECT status, job_type, last_error FROM retry_jobs WHERE status = 'FAILED'")
    failed_jobs = [r._asdict() for r in rows]
    assert len(failed_jobs) == 1
    assert failed_jobs[0]["job_type"] == "summary"
    assert "Max retries exhausted" in failed_jobs[0]["last_error"]
