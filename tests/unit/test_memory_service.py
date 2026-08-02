"""
tests/unit/test_memory_service.py

Unit tests for Phase 12 Memory State Machine.
Mocks repositories to assert valid state transitions, invalid pathway blocks,
and failure/retry scheduling rules in the Cassandra retry_jobs table.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.models.memory import MemoryState
from app.repositories.memory_repository import MemoryRepository
from app.services.memory_service import MemoryService


@pytest.fixture
def mock_repos():
    """Constructs mock Cassandra and Redis repositories, wrapping them in a MemoryRepository."""
    cassandra_repo = MagicMock()
    redis_repo = MagicMock()
    
    redis_repo.get_snapshot = AsyncMock(return_value=None)
    redis_repo.set_snapshot = AsyncMock()
    
    cassandra_repo.get_snapshot = MagicMock(return_value=None)
    cassandra_repo.upsert_snapshot = MagicMock()
    cassandra_repo.insert_retry_job = MagicMock()
    
    memory_repo = MemoryRepository(cassandra_repo, redis_repo)
    
    return memory_repo, cassandra_repo, redis_repo


@pytest.mark.asyncio
async def test_memory_service_initialization_flow(mock_repos):
    """Verifies that transitioning a non-existent snapshot to ACTIVE initializes a fresh snapshot."""
    memory_repo, cassandra_repo, redis_repo = mock_repos
    service = MemoryService(memory_repo, cassandra_repo)

    # Transitioning to any state besides ACTIVE on a non-existent snapshot must fail
    with pytest.raises(ValueError, match="ACTIVE first"):
        await service.transition_state(conversation_id="conv-1", new_state=MemoryState.READY)

    # Correct initial transition to ACTIVE
    snapshot = await service.transition_state(
        conversation_id="conv-1",
        new_state=MemoryState.ACTIVE,
        user_id="user-xyz"
    )

    assert snapshot["conversation_id"] == "conv-1"
    assert snapshot["user_id"] == "user-xyz"
    assert snapshot["state"] == MemoryState.ACTIVE
    assert snapshot["snapshot_version"] == 1

    cassandra_repo.upsert_snapshot.assert_called_once()
    redis_repo.set_snapshot.assert_awaited_once()


@pytest.mark.asyncio
async def test_memory_service_valid_linear_transitions(mock_repos):
    """Asserts that the standard forward execution paths transition correctly."""
    memory_repo, cassandra_repo, redis_repo = mock_repos
    service = MemoryService(memory_repo, cassandra_repo)

    # Mock snapshot existing in ACTIVE state
    existing_snapshot = {
        "conversation_id": "conv-1",
        "user_id": "user-xyz",
        "message_count": 5,
        "state": MemoryState.ACTIVE,
        "summary_version": 1,
        "fact_version": 1,
        "snapshot_version": 2,
        "last_summary_msg_id": "msg-1",
        "updated_at": "2026-08-01T00:00:00Z"
    }
    redis_repo.get_snapshot.return_value = existing_snapshot

    # Test linear pipeline path updates
    states = [
        MemoryState.SUMMARY_PENDING,
        MemoryState.SUMMARIZING,
        MemoryState.FACT_PENDING,
        MemoryState.EXTRACTING_FACTS,
        MemoryState.EMBEDDING_PENDING,
        MemoryState.READY,
        MemoryState.ACTIVE
    ]

    current_version = 2
    for state in states:
        # Update mock get_snapshot to return current updated state
        redis_repo.get_snapshot.return_value = existing_snapshot
        
        updated_snap = await service.transition_state(conversation_id="conv-1", new_state=state)
        current_version += 1
        
        assert updated_snap["state"] == state
        assert updated_snap["snapshot_version"] == current_version
        
        # Prepare mock for next loop
        existing_snapshot = updated_snap


@pytest.mark.asyncio
async def test_memory_service_invalid_transitions(mock_repos):
    """Verifies that invalid state changes (e.g. ACTIVE -> READY) raise a ValueError."""
    memory_repo, cassandra_repo, redis_repo = mock_repos
    service = MemoryService(memory_repo, cassandra_repo)

    existing_snapshot = {
        "conversation_id": "conv-1",
        "user_id": "user-xyz",
        "state": MemoryState.ACTIVE,
        "snapshot_version": 1
    }
    redis_repo.get_snapshot.return_value = existing_snapshot

    # ACTIVE to READY is invalid
    with pytest.raises(ValueError, match="Invalid state transition"):
        await service.transition_state(conversation_id="conv-1", new_state=MemoryState.READY)

    # ACTIVE to EXTRACTING_FACTS is invalid
    with pytest.raises(ValueError, match="Invalid state transition"):
        await service.transition_state(conversation_id="conv-1", new_state=MemoryState.EXTRACTING_FACTS)


@pytest.mark.asyncio
async def test_memory_service_handle_failure_under_threshold(mock_repos):
    """Asserts that failures under the threshold register a PENDING retry job and preserve snapshot state."""
    memory_repo, cassandra_repo, redis_repo = mock_repos
    service = MemoryService(memory_repo, cassandra_repo)

    # Simulate snapshot in SUMMARIZING state
    existing_snapshot = {
        "conversation_id": "conv-1",
        "user_id": "user-xyz",
        "state": MemoryState.SUMMARIZING,
        "snapshot_version": 1
    }
    redis_repo.get_snapshot.return_value = existing_snapshot

    await service.handle_failure(
        conversation_id="conv-1",
        failed_state=MemoryState.SUMMARIZING,
        job_type="summary",
        payload={"conv_id": "conv-1"},
        error_msg="LLM Timeout",
        attempt_count=2,
        max_retries=5
    )

    # Should register a PENDING retry job
    cassandra_repo.insert_retry_job.assert_called_once()
    job = cassandra_repo.insert_retry_job.call_args[0][0]
    assert job["status"] == "PENDING"
    assert job["job_type"] == "summary"
    assert job["retry_count"] == 2
    assert job["max_retry"] == 5
    assert "LLM Timeout" in job["last_error"]

    # State transition to FAILED should NOT have run
    cassandra_repo.upsert_snapshot.assert_not_called()


@pytest.mark.asyncio
async def test_memory_service_handle_failure_threshold_reached(mock_repos):
    """Asserts that reaching max retries transitions snapshot to FAILED and inserts FAILED job record."""
    memory_repo, cassandra_repo, redis_repo = mock_repos
    service = MemoryService(memory_repo, cassandra_repo)

    # Simulate snapshot in SUMMARIZING state
    existing_snapshot = {
        "conversation_id": "conv-1",
        "user_id": "user-xyz",
        "state": MemoryState.SUMMARIZING,
        "snapshot_version": 1
    }
    redis_repo.get_snapshot.return_value = existing_snapshot

    await service.handle_failure(
        conversation_id="conv-1",
        failed_state=MemoryState.SUMMARIZING,
        job_type="summary",
        payload={"conv_id": "conv-1"},
        error_msg="LLM Timeout",
        attempt_count=5,
        max_retries=5
    )

    # Should upsert snapshot state to FAILED
    cassandra_repo.upsert_snapshot.assert_called_once()
    updated_snap = cassandra_repo.upsert_snapshot.call_args[0][0]
    assert updated_snap["state"] == MemoryState.FAILED

    # Should register a FAILED retry job (DLQ audit row)
    cassandra_repo.insert_retry_job.assert_called_once()
    job = cassandra_repo.insert_retry_job.call_args[0][0]
    assert job["status"] == "FAILED"
    assert "Max retries exhausted" in job["last_error"]
