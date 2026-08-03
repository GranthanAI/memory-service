"""
tests/unit/test_outbox_worker.py

Unit tests for Phase 17 Outbox Daemon Worker.
Mocks CassandraRepository to assert claiming, successful deletion,
skipped duplicate claims, and failure retry mutations.
"""

import json
import uuid
import pytest
from unittest.mock import AsyncMock, MagicMock

from app.workers.outbox_worker import OutboxDaemonWorker


@pytest.fixture
def mock_repo_and_producer():
    """Mocks CassandraRepository and Kafka Producer."""
    session = MagicMock()
    session.prepare = MagicMock()
    
    cassandra_repo = MagicMock()
    producer = AsyncMock()
    return session, cassandra_repo, producer


@pytest.mark.asyncio
async def test_outbox_worker_lwt_claim_success(mock_repo_and_producer):
    """Asserts that successful LWT claiming triggers Kafka publish and deletion."""
    session, cassandra_repo, producer = mock_repo_and_producer
    
    worker = OutboxDaemonWorker(session, producer)
    worker.cassandra_repo = cassandra_repo

    # 1. Seed a mock pending outbox row
    job_id = uuid.uuid4()
    mock_row = {
        "job_id": job_id,
        "topic": "test-topic",
        "conversation_id": "conv-123",
        "payload": json.dumps({"event": "summary_ready", "version": 1}),
        "attempt_count": 0,
        "created_at": "mock-created-timestamp"
    }

    # Mock repository responses
    cassandra_repo.get_pending_outbox_jobs.return_value = [mock_row]
    cassandra_repo.claim_outbox_job.return_value = True

    # 2. Process one batch
    await worker._process_batch()

    # 3. Assertions
    # Verify claim statement executed with proper variables
    cassandra_repo.claim_outbox_job.assert_called_once_with(mock_row)

    # Verify message is published to Kafka
    producer.publish_task.assert_called_once_with(
        topic="test-topic",
        conversation_id="conv-123",
        payload={"event": "summary_ready", "version": 1}
    )

    # Verify Cassandra delete statement is executed
    cassandra_repo.delete_outbox_job.assert_called_once_with(
        status="PROCESSING",
        created_at="mock-created-timestamp",
        job_id=job_id
    )


@pytest.mark.asyncio
async def test_outbox_worker_lwt_claim_failed_skipped(mock_repo_and_producer):
    """Asserts that failed LWT claims (another worker won) skip processing without publishing or deleting."""
    session, cassandra_repo, producer = mock_repo_and_producer
    
    worker = OutboxDaemonWorker(session, producer)
    worker.cassandra_repo = cassandra_repo

    job_id = uuid.uuid4()
    mock_row = {
        "job_id": job_id,
        "topic": "test-topic",
        "conversation_id": "conv-123",
        "payload": "{}",
        "attempt_count": 0,
        "created_at": "mock-created-timestamp"
    }

    cassandra_repo.get_pending_outbox_jobs.return_value = [mock_row]
    cassandra_repo.claim_outbox_job.return_value = False

    await worker._process_batch()

    # Assertions
    cassandra_repo.claim_outbox_job.assert_called_once_with(mock_row)
    
    # Kafka publish was skipped
    producer.publish_task.assert_not_called()
    
    # Cassandra delete and fail updates were never executed
    cassandra_repo.delete_outbox_job.assert_not_called()
    cassandra_repo.fail_outbox_job.assert_not_called()


@pytest.mark.asyncio
async def test_outbox_worker_publish_failure_writes_error(mock_repo_and_producer):
    """Asserts that Kafka producer exceptions update attempt count and record errors in Cassandra."""
    session, cassandra_repo, producer = mock_repo_and_producer
    
    worker = OutboxDaemonWorker(session, producer)
    worker.cassandra_repo = cassandra_repo

    job_id = uuid.uuid4()
    mock_row = {
        "job_id": job_id,
        "topic": "test-topic",
        "conversation_id": "conv-123",
        "payload": "{}",
        "attempt_count": 2,
        "created_at": "mock-created-timestamp"
    }

    # Make publisher raise error
    producer.publish_task.side_effect = RuntimeError("Broker connection failed")

    cassandra_repo.get_pending_outbox_jobs.return_value = [mock_row]
    cassandra_repo.claim_outbox_job.return_value = True

    await worker._process_batch()

    # Assertions
    producer.publish_task.assert_called_once()
    
    # Verify fail update statement was executed instead of delete
    cassandra_repo.fail_outbox_job.assert_called_once_with(mock_row, "Broker connection failed")
    cassandra_repo.delete_outbox_job.assert_not_called()
