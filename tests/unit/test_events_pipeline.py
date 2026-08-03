"""
tests/unit/test_events_pipeline.py

Unit tests for Phase 18 Kafka Consumer, Event Dispatcher, and Retry Scheduler.
"""

import json
import uuid
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from app.models.memory import MemoryState
from app.events.dispatcher import EventDispatcher
from app.events.kafka_consumer import KafkaEventConsumer
from app.events.retry_scheduler import RetryScheduler
from app.core.config import settings


@pytest.fixture
def mock_dispatcher_dependencies():
    processed_event_repo = MagicMock()
    memory_repo = MagicMock()
    memory_repo.get_snapshot = AsyncMock()
    memory_repo.get_recent_messages = AsyncMock()
    snapshot_service = MagicMock()
    snapshot_service.post_commit_invalidation = AsyncMock()
    return processed_event_repo, memory_repo, snapshot_service


@pytest.mark.asyncio
async def test_dispatcher_duplicate_event_skipped(mock_dispatcher_dependencies):
    """EventDispatcher must skip already processed event IDs."""
    processed_event_repo, memory_repo, snapshot_service = mock_dispatcher_dependencies
    
    dispatcher = EventDispatcher(processed_event_repo, memory_repo, snapshot_service)

    raw_event = {
        "event_id": "duplicate-event-id",
        "event_type": "chat.message.created",
        "conversation_id": "conv-123",
        "user_id": "user-456",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": {
            "message_id": "msg-999",
            "role": "user",
            "content": "Hello",
            "created_at": datetime.now(timezone.utc).isoformat()
        }
    }

    processed_event_repo.is_event_processed.return_value = True

    await dispatcher.dispatch(raw_event)

    processed_event_repo.is_event_processed.assert_called_once_with("duplicate-event-id")
    memory_repo.get_snapshot.assert_not_called()
    snapshot_service.commit_snapshot_and_outbox.assert_not_called()


@pytest.mark.asyncio
async def test_dispatcher_normal_event_commits(mock_dispatcher_dependencies):
    """EventDispatcher commits snapshot updates and invalidates cache on new events."""
    processed_event_repo, memory_repo, snapshot_service = mock_dispatcher_dependencies
    
    dispatcher = EventDispatcher(processed_event_repo, memory_repo, snapshot_service, summary_threshold=20)

    raw_event = {
        "event_id": "new-event-id",
        "event_type": "chat.message.created",
        "conversation_id": "conv-123",
        "user_id": "user-456",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": {
            "message_id": "msg-999",
            "role": "user",
            "content": "Hello",
            "created_at": datetime.now(timezone.utc).isoformat()
        }
    }

    processed_event_repo.is_event_processed.return_value = False
    memory_repo.get_snapshot.return_value = None  # Force initialization

    await dispatcher.dispatch(raw_event)

    # Verify snap service commits
    snapshot_service.commit_snapshot_and_outbox.assert_called_once()
    args, kwargs = snapshot_service.commit_snapshot_and_outbox.call_args
    
    assert kwargs["snapshot"]["message_count"] == 1
    assert kwargs["snapshot"]["state"] == MemoryState.ACTIVE.value
    assert kwargs["snapshot"]["user_id"] == "user-456"
    assert kwargs["outbox_topic"] is None  # Below threshold

    # Verify cache invalidation
    snapshot_service.post_commit_invalidation.assert_called_once_with("conv-123")


@pytest.mark.asyncio
async def test_dispatcher_summary_threshold_triggers_outbox(mock_dispatcher_dependencies):
    """EventDispatcher triggers summary pending outbox job at threshold intervals."""
    processed_event_repo, memory_repo, snapshot_service = mock_dispatcher_dependencies
    
    # Set threshold to 2 messages for easy trigger
    dispatcher = EventDispatcher(processed_event_repo, memory_repo, snapshot_service, summary_threshold=2)

    raw_event = {
        "event_id": "trigger-event-id",
        "event_type": "chat.message.created",
        "conversation_id": "conv-123",
        "user_id": "user-456",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": {
            "message_id": "msg-999",
            "role": "user",
            "content": "Hello",
            "created_at": datetime.now(timezone.utc).isoformat()
        }
    }

    processed_event_repo.is_event_processed.return_value = False
    
    # Mocking existing snapshot with 1 message so next count is 2 (triggering threshold)
    memory_repo.get_snapshot.return_value = {
        "conversation_id": "conv-123",
        "user_id": "user-456",
        "message_count": 1,
        "state": MemoryState.ACTIVE.value,
        "summary_version": 0,
        "fact_version": 0,
        "snapshot_version": 1,
        "last_summary_msg_id": None
    }

    await dispatcher.dispatch(raw_event)

    snapshot_service.commit_snapshot_and_outbox.assert_called_once()
    args, kwargs = snapshot_service.commit_snapshot_and_outbox.call_args
    
    assert kwargs["snapshot"]["message_count"] == 2
    assert kwargs["snapshot"]["state"] == MemoryState.SUMMARY_PENDING.value
    assert kwargs["outbox_topic"] == settings.KAFKA_SUMMARY_TOPIC
    assert kwargs["outbox_payload"]["conversation_id"] == "conv-123"


@pytest.mark.asyncio
async def test_consumer_manual_commits():
    """KafkaEventConsumer calls commit offsets only after dispatcher finishes successfully."""
    mock_dispatcher = AsyncMock()
    consumer = KafkaEventConsumer(mock_dispatcher)

    # Patch AIOKafkaConsumer start/stop
    with patch("app.events.kafka_consumer.AIOKafkaConsumer") as mock_consumer_class:
        mock_aio_consumer = MagicMock()
        mock_aio_consumer.start = AsyncMock()
        mock_aio_consumer.stop = AsyncMock()
        mock_aio_consumer.commit = AsyncMock()
        
        # Mock poll return batch
        tp = MagicMock(topic="chat.message.created", partition=0)
        msg = MagicMock(value=json.dumps({"event_id": "1"}).encode("utf-8"))
        mock_aio_consumer.getmany = AsyncMock(return_value={tp: [msg]})
        
        mock_consumer_class.return_value = mock_aio_consumer

        await consumer.start()
        
        # Let polling loop execute one iteration
        await asyncio.sleep(0.1)
        
        await consumer.stop()

        # Assertions
        mock_dispatcher.dispatch.assert_called_once_with({"event_id": "1"})
        mock_aio_consumer.commit.assert_called_once()


@pytest.mark.asyncio
async def test_retry_scheduler_re_dispatches_under_limit():
    """RetryScheduler claims, publishes to original topic, and deletes claimed retry rows."""
    session = MagicMock()
    producer = AsyncMock()
    
    scheduler = RetryScheduler(session, producer)
    
    # Mock Repository operations
    cassandra_repo = MagicMock()
    scheduler.cassandra_repo = cassandra_repo

    job_id = uuid.uuid4()
    mock_job = {
        "job_id": job_id,
        "job_type": "summary",
        "payload": json.dumps({"conversation_id": "conv-123", "version": 1}),
        "retry_count": 2,
        "max_retry": 5,
        "last_error": "Timeout",
        "created_at": datetime.now(timezone.utc),
        "next_retry": datetime.now(timezone.utc)
    }

    cassandra_repo.get_pending_retry_jobs.return_value = [mock_job]
    cassandra_repo.claim_retry_job.return_value = True

    await scheduler._process_retries()

    # Verify atomic claim
    cassandra_repo.claim_retry_job.assert_called_once_with(mock_job)

    # Verify task re-dispatch
    producer.publish_task.assert_called_once_with(
        topic=settings.KAFKA_SUMMARY_TOPIC,
        conversation_id="conv-123",
        payload={"conversation_id": "conv-123", "version": 1, "attempt_count": 3}
    )

    # Verify row delete
    cassandra_repo.delete_retry_job.assert_called_once_with(
        "PROCESSING", mock_job["next_retry"], job_id
    )


@pytest.mark.asyncio
async def test_retry_scheduler_dlq_on_max_limits():
    """RetryScheduler publishes payload to DLQ when retry counts equal or exceed limits."""
    session = MagicMock()
    producer = AsyncMock()
    
    scheduler = RetryScheduler(session, producer)
    cassandra_repo = MagicMock()
    scheduler.cassandra_repo = cassandra_repo

    job_id = uuid.uuid4()
    mock_job = {
        "job_id": job_id,
        "job_type": "summary",
        "payload": json.dumps({"conversation_id": "conv-123"}),
        "retry_count": 5,
        "max_retry": 5,
        "last_error": "Persistent Timeout",
        "created_at": datetime.now(timezone.utc),
        "next_retry": datetime.now(timezone.utc)
    }

    cassandra_repo.get_pending_retry_jobs.return_value = [mock_job]
    cassandra_repo.claim_retry_job.return_value = True

    await scheduler._process_retries()

    # Verify DLQ dispatch
    producer.publish_task.assert_called_once_with(
        topic=settings.KAFKA_DLQ_TOPIC,
        conversation_id="conv-123",
        payload={
            "job_id": str(job_id),
            "job_type": "summary",
            "original_payload": {"conversation_id": "conv-123"},
            "last_error": "Persistent Timeout"
        }
    )

    # Verify deletion of processing row
    cassandra_repo.delete_retry_job.assert_called_once_with(
        "PROCESSING", mock_job["next_retry"], job_id
    )

    # Verify final FAILED row insertion
    cassandra_repo.insert_retry_job.assert_called_once()
    arg = cassandra_repo.insert_retry_job.call_args[0][0]
    assert arg["status"] == "FAILED"
