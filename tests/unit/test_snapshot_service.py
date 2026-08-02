"""
tests/unit/test_snapshot_service.py

Unit tests for Phase 9: Snapshot Builder Service.
Mocks the Cassandra session and Redis repository to verify atomic logged batch construction
and cache invalidation hooks.
"""

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from cassandra.query import BatchStatement, BatchType

from app.services.snapshot_service import SnapshotService


def test_snapshot_service_initialization_prepares_statements():
    """Verifies that SnapshotService prepares the four required statements upon initialization."""
    mock_session = MagicMock()
    mock_redis_repo = MagicMock()
    
    mock_session.prepare.return_value = "prepared_statement"

    service = SnapshotService(mock_session, mock_redis_repo)

    assert mock_session.prepare.call_count == 4
    assert service._snap_upsert == "prepared_statement"
    assert service._recent_msg_append == "prepared_statement"
    assert service._idemp_insert == "prepared_statement"
    assert service._outbox_insert == "prepared_statement"


def test_commit_snapshot_and_outbox_constructs_batch():
    """
    Verifies that commit_snapshot_and_outbox constructs a logged batch
    containing snapshot upsert, message append, idempotency marker, and outbox insert,
    and executes it against the Cassandra session.
    """
    mock_session = MagicMock()
    mock_redis_repo = MagicMock()
    
    # Setup prepared statements
    mock_session.prepare.side_effect = [
        "snap_upsert_stmt",
        "recent_msg_stmt",
        "idemp_stmt",
        "outbox_stmt"
    ]

    service = SnapshotService(mock_session, mock_redis_repo)

    conv_id = "conv-123"
    snapshot = {
        "conversation_id": conv_id,
        "user_id": "user-456",
        "message_count": 5,
        "state": "ACTIVE",
        "summary_version": 1,
        "fact_version": 2,
        "snapshot_version": 1,
        "last_summary_msg_id": "msg-4"
    }
    
    message = {
        "message_id": "msg-5",
        "role": "user",
        "content": "Hello world",
        "created_at": datetime(2026, 8, 2, 12, 0, 0, tzinfo=timezone.utc)
    }

    event_id = "event-789"
    outbox_topic = "test.topic"
    outbox_payload = {"foo": "bar"}

    # Mock execute
    mock_session.execute = MagicMock()

    with patch("app.services.snapshot_service.BatchStatement") as mock_batch_class:
        mock_batch = MagicMock()
        mock_batch_class.return_value = mock_batch

        service.commit_snapshot_and_outbox(
            snapshot=snapshot,
            event_id=event_id,
            outbox_topic=outbox_topic,
            outbox_payload=outbox_payload,
            message=message
        )

        # Verify batch initialization as LOGGED
        mock_batch_class.assert_called_once_with(batch_type=BatchType.LOGGED)

        # Verify statements added to batch
        assert mock_batch.add.call_count == 4
        
        # Verify execute was called with batch
        mock_session.execute.assert_called_once_with(mock_batch)


@pytest.mark.asyncio
async def test_post_commit_invalidation_calls_redis():
    """Verifies that post_commit_invalidation calls redis_repo invalidation."""
    mock_session = MagicMock()
    mock_redis_repo = MagicMock()
    mock_redis_repo.invalidate_conversation = AsyncMock()

    service = SnapshotService(mock_session, mock_redis_repo)
    await service.post_commit_invalidation("conv-123")

    mock_redis_repo.invalidate_conversation.assert_called_once_with("conv-123")
