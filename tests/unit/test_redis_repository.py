"""
tests/unit/test_redis_repository.py

Unit tests for Phase 8: Redis Repository Layer.
Mocks the async Redis client to verify cache read/write behaviors and key formats.
"""

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.repositories.redis_repository import RedisRepository


@pytest.mark.asyncio
async def test_redis_snapshot_caching():
    """
    Verifies that set_snapshot caches snapshot properties as strings
    and get_snapshot parses them back to correct Python types.
    """
    mock_redis = MagicMock()
    mock_redis.hgetall = AsyncMock(return_value={
        "conversation_id": "conv-abc",
        "user_id": "user-xyz",
        "message_count": "45",
        "state": "ACTIVE",
        "summary_version": "3",
        "fact_version": "5",
        "snapshot_version": "1",
        "last_summary_msg_id": "msg-12",
        "updated_at": "2026-08-02T12:00:00+00:00"
    })
    
    mock_pipe = MagicMock()
    mock_pipe.hset = MagicMock()
    mock_pipe.expire = MagicMock()
    mock_pipe.execute = AsyncMock()
    mock_redis.pipeline.return_value.__aenter__ = AsyncMock(return_value=mock_pipe)

    repo = RedisRepository(mock_redis)

    # 1. Test get_snapshot
    snap = await repo.get_snapshot("conv-abc")
    assert snap is not None
    assert snap["conversation_id"] == "conv-abc"
    assert snap["message_count"] == 45
    assert snap["summary_version"] == 3
    assert snap["updated_at"].tzinfo == timezone.utc

    # 2. Test set_snapshot
    input_snap = {
        "conversation_id": "conv-abc",
        "user_id": "user-xyz",
        "message_count": 45,
        "state": "ACTIVE",
        "summary_version": 3,
        "fact_version": 5,
        "snapshot_version": 1,
        "last_summary_msg_id": "msg-12",
        "updated_at": datetime(2026, 8, 2, 12, 0, 0, tzinfo=timezone.utc)
    }
    await repo.set_snapshot(input_snap)
    
    mock_redis.pipeline.assert_called_once_with(transaction=True)
    mock_pipe.hset.assert_called_once()
    mock_pipe.expire.assert_called_once_with("snapshot:conv-abc", 2592000)


@pytest.mark.asyncio
async def test_redis_summary_compression():
    """
    Verifies that summaries are compressed, base64-encoded, and correctly retrieved.
    """
    mock_redis = MagicMock()
    
    # Pre-calculated base64 zstd payload of 'This is a long summary'
    import base64
    from app.utils.compression import compress_string
    compressed = compress_string("This is a long summary")
    encoded_str = base64.b64encode(compressed).decode("utf-8")
    
    mock_redis.get = AsyncMock(return_value=encoded_str)
    mock_redis.set = AsyncMock()

    repo = RedisRepository(mock_redis)

    # 1. Test get_summary
    summary = await repo.get_summary("conv-abc")
    assert summary == "This is a long summary"

    # 2. Test set_summary
    await repo.set_summary("conv-abc", "This is a long summary")
    mock_redis.set.assert_called_once()
    call_args = mock_redis.set.call_args
    # First is key, second is base64 string, ex is expiration
    assert call_args[0][0] == "summary:conv-abc"
    assert call_args[0][1] == encoded_str
    assert call_args.kwargs.get("ex") == 2592000


@pytest.mark.asyncio
async def test_redis_recent_messages_sliding_list():
    """
    Verifies that push_recent_message applies LPUSH and LTRIM constraints.
    """
    mock_redis = MagicMock()
    mock_pipe = MagicMock()
    mock_pipe.lpush = MagicMock()
    mock_pipe.ltrim = MagicMock()
    mock_pipe.expire = MagicMock()
    mock_pipe.execute = AsyncMock()
    mock_redis.pipeline.return_value.__aenter__ = AsyncMock(return_value=mock_pipe)

    repo = RedisRepository(mock_redis)
    repo.message_limit = 20

    msg = {
        "message_id": "msg-123",
        "role": "user",
        "content": "Hello",
        "created_at": datetime.utcnow()
    }

    await repo.push_recent_message("conv-abc", msg)

    mock_redis.pipeline.assert_called_once_with(transaction=True)
    mock_pipe.lpush.assert_called_once()
    mock_pipe.ltrim.assert_called_once_with("recent:conv-abc", 0, 19)
    mock_pipe.expire.assert_called_once_with("recent:conv-abc", 2592000)


@pytest.mark.asyncio
async def test_redis_cache_invalidation():
    """
    Verifies that invalidate_conversation deletes snapshot, summary, and recent list keys.
    """
    mock_redis = MagicMock()
    mock_redis.delete = AsyncMock()

    repo = RedisRepository(mock_redis)
    await repo.invalidate_conversation("conv-abc")

    mock_redis.delete.assert_called_once_with(
        "snapshot:conv-abc",
        "summary:conv-abc",
        "recent:conv-abc"
    )
