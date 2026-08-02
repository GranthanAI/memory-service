"""
tests/unit/test_utils.py

Unit tests for Phase 4: Shared Utilities.
Tests cover zstd compression, tokenized Redis locks with Lua release, RedisLockWatchdog,
JSON CustomJSONEncoder serialization, and the latency Timer.

All tests run locally using pytest-asyncio and unittest mocks.
"""

import asyncio
import json
from datetime import datetime
from uuid import UUID, uuid4
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from app.utils.compression import compress_string, decompress_to_string
from app.utils.locks import (
    acquire_redis_lock,
    release_redis_lock,
    RedisLockWatchdog,
)
from app.utils.serialization import to_json, from_json
from app.utils.timers import Timer


# ─── Mock Pydantic Model for testing serialization ──────────────────────────

class MockUser(BaseModel):
    id: UUID
    name: str
    created_at: datetime


# ─── Compression & Decompression Tests ────────────────────────────────────────

def test_compression_and_decompression():
    """
    Verifies that text compression via zstd reduces space and decompresses correctly.
    """
    raw_text = "This is a long sample text to test zstd compression ratios. " * 50
    compressed = compress_string(raw_text)

    # Assert size is reduced
    assert len(compressed) < len(raw_text.encode("utf-8"))

    # Assert decompressed text matches the original raw text
    decompressed = decompress_to_string(compressed)
    assert decompressed == raw_text

    # Empty inputs
    assert compress_string("") == b""
    assert decompress_to_string(b"") == ""


# ─── Tokenized Redis Locks & Lua release Tests ───────────────────────────────

@pytest.mark.asyncio
async def test_redis_lock_lifecycle_success():
    """
    Verifies successful lock acquisition returning a token, and a matching
    token releasing the lock successfully.
    """
    mock_redis = MagicMock()
    mock_redis.set = AsyncMock(return_value=True)
    mock_redis.eval = AsyncMock(return_value=1)

    lock_key = "lock:123"

    # 1. Acquire Lock
    token = await acquire_redis_lock(mock_redis, lock_key, ttl_seconds=10)
    assert token is not None
    assert isinstance(token, str)
    # Check that SET is called with custom value and expiration params
    mock_redis.set.assert_called_once_with(lock_key, token, ex=10, nx=True)

    # 2. Release Lock with matching token
    released = await release_redis_lock(mock_redis, lock_key, token)
    assert released is True
    # Verify Lua script evaluation
    mock_redis.eval.assert_called_once()
    args = mock_redis.eval.call_args
    # First arg is script, second is numkeys, then keys/args
    assert args[0][1] == 1  # numkeys
    assert args[0][2] == lock_key  # KEYS[1]
    assert args[0][3] == token  # ARGV[1]


@pytest.mark.asyncio
async def test_redis_lock_acquire_failure():
    """
    Verifies that if Redis SETNX returns False, acquire_redis_lock returns None.
    """
    mock_redis = MagicMock()
    mock_redis.set = AsyncMock(return_value=False)

    token = await acquire_redis_lock(mock_redis, "lock:123")
    assert token is None


@pytest.mark.asyncio
async def test_redis_lock_release_mismatch():
    """
    Verifies that if Lua script returns 0 (token mismatch or expired),
    release_redis_lock returns False.
    """
    mock_redis = MagicMock()
    mock_redis.eval = AsyncMock(return_value=0)

    released = await release_redis_lock(mock_redis, "lock:123", "stale_token")
    assert released is False


# ─── Watchdog Heartbeat Tests ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_redis_lock_watchdog_heartbeat():
    """
    Asserts that the RedisLockWatchdog runs in the background and periodically
    extends the lock expiration.
    """
    mock_redis = MagicMock()
    # Mock that we still own the lock
    mock_redis.get = AsyncMock(return_value="owner_token")
    mock_redis.expire = AsyncMock(return_value=True)

    watchdog = RedisLockWatchdog(
        client=mock_redis,
        lock_key="lock:123",
        owner_token="owner_token",
        interval=0.01,  # Short interval for testing speed
        extend_by=5,
    )

    await watchdog.start()
    # Let the loop run at least once
    await asyncio.sleep(0.025)
    await watchdog.stop()

    # Verify that mock_redis.get was queried to check ownership
    assert mock_redis.get.call_count >= 1
    # Verify that expire was called to extend the TTL
    assert mock_redis.expire.call_count >= 1
    mock_redis.expire.assert_called_with("lock:123", 5)


@pytest.mark.asyncio
async def test_redis_lock_watchdog_stops_if_ownership_lost():
    """
    Verifies that if the watchdog detects the lock is held by another client,
    it automatically exits and stops heartbeats.
    """
    mock_redis = MagicMock()
    # Mock that lock ownership was lost (returns someone else's token)
    mock_redis.get = AsyncMock(return_value="someone_else")
    mock_redis.expire = AsyncMock(return_value=True)

    watchdog = RedisLockWatchdog(
        client=mock_redis,
        lock_key="lock:123",
        owner_token="owner_token",
        interval=0.01,
        extend_by=5,
    )

    await watchdog.start()
    await asyncio.sleep(0.025)
    # The task should have completed/stopped because loop broke on mismatch
    assert watchdog._task is None or watchdog._task.done()

    # Expire should not be called because ownership was lost
    mock_redis.expire.assert_not_called()


# ─── Serialization Tests ─────────────────────────────────────────────────────

def test_json_serialization():
    """
    Asserts JSON custom encoding processes complex datatypes successfully.
    """
    user_id = uuid4()
    now = datetime.now()
    user = MockUser(id=user_id, name="Alice", created_at=now)

    # Test serialization of pydantic model, datetime, uuid
    json_str = to_json(user)
    parsed = json.loads(json_str)

    assert parsed["id"] == str(user_id)
    assert parsed["name"] == "Alice"
    assert parsed["created_at"] == now.isoformat()

    # Test deserialization
    deserialized = from_json(json_str)
    assert deserialized["name"] == "Alice"
    assert deserialized["id"] == str(user_id)


# ─── Timer Context Manager Tests ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_timer_context_manager():
    """
    Asserts execution timing captures correct delta duration metrics.
    """
    with Timer() as t:
        await asyncio.sleep(0.1)

    assert t.elapsed > 0.05
    assert t.elapsed < 0.2
