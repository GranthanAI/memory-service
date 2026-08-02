"""
tests/integration/test_redis.py

Integration tests for Phase 8: Redis Repository Layer.
Uses real Redis container connections to verify hot caching, zstd-compression,
and sliding list limits.
"""

import asyncio
from datetime import datetime, timezone
import pytest

from app.db.session import initialize_db_sessions, close_db_sessions
from app.db.redis import get_redis_client
from app.repositories.redis_repository import RedisRepository


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
def clean_redis():
    """Clears all keys in the test Redis database before each test run."""
    client = get_redis_client()
    run_async(client.flushdb())
    yield


def test_redis_snapshot_integration(clean_redis):
    """Verifies that RedisRepository writes snapshot hashes and parses types correctly."""
    client = get_redis_client()
    repo = RedisRepository(client)

    conv_id = "conv-redis-snap"
    now = datetime.now(timezone.utc).replace(microsecond=0)

    snapshot = {
        "conversation_id": conv_id,
        "user_id": "user-redis-snap",
        "message_count": 10,
        "state": "ACTIVE",
        "summary_version": 2,
        "fact_version": 4,
        "snapshot_version": 1,
        "last_summary_msg_id": "msg-5",
        "updated_at": now
    }

    async def _run():
        # Verify None on cache miss
        missing = await repo.get_snapshot(conv_id)
        assert missing is None

        # Write and read back
        await repo.set_snapshot(snapshot)
        fetched = await repo.get_snapshot(conv_id)
        assert fetched is not None
        assert fetched["conversation_id"] == conv_id
        assert fetched["user_id"] == "user-redis-snap"
        assert fetched["message_count"] == 10
        assert fetched["state"] == "ACTIVE"
        assert fetched["summary_version"] == 2
        assert fetched["fact_version"] == 4
        assert fetched["snapshot_version"] == 1
        assert fetched["last_summary_msg_id"] == "msg-5"
        assert fetched["updated_at"] == now

    run_async(_run())


def test_redis_summary_integration(clean_redis):
    """Verifies summary text compression and base64 string storage."""
    client = get_redis_client()
    repo = RedisRepository(client)

    conv_id = "conv-redis-sum"
    original_summary = "User discussed GraphGPT integration patterns. Key points: 1. Graph fallback. 2. Circuit breakers."

    async def _run():
        # Cache miss
        assert await repo.get_summary(conv_id) is None

        # Cache hit
        await repo.set_summary(conv_id, original_summary)
        fetched = await repo.get_summary(conv_id)
        assert fetched == original_summary

        # Ensure key exists in raw redis and is base64 encoded string
        raw_val = await client.get(f"summary:{conv_id}")
        assert raw_val is not None
        assert isinstance(raw_val, str)

    run_async(_run())


def test_redis_recent_messages_sliding_list_integration(clean_redis):
    """Verifies LPUSH + LTRIM limits elements and enforces the N-message limit."""
    client = get_redis_client()
    repo = RedisRepository(client)
    repo.message_limit = 3

    conv_id = "conv-redis-recent"
    
    msgs = [
        {"message_id": f"msg-{i}", "role": "user", "content": f"Message {i}", "created_at": datetime.now(timezone.utc).isoformat()}
        for i in range(1, 6)
    ]

    async def _run():
        # Push 5 messages (limit is 3)
        for m in msgs:
            await repo.push_recent_message(conv_id, m)

        # Fetch recent messages (should return last 3 in reverse order: msg-5, msg-4, msg-3)
        recent = await repo.get_recent_messages(conv_id)
        assert len(recent) == 3
        assert recent[0]["message_id"] == "msg-5"
        assert recent[1]["message_id"] == "msg-4"
        assert recent[2]["message_id"] == "msg-3"

    run_async(_run())


def test_redis_recent_messages_hydration_integration(clean_redis):
    """Verifies that hydrating the cache completely overrides the current list."""
    client = get_redis_client()
    repo = RedisRepository(client)
    repo.message_limit = 5

    conv_id = "conv-redis-hydrate"
    
    msgs = [
        {"message_id": f"msg-{i}", "role": "user", "content": f"Message {i}", "created_at": datetime.now(timezone.utc).isoformat()}
        for i in range(1, 4)
    ]

    async def _run():
        await repo.set_recent_messages(conv_id, msgs)

        # Check that they match the order: index 0 is newest (msgs[0])
        recent = await repo.get_recent_messages(conv_id)
        assert len(recent) == 3
        assert recent[0]["message_id"] == "msg-1"
        assert recent[1]["message_id"] == "msg-2"
        assert recent[2]["message_id"] == "msg-3"

    run_async(_run())


def test_redis_cache_invalidation_integration(clean_redis):
    """Verifies that invalidate_conversation deletes snapshot, summary, and list keys."""
    client = get_redis_client()
    repo = RedisRepository(client)

    conv_id = "conv-redis-invalidate"

    async def _run():
        await repo.set_snapshot({
            "conversation_id": conv_id,
            "user_id": "user-1",
            "message_count": 1,
            "state": "ACTIVE",
            "summary_version": 1,
            "fact_version": 1,
            "snapshot_version": 1,
            "updated_at": datetime.now(timezone.utc)
        })
        await repo.set_summary(conv_id, "Summary text")
        await repo.push_recent_message(conv_id, {"message_id": "msg-1", "role": "user", "content": "hello"})

        # Keys should exist
        assert await client.exists(f"snapshot:{conv_id}") == 1
        assert await client.exists(f"summary:{conv_id}") == 1
        assert await client.exists(f"recent:{conv_id}") == 1

        # Invalidate
        await repo.invalidate_conversation(conv_id)

        # Keys should be deleted
        assert await client.exists(f"snapshot:{conv_id}") == 0
        assert await client.exists(f"summary:{conv_id}") == 0
        assert await client.exists(f"recent:{conv_id}") == 0

    run_async(_run())
