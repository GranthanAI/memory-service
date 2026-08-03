"""
tests/integration/test_context_builder_integration.py

Integration tests for Phase 16 Structured Context Builder & Retrieval Service.
Verifies real Cassandra reads, Redis hot cache hydration, and concurrent assembly.
"""

import asyncio
import uuid
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock

from app.db.cassandra import get_session
from app.db.redis import get_redis_client
from app.db.session import initialize_db_sessions, close_db_sessions
from app.repositories.cassandra_repository import CassandraRepository
from app.repositories.redis_repository import RedisRepository
from app.repositories.memory_repository import MemoryRepository
from app.repositories.milvus_repository import MilvusRepository
from app.clients.graph_client import GraphClient
from app.services.retrieval_service import RetrievalService
from app.services.context_builder import ContextBuilder


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
    """Cleans up Cassandra and Redis for context testing."""
    session = get_session()
    redis = get_redis_client()
    conversation_id = "test-integration-context-builder"
    user_id = "user-integration-context"
    
    # Clean Cassandra
    session.execute("DELETE FROM conversation_snapshots WHERE conversation_id = %s", (conversation_id,))
    session.execute("DELETE FROM conversation_summaries WHERE conversation_id = %s", (conversation_id,))
    session.execute("DELETE FROM conversation_recent_messages WHERE conversation_id = %s", (conversation_id,))
    
    # Clean Redis
    run_async(redis.delete(
        f"snapshot:{conversation_id}",
        f"summary:{conversation_id}",
        f"recent:{conversation_id}"
    ))
    
    yield conversation_id, user_id
    
    # Post-clean
    session.execute("DELETE FROM conversation_snapshots WHERE conversation_id = %s", (conversation_id,))
    session.execute("DELETE FROM conversation_summaries WHERE conversation_id = %s", (conversation_id,))
    session.execute("DELETE FROM conversation_recent_messages WHERE conversation_id = %s", (conversation_id,))
    run_async(redis.delete(
        f"snapshot:{conversation_id}",
        f"summary:{conversation_id}",
        f"recent:{conversation_id}"
    ))


def test_context_builder_read_through_hydration_integration(clean_databases):
    """
    Asserts that build_context reads from Cassandra on cache miss,
    hydrates Redis cache, and retrieves assembled context correctly.
    """
    conversation_id, user_id = clean_databases
    cassandra_session = get_session()
    redis_client = get_redis_client()

    cassandra_repo = CassandraRepository(cassandra_session)
    redis_repo = RedisRepository(redis_client)
    memory_repo = MemoryRepository(cassandra_repo, redis_repo)
    milvus_repo = MilvusRepository()

    # 1. Seed Cassandra metadata and message data
    snapshot = {
        "conversation_id": conversation_id,
        "user_id": user_id,
        "message_count": 2,
        "state": "ACTIVE",
        "summary_version": 1,
        "fact_version": 1,
        "snapshot_version": 1,
        "last_summary_msg_id": "msg-2",
        "updated_at": datetime.now(timezone.utc)
    }
    cassandra_repo.upsert_snapshot(snapshot)

    summary = {
        "conversation_id": conversation_id,
        "summary_text": "Conversation has just started.",
        "summary_version": 1,
        "model_name": "mock-gpt",
        "model_version": "v1.0.0",
        "generated_at": datetime.now(timezone.utc)
    }
    cassandra_repo.upsert_summary(summary)

    msg_id = str(uuid.uuid4())
    cassandra_repo.append_recent_message(conversation_id, {
        "message_id": msg_id,
        "role": "user",
        "content": "I like dark roast coffee.",
        "created_at": datetime.now(timezone.utc)
    })

    # 2. Instantiate services
    retrieval_service = RetrievalService(memory_repo, milvus_repo)
    mock_graph_client = MagicMock(spec=GraphClient)
    mock_graph_client.get_ancestors = AsyncMock(return_value=[
        {"conversation_id": "conv-parent-old", "summary": "Old coffee details"}
    ])

    builder = ContextBuilder(retrieval_service, mock_graph_client)

    # 3. First call: Cache miss -> Read from Cassandra and hydrate Redis
    context_first = run_async(builder.build_context(user_id, conversation_id))
    
    assert context_first["current_summary"] == "Conversation has just started."
    assert len(context_first["short_term_messages"]) == 1
    assert context_first["short_term_messages"][0]["content"] == "I like dark roast coffee."
    assert len(context_first["parent_summaries"]) == 1
    assert context_first["parent_summaries"][0]["summary"] == "Old coffee details"

    # Verify Redis is hydrated
    redis_snap = run_async(redis_repo.get_snapshot(conversation_id))
    assert redis_snap is not None
    assert redis_snap["state"] == "ACTIVE"

    redis_summary = run_async(redis_repo.get_summary(conversation_id))
    assert redis_summary == "Conversation has just started."

    redis_recent = run_async(redis_repo.get_recent_messages(conversation_id))
    assert len(redis_recent) == 1
    assert redis_recent[0]["content"] == "I like dark roast coffee."

    # 4. Second call: Cache hit -> Loads directly from Redis
    # Let's delete from Cassandra to confirm it reads from cache
    cassandra_session.execute("DELETE FROM conversation_summaries WHERE conversation_id = %s", (conversation_id,))
    
    context_second = run_async(builder.build_context(user_id, conversation_id))
    assert context_second["current_summary"] == "Conversation has just started."  # Loaded from Redis!
