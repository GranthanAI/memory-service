"""
tests/integration/test_summary_service_integration.py

Integration tests for SummaryService.
Verifies the database interactions (Cassandra, Redis) and cache eviction logic
along with LLMService wiring.
"""

import json
from datetime import datetime, timedelta, timezone

import pytest

from app.core.config import settings
from app.db.cassandra import get_session
from app.db.redis import get_redis_client
from app.models.memory import MemoryState
from app.repositories.cassandra_repository import CassandraRepository
from app.repositories.memory_repository import MemoryRepository
from app.repositories.redis_repository import RedisRepository
from app.services.snapshot_service import SnapshotService
from app.services.summary_service import SummaryService
from app.providers.mock_provider import MockLLMProvider
from app.managers.llm_manager import LLMManager
from app.services.llm_service import LLMService as InternalLLMService


@pytest.fixture
def clean_databases():
    """Seeds databases before each test and cleans them up after."""
    session = get_session()
    redis_client = get_redis_client()
    
    conversation_id = "test-conv-999"
    user_id = "test-user-999"

    def do_cleanup():
        # Clean Cassandra
        session.execute(
            f"DELETE FROM conversation_snapshots WHERE conversation_id='{conversation_id}'"
        )
        session.execute(
            f"DELETE FROM conversation_summaries WHERE conversation_id='{conversation_id}'"
        )
        session.execute(
            f"DELETE FROM conversation_recent_messages WHERE conversation_id='{conversation_id}'"
        )
        # Clean Redis
        import asyncio
        loop = asyncio.get_event_loop()
        loop.run_until_complete(redis_client.delete(f"snapshot:{conversation_id}"))
        loop.run_until_complete(redis_client.delete(f"summary:{conversation_id}"))

    do_cleanup()
    yield conversation_id, user_id
    do_cleanup()


class CustomMockProvider(MockLLMProvider):
    """Custom mock provider returning a specific summary text."""
    async def generate(self, messages, system_prompt=None):
        return "Mocked incremental summary response."


@pytest.mark.asyncio
async def test_summary_service_incremental_algorithm_integration(clean_databases):
    """
    Validates the end-to-end integration:
    - Saves initial snapshots, summaries, and raw messages in Cassandra.
    - Runs the summarizer to verify message ordering, LLM calls, Cassandra updates, and Redis eviction.
    """
    conversation_id, user_id = clean_databases
    cassandra_session = get_session()
    redis_client = get_redis_client()

    # 1. Setup repos, internal LLM Service, and summary service
    llm_manager = LLMManager(CustomMockProvider())
    llm_service = InternalLLMService(llm_manager)

    cassandra_repo = CassandraRepository(cassandra_session)
    redis_repo = RedisRepository(redis_client)
    memory_repo = MemoryRepository(cassandra_repo, redis_repo)
    service = SummaryService(memory_repo, cassandra_repo, llm_service)

    # 2. Seed Cassandra with initial state
    # Previous snapshot in SUMMARIZING state
    snapshot = {
        "conversation_id": conversation_id,
        "user_id": user_id,
        "message_count": 2,
        "state": MemoryState.SUMMARIZING,
        "summary_version": 1,
        "fact_version": 0,
        "snapshot_version": 2,
        "last_summary_msg_id": "msg-old",
        "updated_at": datetime.now(timezone.utc)
    }
    cassandra_repo.upsert_snapshot(snapshot)

    # Previous version 1 summary in Cassandra
    prev_summary_record = {
        "conversation_id": conversation_id,
        "summary_text": "Conversation started with general greetings.",
        "summary_version": 1,
        "model_name": "test-summary-model",
        "model_version": "v1.0.0",
        "generated_at": datetime.now(timezone.utc) - timedelta(hours=1)
    }
    cassandra_repo.upsert_summary(prev_summary_record)

    # Recent messages in Cassandra (newest messages have more recent timestamps)
    # We append two new messages
    msg1 = {
        "message_id": "msg-new-1",
        "role": "user",
        "content": "What is the capital of France?",
        "created_at": datetime.now(timezone.utc) - timedelta(seconds=10)
    }
    msg2 = {
        "message_id": "msg-new-2",
        "role": "assistant",
        "content": "The capital of France is Paris.",
        "created_at": datetime.now(timezone.utc) - timedelta(seconds=5)
    }
    cassandra_repo.append_recent_message(conversation_id, msg1)
    cassandra_repo.append_recent_message(conversation_id, msg2)

    # Pre-populate Redis hot cache for the snapshot to verify invalidation deletes it
    await redis_repo.set_snapshot(snapshot)
    await redis_repo.set_summary(conversation_id, "Conversation started with general greetings.")

    # 3. Invoke process_incremental_summary
    updated_snap = await service.process_incremental_summary(conversation_id)

    # 4. Verify results
    assert updated_snap["summary_version"] == 2
    assert updated_snap["last_summary_msg_id"] == "msg-new-2"

    # Verify new summary written to Cassandra
    cassandra_summary = cassandra_repo.get_summary(conversation_id)
    assert cassandra_summary is not None
    assert cassandra_summary["summary_text"] == "Mocked incremental summary response."
    assert cassandra_summary["summary_version"] == 2

    # Verify Redis cache has been evicted/invalidated
    cached_summary = await redis_repo.get_summary(conversation_id)
    assert cached_summary is None

    cached_snap = await redis_repo.get_snapshot(conversation_id)
    assert cached_snap is None
