"""
tests/integration/test_summary_service_integration.py

Integration tests for Phase 13 Incremental Summarization Service.
Tests snapshot loading, chronological message window reversing, LLM gRPC execution,
and Cassandra persistence / Redis cache invalidations against live databases.
"""

import asyncio
import json
import socket
import pytest
from datetime import datetime, timezone, timedelta

import grpc
from grpc import aio as grpc_aio

from app.db.cassandra import get_session
from app.db.redis import get_redis_client
from app.db.session import initialize_db_sessions, close_db_sessions
from app.models.memory import MemoryState
from app.clients.llm_client import AsyncGRPCConnectionPool, LLMClient
from app.repositories.cassandra_repository import CassandraRepository
from app.repositories.redis_repository import RedisRepository
from app.repositories.memory_repository import MemoryRepository
from app.services.summary_service import SummaryService
from app.proto import llm_pb2, llm_pb2_grpc


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
    """Deletes test summary, snapshot, and message records before and after tests."""
    session = get_session()
    redis_client = get_redis_client()
    
    conversation_id = "test-integration-conv-summary"
    user_id = "test-integration-user-summary"
    
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
        # Delete recent message window rows (clustering key ordering)
        session.execute(
            "DELETE FROM conversation_recent_messages WHERE conversation_id = %s",
            (conversation_id,)
        )

        # Clean Redis
        run_async(redis_client.delete(
            f"snapshot:{conversation_id}",
            f"summary:{conversation_id}",
            f"recent:{conversation_id}"
        ))

    do_cleanup()
    yield conversation_id, user_id
    do_cleanup()


class MockLLMServiceServicer(llm_pb2_grpc.LLMServiceServicer):
    """Servicer implementation capturing GenerateSummary requests and returning mock responses."""
    def __init__(self):
        self.received_requests = []
        self.summary_to_return = "Mocked incremental summary response."

    async def GenerateSummary(self, request, context):
        self.received_requests.append(request)
        return llm_pb2.SummaryResponse(summary_text=self.summary_to_return)


def get_free_port() -> int:
    """Allocates a free TCP port dynamically on localhost."""
    s = socket.socket()
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture
def test_port():
    return get_free_port()


def test_summary_service_incremental_algorithm_integration(clean_databases, test_port):
    """
    Validates the end-to-end integration:
    - Sets up a live gRPC server.
    - Saves initial snapshots, summaries, and raw messages in Cassandra.
    - Runs the summarizer to verify message ordering, LLM calls, Cassandra updates, and Redis eviction.
    """
    conversation_id, user_id = clean_databases
    cassandra_session = get_session()
    redis_client = get_redis_client()

    # 1. Start the mock gRPC Server
    server = grpc_aio.server()
    servicer = MockLLMServiceServicer()
    llm_pb2_grpc.add_LLMServiceServicer_to_server(servicer, server)
    server.add_insecure_port(f"127.0.0.1:{test_port}")
    run_async(server.start())

    # 2. Setup repos, connection pool, and services
    pool = AsyncGRPCConnectionPool(target=f"127.0.0.1:{test_port}", pool_size=2)
    run_async(pool.connect())
    llm_client = LLMClient(pool)

    cassandra_repo = CassandraRepository(cassandra_session)
    redis_repo = RedisRepository(redis_client)
    memory_repo = MemoryRepository(cassandra_repo, redis_repo)
    service = SummaryService(memory_repo, cassandra_repo, llm_client)

    # 3. Seed Cassandra with initial state
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
    run_async(redis_repo.set_snapshot(snapshot))
    run_async(redis_repo.set_summary(conversation_id, "Conversation started with general greetings."))

    # 4. Invoke process_incremental_summary
    updated_snap = run_async(service.process_incremental_summary(conversation_id))

    # 5. Verify results
    assert updated_snap["summary_version"] == 2
    assert updated_snap["last_summary_msg_id"] == "msg-new-2"

    # Verify gRPC servicer received the correct request parameters
    assert len(servicer.received_requests) == 1
    req = servicer.received_requests[0]
    assert req.previous_summary == "Conversation started with general greetings."
    
    # Assert JSON payload contains messages in chronological order (France request first, Paris answer second)
    messages_payload = json.loads(req.new_messages_json)
    assert len(messages_payload) == 2
    assert messages_payload[0]["message_id"] == "msg-new-1"
    assert messages_payload[1]["message_id"] == "msg-new-2"

    # Verify new summary written to Cassandra
    cassandra_summary = cassandra_repo.get_summary(conversation_id)
    assert cassandra_summary is not None
    assert cassandra_summary["summary_text"] == "Mocked incremental summary response."
    assert cassandra_summary["summary_version"] == 2

    # Verify Redis cache has been evicted/invalidated
    cached_summary = run_async(redis_repo.get_summary(conversation_id))
    assert cached_summary is None

    cached_snap = run_async(redis_repo.get_snapshot(conversation_id))
    assert cached_snap is None

    # Clean up gRPC server and connection pool
    run_async(pool.close())
    run_async(server.stop(grace=0))
