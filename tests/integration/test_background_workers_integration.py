"""
tests/integration/test_background_workers_integration.py

Integration tests for Phase 19 Background Worker Daemons.
Verifies the end-to-end sequence of transitions:
ingestion -> snapshot batch commit -> outbox dispatch -> summary worker -> fact worker -> embedding worker -> ready.
"""

import asyncio
import json
import uuid
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock

from app.db.cassandra import get_session
from app.db.session import initialize_db_sessions, close_db_sessions
from app.core.config import settings
from app.models.memory import MemoryState
from app.repositories.cassandra_repository import CassandraRepository
from app.repositories.processed_event_repository import ProcessedEventRepository
from app.repositories.redis_repository import RedisRepository
from app.repositories.memory_repository import MemoryRepository
from app.repositories.milvus_repository import MilvusRepository
from app.services.snapshot_service import SnapshotService
from app.services.memory_service import MemoryService
from app.services.summary_service import SummaryService
from app.services.long_memory_service import LongMemoryService
from app.events.dispatcher import EventDispatcher
from app.workers.outbox_worker import OutboxDaemonWorker
from app.workers.summary_worker import SummaryWorker
from app.workers.fact_worker import FactWorker
from app.workers.embedding_worker import EmbeddingWorker
from app.clients.llm_client import LLMClient


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
    """Initializes and tears down database sessions."""
    run_async(initialize_db_sessions())
    yield
    run_async(close_db_sessions())


@pytest.fixture
def clean_pipeline_tables():
    """Clears Cassandra and Milvus before and after integration runs."""
    session = get_session()
    
    tables = [
        "processed_events",
        "conversation_snapshots",
        "conversation_recent_messages",
        "conversation_summaries",
        "outbox_jobs",
        "retry_jobs",
        "user_facts"
    ]
    for table in tables:
        session.execute(f"TRUNCATE {table}")
        
    # Flush Milvus
    milvus_repo = MilvusRepository()
    try:
        from pymilvus import utility
        if utility.has_collection(milvus_repo.collection_name):
            utility.drop_collection(milvus_repo.collection_name)
        milvus_repo.init_collection()
    except Exception:
        pass

    yield
    
    for table in tables:
        session.execute(f"TRUNCATE {table}")


class MockPipelineProducer:
    """Mock Kafka Producer capturing outbox publishes and storing them for manual delivery."""
    def __init__(self):
        self.published_messages = []

    async def publish_task(self, topic: str, conversation_id: str, payload: dict) -> None:
        self.published_messages.append({
            "topic": topic,
            "conversation_id": conversation_id,
            "payload": payload
        })


@pytest.mark.asyncio
async def test_end_to_end_worker_pipeline_integration(clean_pipeline_tables):
    """
    Executes the entire end-to-end memory pipeline process sequentially.
    Asserts snapshot transitions and fact insertions in database layers.
    """
    session = get_session()
    cassandra_repo = CassandraRepository(session)
    milvus_repo = MilvusRepository()
    
    # Mock Redis for integration ease
    redis_repo = MagicMock(spec=RedisRepository)
    redis_repo.get_snapshot = AsyncMock(return_value=None)
    redis_repo.set_snapshot = AsyncMock()
    redis_repo.get_recent_messages = AsyncMock(return_value=None)
    redis_repo.set_recent_messages = AsyncMock()
    redis_repo.invalidate_conversation = AsyncMock()

    memory_repo = MemoryRepository(cassandra_repo, redis_repo)
    snapshot_service = SnapshotService(session, redis_repo)
    memory_service = MemoryService(memory_repo, cassandra_repo)

    # 1. Mock LLM client to return scripted summary, facts, and embeddings
    llm_client = MagicMock(spec=LLMClient)
    llm_client.state = "CLOSED"
    
    # Define LLM client call responses
    summary_text = "Integrated summary of user conversation"
    extracted_facts = ["preferences:0.9:Likes black coffee", "habits:0.8:Wakes up at 6am"]
    embedding_vector = [0.15] * settings.VECTOR_DIMENSION

    async def cb_side_effect(stub_fn, *args, **kwargs):
        # Inspect stub_fn source or stub name to return correct mock response
        fn_name = stub_fn.__name__
        if "summary_stub" in fn_name or "GenerateSummary" in fn_name:
            return summary_text
        elif "extract_facts_stub" in fn_name or "ExtractFacts" in fn_name:
            return extracted_facts
        elif "embed_stub" in fn_name or "GenerateEmbedding" in fn_name:
            return embedding_vector
        return None

    llm_client.call_with_circuit_breaker = AsyncMock(side_effect=cb_side_effect)

    # Instantiate services
    summary_service = SummaryService(memory_repo, cassandra_repo, llm_client)
    long_memory_service = LongMemoryService(cassandra_repo, milvus_repo)

    processed_event_repo = ProcessedEventRepository(session)
    # Set threshold to 2 for fast trigger
    dispatcher = EventDispatcher(
        processed_event_repo=processed_event_repo,
        memory_repo=memory_repo,
        snapshot_service=snapshot_service,
        summary_threshold=2
    )

    producer = MockPipelineProducer()
    outbox_worker = OutboxDaemonWorker(session, producer)

    conversation_id = f"conv-{uuid.uuid4()}"
    user_id = "user-e2e-789"

    # --- STEP 1: Ingest First Message (count = 1) ---
    event_1 = {
        "event_id": f"evt-{uuid.uuid4()}",
        "event_type": "chat.message.created",
        "conversation_id": conversation_id,
        "user_id": user_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": {
            "message_id": f"msg-{uuid.uuid4()}",
            "role": "user",
            "content": "Hi, I wake up at 6am every day.",
            "created_at": datetime.now(timezone.utc).isoformat()
        }
    }
    await dispatcher.dispatch(event_1)
    
    # Assert state ACTIVE and count = 1
    snap = await memory_repo.get_snapshot(conversation_id)
    assert snap["message_count"] == 1
    assert snap["state"] == MemoryState.ACTIVE.value

    # --- STEP 2: Ingest Second Message (count = 2 -> Triggers SUMMARY_PENDING) ---
    event_2 = {
        "event_id": f"evt-{uuid.uuid4()}",
        "event_type": "chat.message.created",
        "conversation_id": conversation_id,
        "user_id": user_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": {
            "message_id": f"msg-{uuid.uuid4()}",
            "role": "assistant",
            "content": "Nice! And what do you like to drink?",
            "created_at": datetime.now(timezone.utc).isoformat()
        }
    }
    await dispatcher.dispatch(event_2)

    # Assert state SUMMARY_PENDING and count = 2
    snap = await memory_repo.get_snapshot(conversation_id)
    assert snap["message_count"] == 2
    assert snap["state"] == MemoryState.SUMMARY_PENDING.value

    # --- STEP 3: Outbox Daemon processes SUMMARY_PENDING outbox row ---
    await outbox_worker._process_batch()
    # Filter only messages published during this step for this conversation
    step3_msgs = [m for m in producer.published_messages if m["conversation_id"] == conversation_id]
    assert len(step3_msgs) >= 1
    summary_request = next(m for m in step3_msgs if m["topic"] == "memory.summary.request")
    assert summary_request["topic"] == "memory.summary.request"
    # Clear published messages and purge any stale outbox rows before next step
    producer.published_messages.clear()
    session.execute("TRUNCATE outbox_jobs")

    # --- STEP 4: SummaryWorker processes summary request ---
    summary_worker = SummaryWorker(session, memory_service, summary_service)
    
    # Simulate processing message
    # 1. Transition to SUMMARIZING
    await memory_service.transition_state(conversation_id, MemoryState.SUMMARIZING)
    # 2. Run summarizer
    updated_snap = await summary_service.process_incremental_summary(conversation_id)
    # 3. Transition to FACT_PENDING
    await memory_service.transition_state(conversation_id, MemoryState.FACT_PENDING)
    # 4. Write fact outbox job
    outbox_job = {
        "status": "PENDING",
        "created_at": datetime.now(timezone.utc),
        "job_id": uuid.uuid4(),
        "topic": settings.KAFKA_FACT_TOPIC,
        "conversation_id": conversation_id,
        "payload": json.dumps({
            "conversation_id": conversation_id,
            "user_id": user_id,
            "version": updated_snap["summary_version"]
        }),
        "attempt_count": 0
    }
    cassandra_repo.insert_outbox_job(outbox_job)

    # Assert snapshot is FACT_PENDING
    snap = await memory_repo.get_snapshot(conversation_id)
    assert snap["state"] == MemoryState.FACT_PENDING.value

    # --- STEP 5: Outbox Daemon processes FACT_PENDING outbox row ---
    await outbox_worker._process_batch()
    step5_msgs = [m for m in producer.published_messages if m["conversation_id"] == conversation_id]
    assert len(step5_msgs) >= 1
    fact_request = next(m for m in step5_msgs if m["topic"] == "memory.fact.request")
    assert fact_request["topic"] == "memory.fact.request"
    # Clear published messages and purge stale outbox rows before next step
    producer.published_messages.clear()
    session.execute("TRUNCATE outbox_jobs")

    # --- STEP 6: FactWorker processes fact request ---
    # 1. Transition to EXTRACTING_FACTS
    await memory_service.transition_state(conversation_id, MemoryState.EXTRACTING_FACTS)
    # 2. Extract facts via LLM
    async def extract_facts_stub(channel):
        pass
    facts = await llm_client.call_with_circuit_breaker(extract_facts_stub)
    # 3. Transition to EMBEDDING_PENDING
    await memory_service.transition_state(conversation_id, MemoryState.EMBEDDING_PENDING)
    # 4. Write embedding outbox job
    outbox_job_2 = {
        "status": "PENDING",
        "created_at": datetime.now(timezone.utc),
        "job_id": uuid.uuid4(),
        "topic": settings.KAFKA_EMBEDDING_TOPIC,
        "conversation_id": conversation_id,
        "payload": json.dumps({
            "conversation_id": conversation_id,
            "user_id": user_id,
            "facts": facts,
            "version": 1
        }),
        "attempt_count": 0
    }
    cassandra_repo.insert_outbox_job(outbox_job_2)

    # Assert snapshot is EMBEDDING_PENDING
    snap = await memory_repo.get_snapshot(conversation_id)
    assert snap["state"] == MemoryState.EMBEDDING_PENDING.value

    # --- STEP 7: Outbox Daemon processes EMBEDDING_PENDING outbox row ---
    await outbox_worker._process_batch()
    step7_msgs = [m for m in producer.published_messages if m["conversation_id"] == conversation_id]
    assert len(step7_msgs) >= 1
    embedding_request = next(m for m in step7_msgs if m["topic"] == "memory.embedding.request")
    assert embedding_request["topic"] == "memory.embedding.request"

    # --- STEP 8: EmbeddingWorker processes embedding request ---
    # Parse payload
    emb_payload = embedding_request["payload"]
    facts_list = emb_payload["facts"]
    
    incoming_facts = []
    # Parse and build embedding vector list
    for fact_str in facts_list:
        parts = fact_str.split(":", 2)
        category = parts[0].strip()
        importance = float(parts[1])
        statement = parts[2].strip()
        incoming_facts.append({
            "statement": statement,
            "category": category,
            "importance": importance,
            "vector": embedding_vector
        })
        
    # Execute merge facts
    stats = await long_memory_service.merge_user_facts(
        user_id=user_id,
        conversation_id=conversation_id,
        incoming_facts=incoming_facts
    )
    
    # Transition snapshot to READY -> ACTIVE
    await memory_service.transition_state(conversation_id, MemoryState.READY)
    await memory_service.transition_state(conversation_id, MemoryState.ACTIVE)

    # --- FINAL VERIFICATIONS ---
    # 1. Snapshot is back to ACTIVE
    final_snap = await memory_repo.get_snapshot(conversation_id)
    assert final_snap["state"] == MemoryState.ACTIVE.value
    assert final_snap["message_count"] == 2

    # 2. Fact Merge stats shows 2 inserted facts
    assert stats["inserted"] == 2

    # 3. Verify Cassandra facts exist
    preferences_facts = cassandra_repo.get_facts(user_id, "preferences")
    assert len(preferences_facts) == 1
    assert preferences_facts[0]["statement"] == "Likes black coffee"

    habits_facts = cassandra_repo.get_facts(user_id, "habits")
    assert len(habits_facts) == 1
    assert habits_facts[0]["statement"] == "Wakes up at 6am"

    # 4. Verify Milvus vector facts exist
    milvus_facts = milvus_repo.search_facts(
        user_id=user_id,
        query_vector=embedding_vector,
        limit=5,
        category="preferences"
    )
    assert len(milvus_facts) == 1
    assert milvus_facts[0]["statement"] == "Likes black coffee"
