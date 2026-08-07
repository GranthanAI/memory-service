"""
tests/unit/test_background_workers.py

Unit tests for Phase 19 Background Worker Daemons.
Verifies workers claim loops, state machine transitions, correct LLM integrations,
and error-handling failure triggers.
"""

import asyncio
import json
import uuid
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from app.core.config import settings
from app.models.memory import MemoryState
from app.workers.summary_worker import SummaryWorker
from app.workers.fact_worker import FactWorker
from app.workers.embedding_worker import EmbeddingWorker, parse_fact_string
from app.workers.delete_worker import DeleteWorker


@pytest.fixture
def mock_worker_dependencies():
    session = MagicMock()
    memory_service = MagicMock()
    memory_service.transition_state = AsyncMock(return_value={"fact_version": 1, "summary_version": 2})
    memory_service.handle_failure = AsyncMock()
    
    summary_service = MagicMock()
    summary_service.process_incremental_summary = AsyncMock(return_value={"summary_version": 2})
    
    memory_repo = MagicMock()
    memory_repo.get_summary = AsyncMock(return_value="Mock summary text.")
    memory_repo.get_recent_messages = AsyncMock()
    
    long_memory_service = MagicMock()
    long_memory_service.merge_user_facts = AsyncMock()
    
    llm_service = MagicMock()
    llm_service.extract_facts = AsyncMock()
    
    embedding_client = MagicMock()
    embedding_client.generate_embedding = AsyncMock()
    
    milvus_repo = MagicMock()
    milvus_repo.delete_fact = MagicMock()
    
    return session, memory_service, summary_service, memory_repo, long_memory_service, llm_service, embedding_client, milvus_repo


def mock_kafka_consumer(msg):
    """Helper to mock AIOKafkaConsumer start/stop/commit/getmany methods."""
    mock_consumer = MagicMock()
    mock_consumer.start = AsyncMock()
    mock_consumer.stop = AsyncMock()

    called = []

    async def getmany_mock(*args, **kwargs):
        await asyncio.sleep(0.001)
        if not called:
            called.append(True)
            return {MagicMock(): [msg]}
        return {}

    mock_consumer.getmany = getmany_mock
    mock_consumer.commit = AsyncMock()
    return mock_consumer


# ─── SummaryWorker Tests ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_summary_worker_success(mock_worker_dependencies):
    """SummaryWorker transitions snapshot states and schedules fact extraction on success."""
    session, memory_service, summary_service, _, _, _, _, _ = mock_worker_dependencies
    
    cassandra_repo = MagicMock()
    worker = SummaryWorker(session, memory_service, summary_service)
    worker.cassandra_repo = cassandra_repo

    payload = {
        "conversation_id": "conv-123",
        "user_id": "user-456",
        "version": 2,
        "attempt_count": 0
    }
    
    # Mock AIOKafka message poll
    msg = MagicMock(value=json.dumps(payload).encode("utf-8"))
    mock_consumer = mock_kafka_consumer(msg)
    with patch("app.workers.summary_worker.AIOKafkaConsumer", return_value=mock_consumer):
        await worker.start()
        await asyncio.sleep(0.05)
        await worker.stop()

    # Assertions
    memory_service.transition_state.assert_any_call("conv-123", MemoryState.SUMMARIZING)
    summary_service.process_incremental_summary.assert_called_once_with("conv-123")
    memory_service.transition_state.assert_any_call("conv-123", MemoryState.FACT_PENDING)
    
    cassandra_repo.insert_outbox_job.assert_called_once()
    outbox_arg = cassandra_repo.insert_outbox_job.call_args[0][0]
    assert outbox_arg["topic"] == settings.KAFKA_FACT_TOPIC
    assert "conv-123" in outbox_arg["payload"]


@pytest.mark.asyncio
async def test_summary_worker_failure_triggers_retry(mock_worker_dependencies):
    """SummaryWorker failures trigger state failure handler retry flows."""
    session, memory_service, summary_service, _, _, _, _, _ = mock_worker_dependencies
    
    worker = SummaryWorker(session, memory_service, summary_service)

    payload = {
        "conversation_id": "conv-123",
        "user_id": "user-456",
        "version": 2,
        "attempt_count": 1
    }
    
    summary_service.process_incremental_summary.side_effect = RuntimeError("LLM Unavailable")

    msg = MagicMock(value=json.dumps(payload).encode("utf-8"))
    mock_consumer = mock_kafka_consumer(msg)
    with patch("app.workers.summary_worker.AIOKafkaConsumer", return_value=mock_consumer):
        await worker.start()
        await asyncio.sleep(0.05)
        await worker.stop()

    memory_service.handle_failure.assert_called_once_with(
        conversation_id="conv-123",
        failed_state=MemoryState.SUMMARIZING,
        job_type="summary",
        payload=payload,
        error_msg="LLM Unavailable",
        attempt_count=1
    )


# ─── FactWorker Tests ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_fact_worker_success(mock_worker_dependencies):
    """FactWorker extracts facts via LLM service and schedules vector embedding."""
    session, memory_service, _, memory_repo, _, llm_service, _, _ = mock_worker_dependencies
    
    cassandra_repo = MagicMock()
    worker = FactWorker(session, memory_service, memory_repo, llm_service)
    worker.cassandra_repo = cassandra_repo

    payload = {
        "conversation_id": "conv-123",
        "user_id": "user-456",
        "version": 1,
        "attempt_count": 0
    }
    
    # Mock LLM return facts
    from app.schemas.llm import FactExtractResponse, ExtractedFact
    llm_service.extract_facts.return_value = FactExtractResponse(
        facts=[
            ExtractedFact(category="preferences", importance=0.85, statement="Likes tea"),
            ExtractedFact(category="habits", importance=0.7, statement="Wakes early")
        ]
    )

    msg = MagicMock(value=json.dumps(payload).encode("utf-8"))
    mock_consumer = mock_kafka_consumer(msg)
    with patch("app.workers.fact_worker.AIOKafkaConsumer", return_value=mock_consumer):
        await worker.start()
        await asyncio.sleep(0.05)
        await worker.stop()

    # Assertions
    memory_service.transition_state.assert_any_call("conv-123", MemoryState.EXTRACTING_FACTS)
    llm_service.extract_facts.assert_called_once()
    memory_service.transition_state.assert_any_call("conv-123", MemoryState.EMBEDDING_PENDING)
    
    cassandra_repo.insert_outbox_job.assert_called_once()
    outbox_arg = cassandra_repo.insert_outbox_job.call_args[0][0]
    assert outbox_arg["topic"] == settings.KAFKA_EMBEDDING_TOPIC
    
    parsed_payload = json.loads(outbox_arg["payload"])
    assert len(parsed_payload["facts"]) == 2
    assert "Likes tea" in parsed_payload["facts"][0]


# ─── EmbeddingWorker Tests ───────────────────────────────────────────────────

def test_parse_fact_string():
    """Asserts robust string parsing behaves with structured and fallback strings."""
    assert parse_fact_string("preferences:0.85:Likes tea") == ("preferences", 0.85, "Likes tea")
    assert parse_fact_string("invalid-string") == ("general", 0.5, "invalid-string")
    assert parse_fact_string("preferences:abc:Likes tea") == ("general", 0.5, "preferences:abc:Likes tea")


@pytest.mark.asyncio
async def test_embedding_worker_success(mock_worker_dependencies):
    """EmbeddingWorker generates embedding vectors and merges facts, transitioning state back to ACTIVE."""
    session, memory_service, _, _, long_memory_service, _, embedding_client, _ = mock_worker_dependencies
    
    worker = EmbeddingWorker(session, memory_service, long_memory_service, embedding_client)

    payload = {
        "conversation_id": "conv-123",
        "user_id": "user-456",
        "facts": ["preferences:0.9:User likes coffee"],
        "attempt_count": 0
    }
    
    # Mock LLM return embedding
    embedding_client.generate_embedding.return_value = [0.1] * settings.VECTOR_DIMENSION

    msg = MagicMock(value=json.dumps(payload).encode("utf-8"))
    mock_consumer = mock_kafka_consumer(msg)
    with patch("app.workers.embedding_worker.AIOKafkaConsumer", return_value=mock_consumer):
        await worker.start()
        await asyncio.sleep(0.05)
        await worker.stop()

    # Assertions
    embedding_client.generate_embedding.assert_called_once_with("User likes coffee")
    long_memory_service.merge_user_facts.assert_called_once()
    
    # Check transitions to READY and then ACTIVE
    memory_service.transition_state.assert_any_call("conv-123", MemoryState.READY)
    memory_service.transition_state.assert_any_call("conv-123", MemoryState.ACTIVE)


# ─── DeleteWorker Tests ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_delete_worker_specific_fact(mock_worker_dependencies):
    """DeleteWorker deletes specific facts from Cassandra and Milvus."""
    session, _, _, _, _, _, _, milvus_repo = mock_worker_dependencies
    
    cassandra_repo = MagicMock()
    worker = DeleteWorker(session, milvus_repo)
    worker.cassandra_repo = cassandra_repo

    fact_id = uuid.uuid4()
    payload = {
        "user_id": "user-123",
        "category": "preferences",
        "fact_id": str(fact_id)
    }

    msg = MagicMock(value=json.dumps(payload).encode("utf-8"))
    mock_consumer = mock_kafka_consumer(msg)
    with patch("app.workers.delete_worker.AIOKafkaConsumer", return_value=mock_consumer):
        await worker.start()
        await asyncio.sleep(0.05)
        await worker.stop()

    cassandra_repo.delete_fact.assert_called_once_with("user-123", "preferences", fact_id)
    milvus_repo.delete_fact.assert_called_once_with("user-123", str(fact_id))


@pytest.mark.asyncio
async def test_delete_worker_by_conversation(mock_worker_dependencies):
    """DeleteWorker purges all facts associated with a target conversation ID."""
    session, _, _, _, _, _, _, milvus_repo = mock_worker_dependencies
    
    cassandra_repo = MagicMock()
    worker = DeleteWorker(session, milvus_repo)
    worker.cassandra_repo = cassandra_repo

    payload = {
        "user_id": "user-123",
        "conversation_id": "conv-target"
    }

    fact_id_1 = uuid.uuid4()
    fact_id_2 = uuid.uuid4()

    # Mock get_facts to return matching and non-matching facts
    def get_facts_side_effect(user_id, category):
        if category == "preferences":
            return [
                {"fact_id": fact_id_1, "conversation_id": "conv-target"},
                {"fact_id": fact_id_2, "conversation_id": "conv-other"}
            ]
        return []

    cassandra_repo.get_facts.side_effect = get_facts_side_effect

    msg = MagicMock(value=json.dumps(payload).encode("utf-8"))
    mock_consumer = mock_kafka_consumer(msg)
    with patch("app.workers.delete_worker.AIOKafkaConsumer", return_value=mock_consumer):
        await worker.start()
        await asyncio.sleep(0.05)
        await worker.stop()

    # Verify delete triggered only on fact_id_1 (matching conv-target)
    cassandra_repo.delete_fact.assert_called_once_with("user-123", "preferences", fact_id_1)
    milvus_repo.delete_fact.assert_called_once_with("user-123", str(fact_id_1))


# ─── Graceful Shutdown Tests ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_summary_worker_graceful_shutdown(mock_worker_dependencies):
    """Verifies that stop() allows an in-flight job to finish and commit offsets before exiting."""
    session, memory_service, summary_service, _, _, _, _, _ = mock_worker_dependencies
    
    cassandra_repo = MagicMock()
    worker = SummaryWorker(session, memory_service, summary_service)
    worker.cassandra_repo = cassandra_repo

    payload = {
        "conversation_id": "conv-graceful",
        "user_id": "user-graceful",
        "version": 1,
        "attempt_count": 0
    }
    
    # Message processing is simulated to take some time
    async def process_mock(*args, **kwargs):
        await asyncio.sleep(0.1)  # Simulate slow LLM call
        return {"summary_version": 2}
    summary_service.process_incremental_summary.side_effect = process_mock

    # Mock AIOKafka message poll returning one batch and then blocking/empty
    msg = MagicMock(value=json.dumps(payload).encode("utf-8"))
    
    mock_consumer = MagicMock()
    mock_consumer.start = AsyncMock()
    mock_consumer.stop = AsyncMock()
    
    # getmany returns the message once, then returns empty dict
    called = []
    async def getmany_mock(*args, **kwargs):
        if not called:
            called.append(True)
            return {MagicMock(): [msg]}
        # Keep returning empty to avoid high CPU spin in test
        await asyncio.sleep(0.01)
        return {}
    mock_consumer.getmany = getmany_mock
    mock_consumer.commit = AsyncMock()

    with patch("app.workers.summary_worker.AIOKafkaConsumer", return_value=mock_consumer):
        # 1. Start the worker task in the background
        await worker.start()
        
        # 2. Wait slightly for the consumer to poll the message, then call stop()
        await asyncio.sleep(0.02)
        
        # 3. Stop should wait for the in-flight process (0.1s sleep) to finish
        await worker.stop()

    # 4. Assert that processing completed fully and offsets were committed
    memory_service.transition_state.assert_any_call("conv-graceful", MemoryState.SUMMARIZING)
    memory_service.transition_state.assert_any_call("conv-graceful", MemoryState.FACT_PENDING)
    mock_consumer.commit.assert_called_once()

