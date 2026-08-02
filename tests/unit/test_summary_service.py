"""
tests/unit/test_summary_service.py

Unit tests for Phase 13 Incremental Summarization Service.
Mocks repository layers and gRPC LLM clients to assert correct linear prompt construction,
chronological message sorting, and cache eviction patterns.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from datetime import datetime, timezone

from app.models.memory import MemoryState
from app.services.summary_service import SummaryService
from app.proto import llm_pb2


@pytest.fixture
def mock_dependencies():
    """Mocks memory_repo, cassandra_repo, and llm_client dependencies."""
    memory_repo = MagicMock()
    cassandra_repo = MagicMock()
    llm_client = MagicMock()

    memory_repo.get_snapshot = AsyncMock(return_value=None)
    memory_repo.get_summary = AsyncMock(return_value=None)
    memory_repo.get_recent_messages = AsyncMock(return_value=[])
    memory_repo.save_snapshot = AsyncMock()
    memory_repo.invalidate_conversation = AsyncMock()

    cassandra_repo.upsert_summary = MagicMock()

    llm_client.call_with_circuit_breaker = AsyncMock()

    return memory_repo, cassandra_repo, llm_client


@pytest.mark.asyncio
async def test_summary_service_raises_on_missing_snapshot(mock_dependencies):
    """Asserts that process_incremental_summary raises ValueError if snapshot doesn't exist."""
    memory_repo, cassandra_repo, llm_client = mock_dependencies
    service = SummaryService(memory_repo, cassandra_repo, llm_client)

    # Snapshot is missing (None)
    memory_repo.get_snapshot.return_value = None

    with pytest.raises(ValueError, match="Snapshot not found"):
        await service.process_incremental_summary(conversation_id="conv-1")


@pytest.mark.asyncio
async def test_summary_service_skips_on_no_messages(mock_dependencies):
    """Asserts that summarization is skipped if there are no recent messages."""
    memory_repo, cassandra_repo, llm_client = mock_dependencies
    service = SummaryService(memory_repo, cassandra_repo, llm_client)

    existing_snapshot = {
        "conversation_id": "conv-1",
        "user_id": "user-xyz",
        "summary_version": 1,
        "state": MemoryState.SUMMARIZING
    }
    memory_repo.get_snapshot.return_value = existing_snapshot

    # No messages (empty list)
    memory_repo.get_recent_messages.return_value = []

    res_snap = await service.process_incremental_summary(conversation_id="conv-1")
    
    assert res_snap == existing_snapshot
    llm_client.call_with_circuit_breaker.assert_not_called()
    cassandra_repo.upsert_summary.assert_not_called()


@pytest.mark.asyncio
async def test_summary_service_incremental_algorithm_and_caching(mock_dependencies):
    """
    Asserts the Incremental Summarization Algorithm:
    - Reverses message list to chronological order (oldest first).
    - Calls gRPC client with correct parameters.
    - Increments summary_version in Cassandra and evicts Redis cache.
    """
    memory_repo, cassandra_repo, llm_client = mock_dependencies
    service = SummaryService(memory_repo, cassandra_repo, llm_client)

    # 1. Mock existing snapshot and summary
    existing_snapshot = {
        "conversation_id": "conv-1",
        "user_id": "user-xyz",
        "summary_version": 2,
        "state": MemoryState.SUMMARIZING,
        "last_summary_msg_id": "msg-1"
    }
    memory_repo.get_snapshot.return_value = existing_snapshot
    memory_repo.get_summary.return_value = "Previous summary text."

    # Mock recent messages returned from repository (newest first)
    recent_messages = [
        {"message_id": "msg-3", "role": "user", "content": "Hello again", "created_at": datetime.now(timezone.utc)},
        {"message_id": "msg-2", "role": "assistant", "content": "Hello", "created_at": datetime.now(timezone.utc)}
    ]
    memory_repo.get_recent_messages.return_value = recent_messages

    # Mock gRPC call response
    llm_response = MagicMock()
    llm_response.summary_text = "New incremented summary text."
    
    # We capture the stub function passed to the circuit breaker
    async def mock_call_breaker(stub_fn, *args, **kwargs):
        # We need a mock channel to invoke the stub_fn
        mock_channel = MagicMock()
        mock_stub = MagicMock()
        mock_stub.GenerateSummary = AsyncMock(return_value=llm_response)
        
        # Patch the LLMServiceStub class constructor inside the stub
        with patch("app.proto.llm_pb2_grpc.LLMServiceStub", return_value=mock_stub) as mock_stub_class:
            result = await stub_fn(mock_channel)
            # Verify the parameters passed to stub
            mock_stub_class.assert_called_once_with(mock_channel)
            mock_stub.GenerateSummary.assert_called_once()
            request = mock_stub.GenerateSummary.call_args[0][0]
            assert isinstance(request, llm_pb2.SummaryRequest)
            assert request.previous_summary == "Previous summary text."
            assert request.instructions == "Preserve key facts from the previous summary. Be concise."
            
            # Assert chronological order of messages in payload (oldest first, i.e. msg-2, then msg-3)
            payload = json.loads(request.new_messages_json)
            assert len(payload) == 2
            assert payload[0]["message_id"] == "msg-2"
            assert payload[1]["message_id"] == "msg-3"
            
            return result

    llm_client.call_with_circuit_breaker.side_effect = mock_call_breaker

    # 2. Run summarization
    updated_snap = await service.process_incremental_summary(
        conversation_id="conv-1",
        instructions="Preserve key facts from the previous summary. Be concise."
    )

    # 3. Verify assertions
    # Incremented version
    assert updated_snap["summary_version"] == 3
    assert updated_snap["last_summary_msg_id"] == "msg-3" # the newest message

    # Verify Cassandra write
    cassandra_repo.upsert_summary.assert_called_once()
    summary_record = cassandra_repo.upsert_summary.call_args[0][0]
    assert summary_record["conversation_id"] == "conv-1"
    assert summary_record["summary_text"] == "New incremented summary text."
    assert summary_record["summary_version"] == 3

    # Verify snapshot write and Redis eviction
    memory_repo.save_snapshot.assert_awaited_once_with(updated_snap)
    memory_repo.invalidate_conversation.assert_awaited_once_with("conv-1")
