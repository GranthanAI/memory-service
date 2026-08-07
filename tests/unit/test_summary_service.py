"""
tests/unit/test_summary_service.py

Unit tests for Incremental Summarization Service.
Mocks repository layers and internal LLM service to assert correct linear prompt construction,
chronological message sorting, and cache eviction patterns.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from datetime import datetime, timezone

from app.models.memory import MemoryState
from app.services.summary_service import SummaryService
from app.schemas.llm import SummarizeResponse


@pytest.fixture
def mock_dependencies():
    """Mocks memory_repo, cassandra_repo, and llm_service dependencies."""
    memory_repo = MagicMock()
    cassandra_repo = MagicMock()
    llm_service = MagicMock()

    memory_repo.get_snapshot = AsyncMock(return_value=None)
    memory_repo.get_summary = AsyncMock(return_value=None)
    memory_repo.get_recent_messages = AsyncMock(return_value=[])
    memory_repo.save_snapshot = AsyncMock()
    memory_repo.invalidate_conversation = AsyncMock()

    cassandra_repo.upsert_summary = MagicMock()
    llm_service.summarize = AsyncMock()

    return memory_repo, cassandra_repo, llm_service


@pytest.mark.asyncio
async def test_summary_service_raises_on_missing_snapshot(mock_dependencies):
    """Asserts that process_incremental_summary raises ValueError if snapshot doesn't exist."""
    memory_repo, cassandra_repo, llm_service = mock_dependencies
    service = SummaryService(memory_repo, cassandra_repo, llm_service)

    # Snapshot is missing (None)
    memory_repo.get_snapshot.return_value = None

    with pytest.raises(ValueError, match="Snapshot not found"):
        await service.process_incremental_summary(conversation_id="conv-1")


@pytest.mark.asyncio
async def test_summary_service_skips_on_no_messages(mock_dependencies):
    """Asserts that summarization is skipped if there are no recent messages."""
    memory_repo, cassandra_repo, llm_service = mock_dependencies
    service = SummaryService(memory_repo, cassandra_repo, llm_service)

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
    llm_service.summarize.assert_not_called()
    cassandra_repo.upsert_summary.assert_not_called()


@pytest.mark.asyncio
async def test_summary_service_incremental_algorithm_and_caching(mock_dependencies):
    """
    Asserts the Incremental Summarization Algorithm:
    - Reverses message list to chronological order (oldest first).
    - Calls internal LLMService with correct parameters.
    - Increments summary_version in Cassandra and evicts Redis cache.
    """
    memory_repo, cassandra_repo, llm_service = mock_dependencies
    service = SummaryService(memory_repo, cassandra_repo, llm_service)

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

    # Mock LLM Service response
    llm_service.summarize.return_value = SummarizeResponse(
        summary="New incremented summary text."
    )

    # 2. Run summarization
    updated_snap = await service.process_incremental_summary(
        conversation_id="conv-1",
        instructions="Preserve key facts from the previous summary. Be concise."
    )

    # 3. Verify assertions
    # Incremented version
    assert updated_snap["summary_version"] == 3
    assert updated_snap["last_summary_msg_id"] == "msg-3"  # the newest message

    # Verify LLM call parameters
    llm_service.summarize.assert_called_once()
    request = llm_service.summarize.call_args[0][0]
    assert request.previous_summary == "Previous summary text."
    assert len(request.new_messages) == 2
    assert request.new_messages[0].role == "assistant"
    assert request.new_messages[0].content == "Hello"
    assert request.new_messages[1].role == "user"
    assert request.new_messages[1].content == "Hello again"

    # Verify Cassandra write
    cassandra_repo.upsert_summary.assert_called_once()
    summary_record = cassandra_repo.upsert_summary.call_args[0][0]
    assert summary_record["conversation_id"] == "conv-1"
    assert summary_record["summary_text"] == "New incremented summary text."
    assert summary_record["summary_version"] == 3

    # Verify snapshot write and Redis eviction
    memory_repo.save_snapshot.assert_awaited_once_with(updated_snap)
    memory_repo.invalidate_conversation.assert_awaited_once_with("conv-1")
