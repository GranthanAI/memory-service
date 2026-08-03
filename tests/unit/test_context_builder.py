"""
tests/unit/test_context_builder.py

Unit tests for Phase 16 Structured Context Builder & Retrieval Service.
Mocks Graph Service timeouts, exceptions, and verifies concurrent context assembly.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.core.config import settings
from app.clients.graph_client import GraphClient
from app.services.retrieval_service import RetrievalService
from app.services.context_builder import ContextBuilder


@pytest.fixture
def mock_retrieval_service():
    """Mocks RetrievalService operations."""
    service = MagicMock(spec=RetrievalService)
    service.get_or_hydrate_snapshot = AsyncMock(return_value={
        "conversation_id": "conv-current",
        "user_id": "user-123",
        "state": "ACTIVE"
    })
    service.get_or_hydrate_summary = AsyncMock(return_value="Current context summary")
    service.get_or_hydrate_recent_messages = AsyncMock(return_value=[
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there!"}
    ])
    service.retrieve_relevant_facts = AsyncMock(return_value=[
        {"fact_id": "fact-1", "statement": "User likes coffee", "score": 0.85}
    ])
    return service


@pytest.fixture
def mock_graph_client():
    """Mocks GraphClient."""
    client = MagicMock(spec=GraphClient)
    client.get_ancestors = AsyncMock(return_value=[
        {"conversation_id": "conv-old-1", "summary": "Old summary content"}
    ])
    return client


@pytest.mark.asyncio
async def test_context_builder_success(mock_retrieval_service, mock_graph_client):
    """Asserts that context is fully assembled when all services respond successfully."""
    builder = ContextBuilder(mock_retrieval_service, mock_graph_client)

    context = await builder.build_context(
        user_id="user-123",
        conversation_id="conv-current",
        query_vector=[0.1] * 1536
    )

    assert context["conversation_id"] == "conv-current"
    assert context["user_id"] == "user-123"
    assert context["current_summary"] == "Current context summary"
    assert len(context["short_term_messages"]) == 2
    assert len(context["parent_summaries"]) == 1
    assert context["parent_summaries"][0]["conversation_id"] == "conv-old-1"
    assert len(context["relevant_facts"]) == 1
    
    assert context["metadata"]["parent_summaries_available"] is True
    assert context["metadata"]["facts_retrieved_count"] == 1


@pytest.mark.asyncio
async def test_context_builder_graph_timeout_fallback(mock_retrieval_service, mock_graph_client):
    """Asserts that Graph Service timeout degrades gracefully without blocking the response."""
    builder = ContextBuilder(mock_retrieval_service, mock_graph_client)

    # Make get_ancestors hang for 500ms
    async def delayed_ancestors(*args, **kwargs):
        await asyncio.sleep(0.5)
        return [{"conversation_id": "conv-old-1", "summary": "Old summary content"}]

    mock_graph_client.get_ancestors.side_effect = delayed_ancestors

    # Temporarily set graph timeout to 50ms for fast testing
    with patch.object(settings, "GRAPH_SERVICE_TIMEOUT_MS", 50):
        context = await builder.build_context(
            user_id="user-123",
            conversation_id="conv-current",
            query_vector=[0.1] * 1536
        )

    # Parent summaries must be empty, but rest of context is present
    assert context["current_summary"] == "Current context summary"
    assert len(context["parent_summaries"]) == 0
    assert context["metadata"]["parent_summaries_available"] is False
    assert len(context["relevant_facts"]) == 1


@pytest.mark.asyncio
async def test_context_builder_graph_exception_fallback(mock_retrieval_service, mock_graph_client):
    """Asserts that Graph Service HTTP/connection exceptions degrade gracefully."""
    builder = ContextBuilder(mock_retrieval_service, mock_graph_client)
    mock_graph_client.get_ancestors.side_effect = RuntimeError("Connection refused")

    context = await builder.build_context(
        user_id="user-123",
        conversation_id="conv-current",
        query_vector=[0.1] * 1536
    )

    assert context["current_summary"] == "Current context summary"
    assert len(context["parent_summaries"]) == 0
    assert context["metadata"]["parent_summaries_available"] is False


@pytest.mark.asyncio
async def test_retrieval_service_read_through_cache_miss():
    """Asserts that RetrievalService utilizes the MemoryRepository read-through queries correctly."""
    mock_mem_repo = MagicMock()
    mock_milvus_repo = MagicMock()

    mock_mem_repo.get_snapshot = AsyncMock(return_value={"state": "ACTIVE"})
    mock_mem_repo.get_recent_messages = AsyncMock(return_value=[{"message_id": "msg-1"}])
    mock_mem_repo.get_summary = AsyncMock(return_value="Hydrated summary")

    service = RetrievalService(mock_mem_repo, mock_milvus_repo)

    snap = await service.get_or_hydrate_snapshot("conv-1")
    assert snap == {"state": "ACTIVE"}
    mock_mem_repo.get_snapshot.assert_called_once_with("conv-1")

    msgs = await service.get_or_hydrate_recent_messages("conv-1", limit=10)
    assert msgs == [{"message_id": "msg-1"}]
    mock_mem_repo.get_recent_messages.assert_called_once_with("conv-1", limit=10)

    summary = await service.get_or_hydrate_summary("conv-1")
    assert summary == "Hydrated summary"
    mock_mem_repo.get_summary.assert_called_once_with("conv-1")
