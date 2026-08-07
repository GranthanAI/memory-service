"""
tests/unit/test_api_grpc.py

Unit tests for Phase 3: APIs.
Tests internal HTTP REST endpoints (/internal/llm/summarize, /internal/llm/facts)
and internal gRPC server handlers.
"""

import asyncio
import socket
import pytest
from fastapi.testclient import TestClient
import grpc

from app.main import app
from app.core.config import settings
from app.core.container import Container
from app.proto import llm_pb2, llm_pb2_grpc
from app.proto.server import GRPCServer
from app.providers.mock_provider import MockLLMProvider
from app.managers.llm_manager import LLMManager
from app.services.llm_service import LLMService


def get_free_port() -> int:
    """Finds a free TCP port dynamically."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def http_client():
    """Provides a TestClient for testing HTTP endpoints."""
    # Ensure security API key is set for testing
    settings.API_KEY = "test-secret"
    with TestClient(app) as client:
        yield client


def test_http_summarize_endpoint_success(http_client):
    """Asserts POST /internal/llm/summarize behaves correctly under valid auth."""
    payload = {
        "previous_summary": "User prefers dark mode.",
        "new_messages": [
            {"role": "user", "content": "I like lightweight themes too."},
            {"role": "assistant", "content": "Noted!"}
        ]
    }
    
    headers = {"X-API-Key": "test-secret"}
    response = http_client.post(
        "/internal/llm/summarize",
        json=payload,
        headers=headers
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "summary" in data
    assert data["summary"].startswith("Mock summary generated for:")


def test_http_extract_facts_endpoint_success(http_client):
    """Asserts POST /internal/llm/facts behaves correctly under valid auth."""
    payload = {
        "summary": "User prefers dark mode. Habits: likes sleeping early."
    }
    
    headers = {"X-API-Key": "test-secret"}
    response = http_client.post(
        "/internal/llm/facts",
        json=payload,
        headers=headers
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "facts" in data
    assert len(data["facts"]) == 2
    assert data["facts"][0]["category"] == "preferences"
    assert data["facts"][0]["statement"] == "Likes coding"


def test_http_endpoints_unauthorized(http_client):
    """Asserts HTTP endpoints reject requests with missing/incorrect headers."""
    response = http_client.post("/internal/llm/summarize", json={})
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_grpc_server_and_handlers():
    """Starts the internal gRPC server on a dynamic port, makes calls via stub, and asserts output."""
    # Reset LLMManager singleton
    LLMManager._instance = None
    
    # 1. Initialize mock service
    mock_provider = MockLLMProvider()
    llm_manager = LLMManager(mock_provider)
    llm_service = LLMService(llm_manager)

    # 2. Start server dynamically
    grpc_port = get_free_port()
    server = GRPCServer(port=grpc_port, llm_service=llm_service)
    await server.start()

    # 3. Create client stub
    channel = grpc.aio.insecure_channel(f"localhost:{grpc_port}")
    stub = llm_pb2_grpc.LLMServiceStub(channel)

    try:
        # 4. Test Summarize RPC
        msg1 = llm_pb2.LLMMessage(role="user", content="Test message 1")
        req_summarize = llm_pb2.SummaryRequest(
            previous_summary="Prev text",
            new_messages=[msg1]
        )
        res_summarize = await stub.Summarize(req_summarize)
        assert res_summarize.summary.startswith("Mock summary generated for:")

        # 5. Test ExtractFacts RPC
        req_facts = llm_pb2.FactRequest(summary="Paris. Likes sleeping early.")
        res_facts = await stub.ExtractFacts(req_facts)
        assert len(res_facts.facts) == 2
        assert res_facts.facts[0].category == "preferences"
        assert res_facts.facts[0].statement == "Likes coding"

    finally:
        # 6. Tear down channel and server
        await channel.close()
        await server.stop(grace=0.0)
