"""
tests/unit/test_graph_and_embedding_clients.py

Unit tests for Phase 23 Graph Service Client and Phase 24 Embedding Client Abstraction.
Verifies connection pooling, transient HTTP retries, circuit breakers, and mock/grpc adapters.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
from app.core.config import settings
from app.core.exceptions import CircuitBreakerOpenException
from app.clients.graph_client import GraphClient
from app.clients.embedding_client import GRPCEmbeddingClient, MockEmbeddingClient
from app.core.container import Container


# ─── Graph Client Tests ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_graph_client_connection_pool_lifecycle():
    """GraphClient initializes and tears down its httpx AsyncClient connection pool cleanly."""
    client = GraphClient()
    assert client.client is None
    
    await client.connect()
    assert client.client is not None
    assert isinstance(client.client, httpx.AsyncClient)
    
    await client.close()
    assert client.client is None


@pytest.mark.asyncio
async def test_graph_client_transient_retry_success():
    """GraphClient retries on transient errors and succeeds within the retry limit."""
    client = GraphClient()
    await client.connect()
    
    # Mock request_with_retry to trigger mock call sequence: 500 error -> 500 error -> 200 success
    mock_resp_fail = MagicMock()
    mock_resp_fail.status_code = 500
    
    mock_resp_ok = MagicMock()
    mock_resp_ok.status_code = 200
    mock_resp_ok.json.return_value = [{"conversation_id": "conv-1", "summary": "ancestor summary"}]
    
    with patch.object(client.client, "request") as mock_req, \
         patch("asyncio.sleep") as mock_sleep:
        
        mock_req.side_effect = [mock_resp_fail, mock_resp_fail, mock_resp_ok]
        
        result = await client.get_ancestors("conv-123")
        
        assert len(result) == 1
        assert result[0]["summary"] == "ancestor summary"
        assert mock_req.call_count == 3
        assert mock_sleep.call_count == 2
        
    await client.close()


@pytest.mark.asyncio
async def test_graph_client_circuit_breaker_trips():
    """GraphClient circuit breaker trips to OPEN after consecutive failures threshold is breached."""
    client = GraphClient()
    await client.connect()
    
    mock_resp = MagicMock()
    mock_resp.status_code = 500
    
    with patch.object(client.client, "request", return_value=mock_resp), \
         patch("asyncio.sleep"):
        
        # Drive consecutive failures to trip threshold
        for _ in range(settings.CB_FAILURE_THRESHOLD):
            with pytest.raises(Exception):
                await client.get_ancestors("conv-123")
                
        # Circuit should now be OPEN
        assert client.state == "OPEN"
        assert client.failures == settings.CB_FAILURE_THRESHOLD
        
        # Subsequent requests must fail immediately with CircuitBreakerOpenException
        with pytest.raises(CircuitBreakerOpenException):
            await client.get_ancestors("conv-123")
            
    await client.close()


@pytest.mark.asyncio
async def test_graph_client_circuit_breaker_recovery():
    """GraphClient circuit breaker recovers from OPEN -> HALF_OPEN -> CLOSED on successful probes."""
    client = GraphClient()
    client._state = "OPEN"
    client._failures = settings.CB_FAILURE_THRESHOLD
    client._last_failure_time = 0.0  # Make cool down period immediately expired
    
    await client.connect()
    
    mock_resp_ok = MagicMock()
    mock_resp_ok.status_code = 200
    mock_resp_ok.json.return_value = []
    
    with patch.object(client.client, "request", return_value=mock_resp_ok):
        # 1st call transitions to HALF_OPEN probe state
        await client.get_ancestors("conv-123")
        assert client.state == "HALF_OPEN"
        
        # 2nd call reaches CB_HALF_OPEN_LIMIT (default is 2) and recovers to CLOSED
        await client.get_ancestors("conv-123")
        assert client.state == "CLOSED"
        assert client.failures == 0
        
    await client.close()


# ─── Embedding Client Tests ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_mock_embedding_client():
    """MockEmbeddingClient generates deterministic vectors matching the configured dimension."""
    client = MockEmbeddingClient(dimension=256)
    await client.connect()
    
    vec1 = await client.generate_embedding("hello world")
    vec2 = await client.generate_embedding("hello world")
    vec3 = await client.generate_embedding("different string")
    
    assert len(vec1) == 256
    assert vec1 == vec2  # Deterministic hashing
    assert vec1 != vec3  # String mismatch yields different float vector
    
    await client.close()


@pytest.mark.asyncio
async def test_grpc_embedding_client_circuit_breaker():
    """GRPCEmbeddingClient circuit breaker trips to OPEN on failures and rejects downstream calls."""
    client = GRPCEmbeddingClient(target="localhost:50051", pool_size=1)
    
    # Mock pool and get_channel to raise error
    with patch.object(client.pool, "connect", new_callable=AsyncMock), \
         patch.object(client.pool, "close", new_callable=AsyncMock), \
         patch.object(client.pool, "get_channel", side_effect=RuntimeError("connection error")):
         
        await client.connect()
        
        for _ in range(settings.CB_FAILURE_THRESHOLD):
            with pytest.raises(RuntimeError):
                await client.generate_embedding("test")
                
        # Circuit should now be OPEN
        assert client.state == "OPEN"
        assert client.failures == settings.CB_FAILURE_THRESHOLD
        
        # Subsequent requests fail with CircuitBreakerOpenException
        with pytest.raises(CircuitBreakerOpenException):
            await client.generate_embedding("test")
            
        await client.close()


# ─── Container Pluggability Tests ───────────────────────────────────────────

@pytest.mark.asyncio
async def test_container_wires_mock_embedding_client():
    """Container instantiates MockEmbeddingClient when settings.EMBEDDING_CLIENT_TYPE is 'mock'."""
    container = Container()
    
    with patch("app.core.container.get_session"), \
         patch("app.core.container.get_redis_client"), \
         patch("app.core.container.MilvusRepository"), \
         patch("app.core.container.AsyncGRPCConnectionPool") as mock_grpc_pool, \
         patch("app.core.container.settings") as mock_settings:
         
        mock_settings.EMBEDDING_CLIENT_TYPE = "mock"
        mock_settings.VECTOR_DIMENSION = 1536
        mock_settings.LLM_SERVICE_HOST = "localhost"
        mock_settings.LLM_SERVICE_PORT = 50051
        mock_settings.GRAPH_SERVICE_URL = "http://localhost:8001"
        
        # Mock connection pool
        pool_mock = MagicMock()
        pool_mock.connect = AsyncMock()
        pool_mock.close = AsyncMock()
        mock_grpc_pool.return_value = pool_mock
        
        await container.init_resources()
        
        assert isinstance(container.embedding_client, MockEmbeddingClient)
        assert container.embedding_client.dimension == 1536
        
        await container.shutdown_resources()


@pytest.mark.asyncio
async def test_container_wires_grpc_embedding_client():
    """Container instantiates GRPCEmbeddingClient when settings.EMBEDDING_CLIENT_TYPE is 'grpc'."""
    container = Container()
    
    with patch("app.core.container.get_session"), \
         patch("app.core.container.get_redis_client"), \
         patch("app.core.container.MilvusRepository"), \
         patch("app.core.container.AsyncGRPCConnectionPool") as mock_grpc_pool, \
         patch("app.core.container.settings") as mock_settings:
         
        mock_settings.EMBEDDING_CLIENT_TYPE = "grpc"
        mock_settings.GRPC_POOL_SIZE = 5
        mock_settings.LLM_SERVICE_HOST = "localhost"
        mock_settings.LLM_SERVICE_PORT = 50051
        mock_settings.GRAPH_SERVICE_URL = "http://localhost:8001"
        
        # Mock connection pool
        pool_mock = MagicMock()
        pool_mock.connect = AsyncMock()
        pool_mock.close = AsyncMock()
        mock_grpc_pool.return_value = pool_mock
        
        await container.init_resources()
        
        assert isinstance(container.embedding_client, GRPCEmbeddingClient)
        
        await container.shutdown_resources()
