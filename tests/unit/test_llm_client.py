"""
tests/unit/test_llm_client.py

Unit tests for Phase 11 LLM Client and gRPC Connection Pool.
Mocks gRPC channels and connectivity states to verify pool routing,
background health check replacement, and state-based circuit breaker triggers.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import grpc
import pytest

from app.clients.llm_client import AsyncGRPCConnectionPool, LLMClient
from app.core.config import settings
from app.core.exceptions import CircuitBreakerOpenException


@pytest.fixture
def mock_grpc_insecure_channel():
    """Mocks the insecure_channel creation function."""
    with patch("app.clients.llm_client.grpc_aio.insecure_channel") as mock_insecure:
        channels = []
        
        def create_mock(*args, **kwargs):
            ch = MagicMock()
            ch.get_state = MagicMock(return_value=grpc.ChannelConnectivity.READY)
            ch.close = AsyncMock()
            channels.append(ch)
            return ch
            
        mock_insecure.side_effect = create_mock
        yield mock_insecure, channels


@pytest.mark.asyncio
async def test_grpc_pool_initialization_and_routing(mock_grpc_insecure_channel):
    """Asserts that connection pool creates target size channels and uses round-robin routing."""
    mock_insecure, channels = mock_grpc_insecure_channel
    pool = AsyncGRPCConnectionPool(target="localhost:50051", pool_size=3)
    
    await pool.connect()
    assert len(channels) == 3
    assert len(pool._channels) == 3

    # Test round-robin routing
    ch1 = await pool.get_channel()
    ch2 = await pool.get_channel()
    ch3 = await pool.get_channel()
    ch4 = await pool.get_channel()

    assert ch1 is channels[0]
    assert ch2 is channels[1]
    assert ch3 is channels[2]
    assert ch4 is channels[0]  # Wraps around

    await pool.close()


@pytest.mark.asyncio
async def test_grpc_pool_health_check_replacement(mock_grpc_insecure_channel):
    """Verifies that channels in TRANSIENT_FAILURE are replaced during health checks."""
    mock_insecure, channels = mock_grpc_insecure_channel
    
    # Temporarily speed up health check interval for testing
    with patch.object(settings, "GRPC_HEALTH_CHECK_INTERVAL_SECONDS", 0.05):
        pool = AsyncGRPCConnectionPool(target="localhost:50051", pool_size=2)
        await pool.connect()
        
        # Keep track of original channels
        orig_ch1 = pool._channels[0]
        orig_ch2 = pool._channels[1]

        # Simulate channel 0 failure
        orig_ch1.get_state.return_value = grpc.ChannelConnectivity.TRANSIENT_FAILURE

        # Wait for the health check task to execute at least one sweep
        await asyncio.sleep(0.15)

        # Assert replacement channel created and old channel closed
        assert pool._channels[0] is not orig_ch1
        assert pool._channels[1] is orig_ch2
        orig_ch1.close.assert_awaited_once()

        await pool.close()


@pytest.mark.asyncio
async def test_circuit_breaker_flow(mock_grpc_insecure_channel):
    """Verifies circuit breaker CLOSED -> OPEN -> HALF_OPEN -> CLOSED/OPEN transitions."""
    mock_insecure, channels = mock_grpc_insecure_channel
    pool = AsyncGRPCConnectionPool(target="localhost:50051", pool_size=2)
    await pool.connect()

    llm_client = LLMClient(pool)
    assert llm_client.state == "CLOSED"

    # Define stub callable
    stub_mock = AsyncMock()
    stub_mock.return_value = "response-ok"

    # 1. Test CLOSED state success
    res = await llm_client.call_with_circuit_breaker(stub_mock, "arg-test")
    assert res == "response-ok"
    assert llm_client.failures == 0
    assert llm_client.state == "CLOSED"

    # Configure small breaker thresholds for fast tests
    with patch.object(settings, "CB_FAILURE_THRESHOLD", 2), \
         patch.object(settings, "CB_RECOVERY_TIMEOUT_SECONDS", 0.1), \
         patch.object(settings, "CB_HALF_OPEN_LIMIT", 2):

        # 2. Trigger failures to trip the breaker
        stub_mock.side_effect = Exception("gRPC timeout error")
        
        with pytest.raises(Exception, match="gRPC timeout error"):
            await llm_client.call_with_circuit_breaker(stub_mock)
        assert llm_client.failures == 1
        assert llm_client.state == "CLOSED"

        with pytest.raises(Exception, match="gRPC timeout error"):
            await llm_client.call_with_circuit_breaker(stub_mock)
        assert llm_client.failures == 2
        assert llm_client.state == "OPEN"

        # 3. Subsequent calls should immediately fail-fast
        with pytest.raises(CircuitBreakerOpenException):
            await llm_client.call_with_circuit_breaker(stub_mock)

        # 4. Wait for recovery timeout to transition to HALF_OPEN
        await asyncio.sleep(0.12)
        
        # Configure stub back to success
        stub_mock.side_effect = None
        stub_mock.return_value = "response-recovered"

        # 5. Perform first probe call in HALF_OPEN (limit = 2)
        res1 = await llm_client.call_with_circuit_breaker(stub_mock)
        assert res1 == "response-recovered"
        assert llm_client.state == "HALF_OPEN"

        # 6. Perform second probe call in HALF_OPEN (threshold reached -> CLOSED)
        res2 = await llm_client.call_with_circuit_breaker(stub_mock)
        assert res2 == "response-recovered"
        assert llm_client.state == "CLOSED"
        assert llm_client.failures == 0

        # 7. Trip back to OPEN instantly on failure during HALF_OPEN
        # Wait to get back to HALF_OPEN by failing in CLOSED -> OPEN
        stub_mock.side_effect = Exception("error-2")
        with pytest.raises(Exception):
            await llm_client.call_with_circuit_breaker(stub_mock)  # Fail 1
        with pytest.raises(Exception):
            await llm_client.call_with_circuit_breaker(stub_mock)  # Fail 2 -> OPEN
        assert llm_client.state == "OPEN"

        await asyncio.sleep(0.12)  # Wait for recovery
        
        # First probe in HALF_OPEN fails
        stub_mock.side_effect = Exception("probe-failed")
        with pytest.raises(Exception, match="probe-failed"):
            await llm_client.call_with_circuit_breaker(stub_mock)
            
        # Breaker should immediately trip back to OPEN (failures reset count doesn't delay it)
        assert llm_client.state == "OPEN"

    await pool.close()
