"""
tests/integration/test_llm_client_integration.py

Integration tests for Phase 11 LLM Client and gRPC Connection Pool.
Spawns a live mock gRPC server to test socket connectivity, round-robin routing,
health watcher channel replacements, and circuit breaker trip/recovery flows.
"""

import asyncio
import socket
import time
from unittest.mock import patch
import pytest
import grpc
from grpc import aio as grpc_aio

from app.clients.llm_client import AsyncGRPCConnectionPool, LLMClient
from app.core.config import settings
from app.core.exceptions import CircuitBreakerOpenException


# --- Mock Protobuf-less gRPC Server implementation ---

class GenericHandler(grpc.RpcMethodHandler):
    def __init__(self, unary_unary_fn):
        self.request_streaming = False
        self.response_streaming = False
        self.request_deserializer = None
        self.response_serializer = None
        self.unary_unary = unary_unary_fn


class MockGrpcServer:
    """
    Real TCP-bound gRPC server that uses a generic RPC handler to intercept calls
    without requiring precompiled pb2 modules.
    """
    def __init__(self, port: int):
        self.port = port
        self.server = None
        self.response_bytes = b"mock-llm-response"
        self.fail_requests = False

    async def _handle_call(self, request, context):
        if self.fail_requests:
            context.set_code(grpc.StatusCode.UNAVAILABLE)
            context.set_details("LLM Service currently offline.")
            raise RuntimeError("Unavailable")
        return self.response_bytes

    async def start(self):
        self.server = grpc_aio.server()
        
        class GenericService(grpc.GenericRpcHandler):
            def service(self_, handler_call_details):
                return GenericHandler(self._handle_call)

        self.server.add_generic_rpc_handlers((GenericService(),))
        self.server.add_insecure_port(f"127.0.0.1:{self.port}")
        await self.server.start()

    async def stop(self):
        if self.server:
            await self.server.stop(grace=0)


def get_free_port() -> int:
    """Allocates a free TCP port dynamically on localhost."""
    s = socket.socket()
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port


# --- Integration Tests ---

@pytest.fixture
def test_port():
    return get_free_port()


@pytest.mark.asyncio
async def test_llm_client_and_pool_integration(test_port):
    """
    Tests pool round-robin calls over a real gRPC network,
    socket disconnect detection, circuit breaker trip, and recovery.
    """
    # 1. Start mock gRPC server
    server = MockGrpcServer(port=test_port)
    await server.start()

    # 2. Configure connection pool targeting mock server
    pool = AsyncGRPCConnectionPool(target=f"127.0.0.1:{test_port}", pool_size=2)
    await pool.connect()

    llm_client = LLMClient(pool)

    # 3. Define raw unary_unary stub function to invoke over channel
    async def predict_stub(channel: grpc_aio.Channel, payload: bytes) -> bytes:
        fn = channel.unary_unary(
            "/MockLLMService/Predict",
            request_serializer=lambda x: x,
            response_deserializer=lambda x: x
        )
        return await fn(payload)

    # 4. Verify successful call in CLOSED state
    res = await llm_client.call_with_circuit_breaker(predict_stub, b"prompt")
    assert res == b"mock-llm-response"
    assert llm_client.state == "CLOSED"

    # Configure small breaker thresholds for fast tests
    with patch.object(settings, "CB_FAILURE_THRESHOLD", 2), \
         patch.object(settings, "CB_RECOVERY_TIMEOUT_SECONDS", 0.15), \
         patch.object(settings, "CB_HALF_OPEN_LIMIT", 1), \
         patch.object(settings, "GRPC_HEALTH_CHECK_INTERVAL_SECONDS", 0.05):

        # 5. Make server fail requests to trip the breaker
        server.fail_requests = True

        with pytest.raises(grpc.RpcError):
            await llm_client.call_with_circuit_breaker(predict_stub, b"prompt")  # Fail 1
        assert llm_client.failures == 1
        assert llm_client.state == "CLOSED"

        with pytest.raises(grpc.RpcError):
            await llm_client.call_with_circuit_breaker(predict_stub, b"prompt")  # Fail 2 -> trips OPEN
        assert llm_client.failures == 2
        assert llm_client.state == "OPEN"

        # 6. Subsequent calls immediately fail-fast
        with pytest.raises(CircuitBreakerOpenException):
            await llm_client.call_with_circuit_breaker(predict_stub, b"prompt")

        # 7. Stop server to force socket connections to go down
        await server.stop()

        # Let some time pass to verify background health watcher executes
        # While server is stopped, channels will enter TRANSIENT_FAILURE state
        await asyncio.sleep(0.1)

        # 8. Start server back up and configure to succeed
        server = MockGrpcServer(port=test_port)
        server.fail_requests = False
        await server.start()

        # 9. Await circuit breaker recovery cooldown
        await asyncio.sleep(0.1)

        # 10. Invoke call -> transitions breaker OPEN -> HALF_OPEN -> CLOSED (since limit = 1)
        res_recovered = await llm_client.call_with_circuit_breaker(predict_stub, b"prompt")
        assert res_recovered == b"mock-llm-response"
        assert llm_client.state == "CLOSED"
        assert llm_client.failures == 0

    # 11. Clean up
    await pool.close()
    await server.stop()
