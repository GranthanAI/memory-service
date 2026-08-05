"""
app/clients/embedding_client.py

Decoupled Embedding Client Abstraction and Pluggable Adapters.
"""

import abc
import asyncio
import logging
import time
from typing import List

from app.core.config import settings
from app.core.exceptions import CircuitBreakerOpenException
from app.clients.llm_client import AsyncGRPCConnectionPool
from app.proto import embedding_pb2, embedding_pb2_grpc

logger = logging.getLogger("memory_service.clients.embedding_client")


class EmbeddingClient(abc.ABC):
    """
    Abstract base class defining the interface for generating semantic vector embeddings.
    """

    @abc.abstractmethod
    async def connect(self) -> None:
        """Initializes connection resource pools."""
        pass

    @abc.abstractmethod
    async def close(self) -> None:
        """Closes active client pools and releases connection resources."""
        pass

    @abc.abstractmethod
    async def generate_embedding(self, text: str) -> List[float]:
        """Generates a semantic embedding vector for the given string text."""
        pass


class GRPCEmbeddingClient(EmbeddingClient):
    """
    gRPC implementation of the EmbeddingClient.
    Connects to the LLM/Embedding Service and enforces a dedicated Circuit Breaker.
    """

    def __init__(self, target: str, pool_size: int):
        self.pool = AsyncGRPCConnectionPool(target, pool_size)
        
        # Dedicated circuit breaker state (decoupled from LLMClient)
        self._state = "CLOSED"  # CLOSED | OPEN | HALF_OPEN
        self._failures = 0
        self._last_failure_time = 0.0
        self._successes_in_half_open = 0
        self._lock = asyncio.Lock()

    @property
    def state(self) -> str:
        """Exposes circuit breaker state for testing and monitoring."""
        return self._state

    @property
    def failures(self) -> int:
        """Exposes consecutive failure counts for metrics."""
        return self._failures

    async def connect(self) -> None:
        await self.pool.connect()
        logger.info("GRPCEmbeddingClient connection pool started.")

    async def close(self) -> None:
        await self.pool.close()
        logger.info("GRPCEmbeddingClient connection pool closed.")

    async def generate_embedding(self, text: str) -> List[float]:
        """
        Invokes LLM Service GenerateEmbedding RPC guarded by the dedicated circuit breaker.
        """
        # 1. State check and check if recovery interval expired
        async with self._lock:
            if self._state == "OPEN":
                elapsed = time.monotonic() - self._last_failure_time
                if elapsed > settings.CB_RECOVERY_TIMEOUT_SECONDS:
                    self._state = "HALF_OPEN"
                    self._successes_in_half_open = 0
                    logger.info("GRPCEmbeddingClient circuit breaker recovery timeout expired. Transitioned to HALF_OPEN probe state.")
                else:
                    logger.warning(
                        f"Rejecting gRPC call: Circuit breaker is OPEN for Embedding service. Cool down: {elapsed:.2f}s"
                    )
                    raise CircuitBreakerOpenException("embedding-service")

        # 2. Invoke GenerateEmbedding stub on the Embedding Service
        async def embed_stub(channel) -> List[float]:
            stub = embedding_pb2_grpc.EmbeddingServiceStub(channel)
            request = embedding_pb2.EmbeddingRequest(
                text=text
            )
            response = await stub.GenerateEmbedding(
                request,
                timeout=settings.GRPC_TIMEOUT_SECONDS
            )
            return list(response.embedding)

        try:
            channel = await self.pool.get_channel()
            result = await embed_stub(channel)

            # 3. Successful execution - handle state transition
            async with self._lock:
                if self._state == "HALF_OPEN":
                    self._successes_in_half_open += 1
                    if self._successes_in_half_open >= settings.CB_HALF_OPEN_LIMIT:
                        self._state = "CLOSED"
                        self._failures = 0
                        self._successes_in_half_open = 0
                        logger.info("GRPCEmbeddingClient circuit breaker probe succeeded. Recovered state to CLOSED.")
                elif self._state == "CLOSED":
                    self._failures = 0  # Reset sequential count

            return result

        except Exception as e:
            # 4. Failed execution - trigger breaker trip rules
            async with self._lock:
                self._failures += 1
                self._last_failure_time = time.monotonic()
                self._successes_in_half_open = 0

                if self._state == "HALF_OPEN":
                    self._state = "OPEN"
                    logger.critical(
                        f"GRPCEmbeddingClient circuit breaker probe call FAILED in HALF_OPEN. Tripping back to OPEN. Error: {e}"
                    )
                elif self._state == "CLOSED" and self._failures >= settings.CB_FAILURE_THRESHOLD:
                    self._state = "OPEN"
                    logger.critical(
                        f"GRPCEmbeddingClient circuit breaker tripped to OPEN after {self._failures} consecutive failures. Error: {e}"
                    )

            raise e


class MockEmbeddingClient(EmbeddingClient):
    """
    Mock implementation of the EmbeddingClient for offline local development and unit tests.
    Generates deterministic vectors matching settings.VECTOR_DIMENSION without performing network requests.
    """

    def __init__(self, dimension: int = settings.VECTOR_DIMENSION):
        self.dimension = dimension

    async def connect(self) -> None:
        logger.info("MockEmbeddingClient connected.")

    async def close(self) -> None:
        logger.info("MockEmbeddingClient closed.")

    async def generate_embedding(self, text: str) -> List[float]:
        """Generates a deterministic float vector using the text hash."""
        import hashlib
        h = hashlib.sha256(text.encode("utf-8")).digest()
        vector = []
        for i in range(self.dimension):
            byte_idx = i % len(h)
            val = float(h[byte_idx]) / 255.0
            vector.append(val)
        return vector
