"""
app/clients/llm_client.py

gRPC client connection pool and state-based Circuit Breaker implementation
for resilient microservice invocations.
"""

import asyncio
import logging
import time
from typing import List, Optional

import grpc
from grpc import aio as grpc_aio

from app.core.config import settings
from app.core.exceptions import CircuitBreakerOpenException

logger = logging.getLogger("memory_service.clients.llm_client")


class AsyncGRPCConnectionPool:
    """
    Manages a pool of persistent async gRPC channels.
    Supports round-robin request routing and automatic replacement of dead channels.
    """

    def __init__(self, target: str, pool_size: Optional[int] = None):
        self.target = target
        self.pool_size = pool_size or settings.GRPC_POOL_SIZE
        self._channels: List[Optional[grpc_aio.Channel]] = []
        self._index = 0
        self._lock = asyncio.Lock()
        self._health_task: Optional[asyncio.Task] = None
        self._closed = False

    async def connect(self) -> None:
        """
        Populates the connection pool and starts the background health check loop.
        """
        logger.info(f"Initializing gRPC connection pool targeting {self.target} with size {self.pool_size}")
        async with self._lock:
            for _ in range(self.pool_size):
                channel = self._create_channel()
                self._channels.append(channel)
            
            self._closed = False
            self._health_task = asyncio.create_task(self._health_check_loop())

    def _create_channel(self) -> grpc_aio.Channel:
        """
        Helper to construct a single configured grpc.aio channel.
        """
        return grpc_aio.insecure_channel(
            self.target,
            options=[
                ('grpc.keepalive_time_ms', 30000),
                ('grpc.keepalive_timeout_ms', 10000),
                ('grpc.keepalive_permit_without_calls', 1),
                ('grpc.max_receive_message_length', 10 * 1024 * 1024),
            ]
        )

    async def get_channel(self) -> grpc_aio.Channel:
        """
        Acquires a gRPC channel using round-robin routing.
        """
        if self._closed:
            raise RuntimeError("Cannot get channel from a closed gRPC connection pool.")

        async with self._lock:
            for _ in range(self.pool_size):
                ch = self._channels[self._index]
                self._index = (self._index + 1) % self.pool_size
                if ch is not None:
                    return ch

        raise RuntimeError("No healthy gRPC channels available in pool.")

    async def _health_check_loop(self) -> None:
        """
        Background daemon that periodically replaces dead or disconnected channels.
        """
        while not self._closed:
            try:
                await asyncio.sleep(settings.GRPC_HEALTH_CHECK_INTERVAL_SECONDS)
                
                async with self._lock:
                    for i, channel in enumerate(self._channels):
                        if channel is None:
                            logger.info(f"gRPC channel at index {i} is None. Spawning replacement.")
                            self._channels[i] = self._create_channel()
                        else:
                            try:
                                state = channel.get_state(try_to_connect=True)
                                if state == grpc.ChannelConnectivity.TRANSIENT_FAILURE:
                                    logger.warning(
                                        f"gRPC channel {i} is in TRANSIENT_FAILURE. Closing and replacing channel."
                                    )
                                    try:
                                        await channel.close()
                                    except Exception as ce:
                                        logger.debug(f"Ignored channel close exception: {ce}")
                                    
                                    self._channels[i] = self._create_channel()
                            except Exception as e:
                                logger.error(f"Error checking state of gRPC channel {i}: {e}")
                                # Replace the channel if checking state throws unexpected exceptions
                                self._channels[i] = self._create_channel()

            except asyncio.CancelledError:
                break
            except Exception as outer_err:
                logger.error(f"Unexpected error in gRPC pool health check loop: {outer_err}")
                await asyncio.sleep(1.0)  # Avoid tight fail loop if sleep fails

    async def close(self) -> None:
        """
        Closes all channels and cancels the background health task.
        """
        async with self._lock:
            self._closed = True
            if self._health_task:
                self._health_task.cancel()
                try:
                    await self._health_task
                except asyncio.CancelledError:
                    pass
                self._health_task = None

            for i, ch in enumerate(self._channels):
                if ch is not None:
                    try:
                        await ch.close()
                    except Exception as e:
                        logger.warning(f"Error closing gRPC channel {i}: {e}")
            self._channels.clear()


class LLMClient:
    """
    gRPC wrapper client for LLM and Embedding services.
    Enforces a custom state-based Circuit Breaker around all call attempts.
    """

    def __init__(self, pool: AsyncGRPCConnectionPool):
        self.pool = pool
        self._failures = 0
        self._successes_in_half_open = 0
        self._state = "CLOSED"  # CLOSED | OPEN | HALF_OPEN
        self._last_failure_time = 0.0
        self._lock = asyncio.Lock()

    @property
    def state(self) -> str:
        """Exposes circuit breaker state for validation/monitoring."""
        return self._state

    @property
    def failures(self) -> int:
        """Exposes sequential failure count for metrics/validation."""
        return self._failures

    async def call_with_circuit_breaker(self, stub_fn, *args, **kwargs):
        """
        Executes a gRPC call stub_fn(channel, *args, **kwargs) guarded by the
        consecutive-failure state-based circuit breaker.
        """
        async with self._lock:
            # 1. State check and check if recovery interval expired
            if self._state == "OPEN":
                elapsed = time.monotonic() - self._last_failure_time
                if elapsed > settings.CB_RECOVERY_TIMEOUT_SECONDS:
                    self._state = "HALF_OPEN"
                    self._successes_in_half_open = 0
                    logger.info("Circuit breaker recovery timeout expired. Transitioned to HALF_OPEN probe state.")
                else:
                    logger.warning(f"Rejecting gRPC call: Circuit breaker is OPEN for LLM service. Cool down: {elapsed:.2f}s")
                    raise CircuitBreakerOpenException("llm-service")

        # 2. Acquire a channel and invoke target stub
        try:
            channel = await self.pool.get_channel()
            # Invoke call (executed outside lock to ensure optimal concurrency)
            result = await stub_fn(channel, *args, **kwargs)

            # 3. Successful execution - handle state transition
            async with self._lock:
                if self._state == "HALF_OPEN":
                    self._successes_in_half_open += 1
                    if self._successes_in_half_open >= settings.CB_HALF_OPEN_LIMIT:
                        self._state = "CLOSED"
                        self._failures = 0
                        self._successes_in_half_open = 0
                        logger.info("Circuit breaker probe succeeded. Recovered state to CLOSED.")
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
                    # Any failure in HALF_OPEN trips immediately back to OPEN
                    self._state = "OPEN"
                    logger.critical(
                        f"Circuit breaker probe call FAILED in HALF_OPEN. Tripping back to OPEN. Error: {e}"
                    )
                elif self._state == "CLOSED" and self._failures >= settings.CB_FAILURE_THRESHOLD:
                    self._state = "OPEN"
                    logger.critical(
                        f"Circuit breaker tripped to OPEN after {self._failures} consecutive failures. Error: {e}"
                    )

            raise e
