"""
app/clients/graph_client.py

Graph Service HTTP Client to fetch conversation ancestor lineage/summaries,
complete with connection pooling, automatic transient retries, and circuit breaker.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional
import httpx

from app.core.config import settings
from app.core.exceptions import CircuitBreakerOpenException

logger = logging.getLogger("memory_service.clients.graph_client")


class GraphClient:
    """
    HTTP Client interacting with the external Graph Service.
    Retrieves historical context lineage (parent/ancestor conversation summaries).
    Implements HTTP connection pooling, transient retries, and circuit breaker protection.
    """

    def __init__(self, base_url: str = settings.GRAPH_SERVICE_URL):
        self.base_url = base_url
        self.timeout_seconds = settings.GRAPH_SERVICE_TIMEOUT_MS / 1000.0
        self.client: Optional[httpx.AsyncClient] = None
        
        # Circuit breaker state
        self._state = "CLOSED"  # CLOSED | OPEN | HALF_OPEN
        self._failures = 0
        self._last_failure_time = 0.0
        self._successes_in_half_open = 0
        self._lock = asyncio.Lock()

    @property
    def state(self) -> str:
        """Exposes circuit breaker state for validation/monitoring."""
        return self._state

    @property
    def failures(self) -> int:
        """Exposes consecutive failures count for metrics/validation."""
        return self._failures

    async def connect(self) -> None:
        """Initializes connection pooling using an httpx AsyncClient."""
        if self.client is None:
            self.client = httpx.AsyncClient(
                limits=httpx.Limits(max_connections=50, max_keepalive_connections=20),
                timeout=httpx.Timeout(self.timeout_seconds)
            )
            logger.info("GraphClient connection pool initialized.")

    async def close(self) -> None:
        """Closes connection pooling and releases resources."""
        if self.client is not None:
            await self.client.aclose()
            self.client = None
            logger.info("GraphClient connection pool closed.")

    async def _request_with_retry(self, method: str, url: str, **kwargs) -> httpx.Response:
        """
        Executes HTTP requests with exponential backoff retries for transient failures.
        Transient failures include connection failures, request timeouts, and server errors (5xx).
        """
        max_attempts = 3
        base_delay = 0.5
        
        for attempt in range(1, max_attempts + 1):
            try:
                # Ensure client is connected
                if self.client is None:
                    await self.connect()
                
                response = await self.client.request(method, url, **kwargs)
                if response.status_code >= 500:
                    response.raise_for_status()
                return response
            except (httpx.TimeoutException, httpx.NetworkError, httpx.HTTPStatusError) as e:
                # Do not retry for non-server client status errors (e.g. 4xx) unless it's a request timeout
                if isinstance(e, httpx.HTTPStatusError) and e.response.status_code < 500:
                    raise e
                
                if attempt == max_attempts:
                    logger.error(f"Graph request failed after {max_attempts} attempts: {e}")
                    raise e
                
                delay = base_delay * (2 ** (attempt - 1))
                logger.warning(
                    f"Transient HTTP error on attempt {attempt}/{max_attempts}: {e}. "
                    f"Retrying in {delay:.2f}s..."
                )
                await asyncio.sleep(delay)
        
        raise RuntimeError("Request failed after exhausting retries.")

    async def get_ancestors(self, conversation_id: str) -> List[Dict[str, Any]]:
        """
        Sends a GET request to the Graph Service to fetch ancestor summaries.
        Guarded by the consecutive-failures circuit breaker and transient retries.
        """
        # Ensure client connection pool is active
        if self.client is None:
            await self.connect()

        # 1. State check and check if recovery interval expired
        async with self._lock:
            if self._state == "OPEN":
                elapsed = time.monotonic() - self._last_failure_time
                if elapsed > settings.CB_RECOVERY_TIMEOUT_SECONDS:
                    self._state = "HALF_OPEN"
                    self._successes_in_half_open = 0
                    logger.info("Circuit breaker recovery timeout expired. Transitioned to HALF_OPEN probe state.")
                else:
                    logger.warning(
                        f"Rejecting HTTP call: Circuit breaker is OPEN for Graph service. Cool down: {elapsed:.2f}s"
                    )
                    raise CircuitBreakerOpenException("graph-service")

        # 2. Invoke request guarded by retry loop
        url = f"{self.base_url}/conversations/{conversation_id}/ancestors"
        try:
            response = await self._request_with_retry("GET", url)
            response.raise_for_status()
            result = response.json()

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
