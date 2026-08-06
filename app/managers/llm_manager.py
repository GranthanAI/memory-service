"""
app/managers/llm_manager.py

Singleton Manager for LLM operations.
Wraps the configured LLM provider strategy and enforces circuit breaker,
exponential backoff retries, and timeouts.
"""

import asyncio
import logging
import time
from typing import List, Optional
from app.providers.base import BaseLLMProvider
from app.core.config import settings
from app.core.exceptions import CircuitBreakerOpenException

logger = logging.getLogger("memory_service.managers.llm_manager")


class LLMManager:
    """
    Singleton Manager for LLM operations.
    Wraps the configured LLM provider strategy and enforces circuit breaker,
    exponential backoff retries, and timeouts.
    """

    _instance: Optional["LLMManager"] = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(LLMManager, cls).__new__(cls)
        return cls._instance

    def __init__(self, provider: BaseLLMProvider):
        # Prevent re-initialization in singleton
        if hasattr(self, "_initialized") and self._initialized:
            return
        self.provider = provider
        self._state = "CLOSED"  # CLOSED | OPEN | HALF_OPEN
        self._failures = 0
        self._last_failure_time = 0.0
        self._successes_in_half_open = 0
        self._lock = asyncio.Lock()
        self._initialized = True

    @property
    def state(self) -> str:
        return self._state

    @property
    def failures(self) -> int:
        return self._failures

    async def generate_with_retry(
        self,
        messages: List[dict],
        model: str = None,
        temperature: float = None,
        max_tokens: int = None,
    ) -> str:
        """
        Delegates generation to the provider, wrapped in circuit breaker and retries.
        """
        # 1. Circuit Breaker Check
        async with self._lock:
            if self._state == "OPEN":
                elapsed = time.monotonic() - self._last_failure_time
                if elapsed > settings.CB_RECOVERY_TIMEOUT_SECONDS:
                    self._state = "HALF_OPEN"
                    self._successes_in_half_open = 0
                    logger.info(
                        "LLMManager circuit breaker recovery timeout expired. "
                        "Transitioned to HALF_OPEN probe state."
                    )
                else:
                    logger.warning(
                        f"Rejecting LLM call: Circuit breaker is OPEN. Cool down: {elapsed:.2f}s"
                    )
                    raise CircuitBreakerOpenException("llm-manager")

        # 2. Execution with Retries & Timeout
        max_retries = settings.LLM_MAX_RETRIES
        timeout = settings.LLM_TIMEOUT_SECONDS
        temp = temperature if temperature is not None else settings.LLM_TEMPERATURE
        tokens = max_tokens if max_tokens is not None else settings.LLM_MAX_TOKENS

        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                # Timeout wrapped
                result = await asyncio.wait_for(
                    self.provider.generate(
                        messages, model=model, temperature=temp, max_tokens=tokens
                    ),
                    timeout=timeout,
                )

                # 3. Successful execution - handle state transition
                async with self._lock:
                    if self._state == "HALF_OPEN":
                        self._successes_in_half_open += 1
                        if self._successes_in_half_open >= settings.CB_HALF_OPEN_LIMIT:
                            self._state = "CLOSED"
                            self._failures = 0
                            self._successes_in_half_open = 0
                            logger.info(
                                "LLMManager circuit breaker probe succeeded. Recovered state to CLOSED."
                            )
                    elif self._state == "CLOSED":
                        self._failures = 0
                return result

            except Exception as e:
                last_error = e
                logger.warning(f"LLM call failed (attempt {attempt}/{max_retries}): {e}")
                if attempt < max_retries:
                    # Exponential backoff: 2^attempt * 0.5s
                    backoff = (2**attempt) * 0.5
                    await asyncio.sleep(backoff)

        # 4. Failed execution - trigger breaker trip rules
        async with self._lock:
            self._failures += 1
            self._last_failure_time = time.monotonic()
            self._successes_in_half_open = 0

            if self._state == "HALF_OPEN":
                self._state = "OPEN"
                logger.critical(
                    f"LLMManager circuit breaker probe call FAILED in HALF_OPEN. "
                    f"Tripping back to OPEN. Error: {last_error}"
                )
            elif self._state == "CLOSED" and self._failures >= settings.CB_FAILURE_THRESHOLD:
                self._state = "OPEN"
                logger.critical(
                    f"LLMManager circuit breaker tripped to OPEN after {self._failures} "
                    f"consecutive failures. Error: {last_error}"
                )

        raise last_error

    async def check_health(self) -> bool:
        """
        Delegates health checking to the underlying provider.
        """
        try:
            return await self.provider.check_health()
        except Exception as e:
            logger.error(f"LLMManager health check failed: {e}")
            return False
