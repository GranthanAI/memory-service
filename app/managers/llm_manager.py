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
        self._semaphore = asyncio.Semaphore(settings.LLM_MAX_CONCURRENT_REQUESTS)
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
        job_type: str = "memory",
    ) -> str:
        """
        Delegates generation to the provider, wrapped in circuit breaker and retries. `job_type` labels the metrics.
        An empty result is retried ONCE at once with the output cap doubled (reasoning can use up max_tokens).
        """
        if getattr(self.provider, "provider_name", "") == "groq":
            max_tokens = await self._groq_gate(messages, model, max_tokens)   # may raise GroqBudgetDeferred
        async with self._semaphore:
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

            provider_is_groq = getattr(self.provider, "provider_name", "") == "groq"
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
                    if provider_is_groq and not (result or "").strip():
                        result = await self._retry_empty(messages, model, temp, tokens, job_type, result, timeout)
                    return result

                except Exception as e:
                    if getattr(e, "wait_s", None) is not None:
                        raise  # GroqBudgetDeferred (Groq said 429 / wait): no retry here, no breaker failure; the job is re-queued
                    last_error = e
                    if isinstance(e, asyncio.TimeoutError) and getattr(self.provider, "meters_usage", False):
                        # the request reached the provider and may have been processed: report its estimated input
                        from app.services.usage_metering import record_call
                        record_call(provider=self.provider.provider_name, model=model or getattr(self.provider, "default_model", "?"),
                                    messages=messages, timed_out=True)
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

    async def _retry_empty(self, messages, model, temp, tokens, job_type, first, timeout):
        """The first answer was empty or blank: ask once more, now with twice the output cap (at most MEMORY_JOB_RETRY_MAX_TOKENS).
        If that is empty too the (empty) result goes back to the caller, whose own failure path handles it."""
        from app.clients.groq_budget import MEMORY_JOB_EMPTY_OUTPUT

        bigger = min(tokens * 2, settings.MEMORY_JOB_RETRY_MAX_TOKENS)
        MEMORY_JOB_EMPTY_OUTPUT.labels(type=job_type).inc()
        logger.warning("memory_job_empty_output type=%s max_tokens=%d: retrying once with max_tokens=%d", job_type, tokens, bigger)
        return await asyncio.wait_for(
            self.provider.generate(messages, model=model, temperature=temp, max_tokens=bigger), timeout=timeout)

    async def _groq_gate(self, messages: List[dict], model: Optional[str], max_tokens: Optional[int]) -> int:
        """Memory jobs are background work: cap their output, claim tokens from the shared Groq budget, and never take what an
        answer needs (GROQ_ANSWER_RESERVE_TOKENS). When it is short the job is DEFERRED until the reset, outside the circuit
        breaker and without using a retry attempt."""
        from app.clients.groq_budget import GroqBudgetDeferred, get_budget

        tokens = settings.MEMORY_JOB_MAX_TOKENS   # every memory job gets the same room; gpt-oss reasoning counts inside it
        from app.clients.groq_budget import estimate_tokens

        input_tokens = int(sum(len(str(m.get("content", ""))) for m in messages) / 3.0) + 50
        estimate = estimate_tokens(input_tokens, tokens, settings.GROQ_OUTPUT_ESTIMATE_TOKENS)   # never the full max_tokens
        target = model or getattr(self.provider, "default_model", "") or settings.MEMORY_LLM_MODEL or settings.LLM_MODEL
        wait = await get_budget().acquire(target, estimate, "memory")
        if wait > 0:
            raise GroqBudgetDeferred(wait, target)   # Groq's own reset time
        return tokens

    async def check_health(self) -> bool:
        """
        Delegates health checking to the underlying provider.
        """
        try:
            return await self.provider.check_health()
        except Exception as e:
            logger.error(f"LLMManager health check failed: {e}")
            return False
