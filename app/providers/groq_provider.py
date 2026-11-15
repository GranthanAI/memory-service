"""
app/providers/groq_provider.py

Groq API strategy implementation of BaseLLMProvider.
"""

import logging
import time
from typing import List
from app.providers.base import BaseLLMProvider
from app.clients.groq_client import GroqClient

logger = logging.getLogger("memory_service.providers.groq_provider")


class GroqProvider(BaseLLMProvider):
    """
    Groq API strategy implementation of BaseLLMProvider.
    """

    meters_usage = True                      # its calls are paid and reported to usage_metering
    provider_name = "groq"

    # How long a health answer is reused. A readiness probe runs every few seconds; it must never cost anything.
    HEALTH_CACHE_SECONDS = 300.0
    HEALTH_FAILURE_CACHE_SECONDS = 30.0      # a failure is re-checked sooner, so a blip does not hold readiness down for minutes

    def __init__(self, client: GroqClient, default_model: str):
        self.client = client
        self.default_model = default_model
        self._health: tuple[float, bool] | None = None      # (monotonic time, answer) of the last probe
        self._last_call: tuple[float, bool] | None = None   # (monotonic time, succeeded) of the last REAL call

    async def generate(
        self,
        messages: List[dict],
        model: str = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> str:
        target_model = model or self.default_model
        try:
            out = await self.client.chat_completion(
                messages=messages,
                model=target_model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception:
            self._last_call = (time.monotonic(), False)
            raise
        self._last_call = (time.monotonic(), True)
        return out

    async def check_health(self) -> bool:
        """Readiness without ever calling a model (a completion costs money and writes a usage event).

        1. A real call that succeeded within the cache window is proof enough: healthy, no request at all.
        2. Otherwise Groq's GET /models (free), and the answer is kept for HEALTH_CACHE_SECONDS.
        """
        now = time.monotonic()
        if self._last_call and self._last_call[1] and now - self._last_call[0] < self.HEALTH_CACHE_SECONDS:
            return True
        if self._health:
            ttl = self.HEALTH_CACHE_SECONDS if self._health[1] else self.HEALTH_FAILURE_CACHE_SECONDS
            if now - self._health[0] < ttl:
                return self._health[1]
        try:
            await self.client.list_models()
            ok = True
        except Exception as e:
            logger.error(f"GroqProvider health check failed: {e}")
            ok = False
        self._health = (now, ok)
        return ok
