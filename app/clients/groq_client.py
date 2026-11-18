"""
app/clients/groq_client.py

Adapter client wrapper for the Groq Async SDK.
Handles low-level API operations, authentication, and communication.
"""

import logging
import re
from groq import AsyncGroq

logger = logging.getLogger("memory_service.clients.groq_client")


def supports_reasoning_effort(model: str) -> bool:
    """Groq accepts reasoning_effort for the gpt-oss family only; other models reject the parameter."""
    return (model or "").startswith("openai/gpt-oss")


class GroqClient:
    """
    Adapter client wrapper for the Groq Async SDK.
    Handles low-level API operations, authentication, and communication.
    """

    def __init__(self, api_key: str, timeout: float = 60.0):
        self.api_key = api_key
        self.timeout = timeout
        self.client = None

    def connect(self) -> None:
        """Initializes the underlying AsyncGroq SDK client."""
        if not self.client:
            import httpx
            from app.core.config import settings
            
            limits = httpx.Limits(
                max_connections=settings.LLM_POOL_MAX_CONNECTIONS,
                max_keepalive_connections=settings.LLM_POOL_MAX_KEEPALIVE_CONNECTIONS,
            )
            http_client = httpx.AsyncClient(limits=limits, timeout=self.timeout)
            self.client = AsyncGroq(
                api_key=self.api_key,
                timeout=self.timeout,
                http_client=http_client
            )
            logger.info(
                f"GroqClient initialized successfully with custom connection pooling: "
                f"max={settings.LLM_POOL_MAX_CONNECTIONS}, keepalive={settings.LLM_POOL_MAX_KEEPALIVE_CONNECTIONS}."
            )

    async def list_models(self) -> int:
        """GET /models: a free call (no tokens, no cost, no usage event). Returns how many models the key can see."""
        self.connect()
        page = await self.client.models.list()  # type: ignore[union-attr]
        return len(getattr(page, "data", None) or [])

    async def chat_completion(
        self,
        messages: list,
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        system: bool = False,
    ) -> str:
        """
        Sends a chat completion request to the Groq API.
        """
        if not self.client:
            self.connect()

        from app.clients.groq_budget import GroqBudgetDeferred, get_budget, parse_reset_seconds

        completions = self.client.chat.completions
        extra = {"reasoning_effort": "low"} if supports_reasoning_effort(model) else {}   # keeps reasoning from eating max_tokens
        try:
            if hasattr(type(completions), "with_raw_response"):  # the real SDK: keep the response headers
                raw = await completions.with_raw_response.create(messages=messages, model=model, temperature=temperature, max_tokens=max_tokens, **extra)
                await get_budget().record_headers(model, raw.headers)
                response = await raw.parse()
            else:
                response = await completions.create(messages=messages, model=model, temperature=temperature, max_tokens=max_tokens, **extra)
        except Exception as exc:
            headers = getattr(getattr(exc, "response", None), "headers", None)
            if headers is not None:
                await get_budget().record_headers(model, headers)
            if getattr(exc, "status_code", None) == 429 or "429" in str(exc) or "rate limit" in str(exc).lower():
                wait = None
                if headers is not None:
                    wait = parse_reset_seconds(headers.get("retry-after")) or parse_reset_seconds(headers.get("x-ratelimit-reset-tokens"))
                if wait is None:
                    m = re.search(r"try again in (\d+(?:\.\d+)?)\s*(ms|s)", str(exc), re.I)
                    wait = (float(m.group(1)) / (1000.0 if m.group(2).lower() == "ms" else 1.0)) if m else 20.0
                raise GroqBudgetDeferred(wait, model) from exc
            raise
        content = response.choices[0].message.content
        from app.services.usage_metering import record_call      # every paid call is metered (plan 4.9)
        record_call(provider="groq", model=model, usage=getattr(response, "usage", None), messages=messages, output_text=content, system=system)
        return content
