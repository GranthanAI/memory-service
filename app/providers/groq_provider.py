"""
app/providers/groq_provider.py

Groq API strategy implementation of BaseLLMProvider.
"""

import logging
from typing import List
from app.providers.base import BaseLLMProvider
from app.clients.groq_client import GroqClient

logger = logging.getLogger("memory_service.providers.groq_provider")


class GroqProvider(BaseLLMProvider):
    """
    Groq API strategy implementation of BaseLLMProvider.
    """

    def __init__(self, client: GroqClient, default_model: str):
        self.client = client
        self.default_model = default_model

    async def generate(
        self,
        messages: List[dict],
        model: str = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> str:
        target_model = model or self.default_model
        return await self.client.chat_completion(
            messages=messages,
            model=target_model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    async def check_health(self) -> bool:
        """
        Validates provider reachability using a simple prompt call.
        """
        try:
            # Send a simple ping messages list to Groq API
            messages = [{"role": "user", "content": "ping"}]
            await self.client.chat_completion(
                messages=messages,
                model=self.default_model,
                temperature=0.0,
                max_tokens=5,
            )
            return True
        except Exception as e:
            logger.error(f"GroqProvider health check failed: {e}")
            return False
