"""
app/clients/groq_client.py

Adapter client wrapper for the Groq Async SDK.
Handles low-level API operations, authentication, and communication.
"""

import logging
from groq import AsyncGroq

logger = logging.getLogger("memory_service.clients.groq_client")


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
            self.client = AsyncGroq(api_key=self.api_key, timeout=self.timeout)
            logger.info("GroqClient initialized successfully.")

    async def chat_completion(
        self,
        messages: list,
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> str:
        """
        Sends a chat completion request to the Groq API.
        """
        if not self.client:
            self.connect()

        response = await self.client.chat.completions.create(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content
