"""
app/factories/llm_factory.py

Factory Method pattern for instantiating LLM Providers based on configuration settings.
"""

import logging
from app.core.config import settings
from app.providers.base import BaseLLMProvider
from app.providers.groq_provider import GroqProvider
from app.providers.mock_provider import MockLLMProvider
from app.clients.groq_client import GroqClient

logger = logging.getLogger("memory_service.factories.llm_factory")


class LLMFactory:
    """
    Factory Method pattern for instantiating LLM Providers based on configuration settings.
    """

    @staticmethod
    def create_provider() -> BaseLLMProvider:
        provider_type = settings.LLM_PROVIDER.lower().strip()
        logger.info(f"Instantiating LLM Provider strategy for type: {provider_type}")

        if provider_type == "groq":
            groq_client = GroqClient(
                api_key=settings.GROQ_API_KEY,
                timeout=settings.LLM_TIMEOUT_SECONDS,
            )
            return GroqProvider(client=groq_client, default_model=settings.LLM_MODEL)
        elif provider_type == "mock":
            return MockLLMProvider()
        else:
            raise ValueError(f"Unsupported LLM provider type: {settings.LLM_PROVIDER}")
