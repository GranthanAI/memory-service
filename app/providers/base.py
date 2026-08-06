"""
app/providers/base.py

Abstract Strategy interface for LLM vendors/providers.
"""

import abc
from typing import List


class BaseLLMProvider(abc.ABC):
    """
    Abstract strategy interface for LLM vendors.
    """

    @abc.abstractmethod
    async def generate(
        self,
        messages: List[dict],
        model: str = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> str:
        """
        Executes a completion/generation request.
        """
        pass

    @abc.abstractmethod
    async def check_health(self) -> bool:
        """
        Performs a health check of the provider endpoint or service connection.
        """
        pass
