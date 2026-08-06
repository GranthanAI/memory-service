"""
app/providers/mock_provider.py

Mock LLM provider for offline testing and development.
"""

import json
from typing import List
from app.providers.base import BaseLLMProvider


class MockLLMProvider(BaseLLMProvider):
    """
    Mock LLM provider strategy for offline testing and development.
    """

    def __init__(self, responses: dict = None):
        self.responses = responses or {}
        self.call_count = 0

    async def generate(
        self,
        messages: List[dict],
        model: str = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> str:
        self.call_count += 1
        user_content = messages[-1]["content"] if messages else ""

        # Check if the user is asking to extract facts or summarize
        if "facts" in user_content.lower() or "extract" in user_content.lower():
            # Return list of facts inside JSON/text
            return json.dumps(
                ["preferences:0.9:Likes coding", "habits:0.8:Wakes up early"]
            )

        return f"Mock summary generated for: {user_content[:30]}..."

    async def check_health(self) -> bool:
        return True
