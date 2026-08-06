"""
tests/unit/test_llm_business_layer.py

Unit tests for Phase 2: LLM Business Layer.
Tests request/response validation schemas, LLMService summarization calls,
and regex fact extraction output parsing.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from app.schemas.llm import (
    LLMMessage,
    SummarizeRequest,
    FactExtractRequest,
    ExtractedFact,
)
from app.services.llm_service import LLMService
from app.managers.llm_manager import LLMManager
from app.providers.mock_provider import MockLLMProvider


@pytest.mark.asyncio
async def test_llm_service_summarize_success():
    """Asserts LLMService formats summary prompts and resolves successfully."""
    # Reset singleton state
    LLMManager._instance = None
    
    mock_provider = MockLLMProvider()
    llm_manager = LLMManager(mock_provider)
    llm_service = LLMService(llm_manager)

    request = SummarizeRequest(
        previous_summary="User lives in Paris.",
        new_messages=[
            LLMMessage(role="user", content="I am moving to London tomorrow."),
            LLMMessage(role="assistant", content="Safe travels!"),
        ]
    )

    response = await llm_service.summarize(request)
    assert response.summary is not None
    assert response.summary.startswith("Mock summary generated for:")
    assert mock_provider.call_count == 1


@pytest.mark.asyncio
async def test_llm_service_extract_facts_parsing():
    """Asserts LLMService extracts facts and handles raw formatting/markdown variations."""
    # Reset singleton state
    LLMManager._instance = None
    
    mock_provider = MagicMock()
    # Mock return value containing multiple lines with markdown formats, empty lines, and bad formats
    mock_provider.generate = AsyncMock(
        return_value=(
            "Here are the facts extracted:\n"
            "- preferences:0.95:Likes dark mode layout\n"
            "* habits:0.7:Wakes up at 6 AM\n"
            "1. goals:0.85:Wants to learn Rust programming\n"
            "invalid_format:no_importance:missing_colons\n"
            "  plans:1.0:Traveling to Japan next month\n"
            "malformed:0.5\n"
        )
    )
    llm_manager = LLMManager(mock_provider)
    llm_service = LLMService(llm_manager)

    request = FactExtractRequest(summary="Paris moving. Wakes up at 6 AM.")
    response = await llm_service.extract_facts(request)

    assert len(response.facts) == 4
    
    # Check preferences
    f1 = response.facts[0]
    assert f1.category == "preferences"
    assert f1.importance == 0.95
    assert f1.statement == "Likes dark mode layout"

    # Check habits
    f2 = response.facts[1]
    assert f2.category == "habits"
    assert f2.importance == 0.7
    assert f2.statement == "Wakes up at 6 AM"

    # Check goals
    f3 = response.facts[2]
    assert f3.category == "goals"
    assert f3.importance == 0.85
    assert f3.statement == "Wants to learn Rust programming"

    # Check plans
    f4 = response.facts[3]
    assert f4.category == "plans"
    assert f4.importance == 1.0
    assert f4.statement == "Traveling to Japan next month"
