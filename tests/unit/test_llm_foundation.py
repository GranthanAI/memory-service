"""
tests/unit/test_llm_foundation.py

Unit tests for Phase 1: LLM Foundation.
Tests configuration, client adapter, provider strategy, factory method,
and circuit breaker retry logic in LLMManager.
"""

import asyncio
import time
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.core.config import settings
from app.core.exceptions import CircuitBreakerOpenException
from app.clients.groq_client import GroqClient
from app.providers.groq_provider import GroqProvider
from app.providers.mock_provider import MockLLMProvider
from app.factories.llm_factory import LLMFactory
from app.managers.llm_manager import LLMManager


def test_system_settings_llm_loading():
    """Asserts that all new LLM configurations load with expected defaults."""
    assert settings.LLM_PROVIDER == "groq"
    assert settings.LLM_MODEL == "llama-3.3-70b-versatile"
    assert settings.LLM_TIMEOUT_SECONDS == 60.0
    assert settings.LLM_MAX_RETRIES == 3
    assert settings.LLM_TEMPERATURE == 0.2
    assert settings.LLM_MAX_TOKENS == 1024


@pytest.mark.asyncio
async def test_mock_llm_provider_generation():
    """Asserts MockLLMProvider generates mock output and checks health successfully."""
    provider = MockLLMProvider()
    
    # Test generation for summary
    res_summary = await provider.generate([{"role": "user", "content": "summarize this"}])
    assert "Mock summary generated" in res_summary
    assert provider.call_count == 1

    # Test generation for facts
    res_facts = await provider.generate([{"role": "user", "content": "extract facts"}])
    assert "Likes coding" in res_facts
    assert provider.call_count == 2

    # Test health check
    assert await provider.check_health() is True


def test_llm_factory_selection():
    """Asserts LLMFactory creates the expected provider based on configuration."""
    with patch("app.factories.llm_factory.settings") as mock_settings:
        # Mock strategy
        mock_settings.LLM_PROVIDER = "mock"
        provider = LLMFactory.create_provider()
        assert isinstance(provider, MockLLMProvider)

        # Invalid strategy
        mock_settings.LLM_PROVIDER = "invalid_vendor"
        with pytest.raises(ValueError, match="Unsupported LLM provider type"):
            LLMFactory.create_provider()


@pytest.mark.asyncio
async def test_llm_manager_singleton():
    """Asserts LLMManager enforces singleton constraints."""
    provider_a = MockLLMProvider()
    provider_b = MockLLMProvider()
    
    manager_1 = LLMManager(provider_a)
    manager_2 = LLMManager(provider_b)
    
    assert manager_1 is manager_2
    assert manager_1.provider is provider_a  # First initialized provider is retained


@pytest.mark.asyncio
async def test_llm_manager_generate_success():
    """Asserts LLMManager successfully delegates call to provider."""
    provider = MockLLMProvider()
    # Reset singleton state for testing
    LLMManager._instance = None
    manager = LLMManager(provider)

    res = await manager.generate_with_retry([{"role": "user", "content": "test text"}])
    assert "Mock summary generated" in res
    assert provider.call_count == 1


@pytest.mark.asyncio
async def test_llm_manager_retries_and_fails():
    """Asserts LLMManager retries transient exceptions and eventually fails."""
    provider = MagicMock()
    provider.generate = AsyncMock(side_effect=RuntimeError("Transient API Error"))
    
    # Reset singleton state
    LLMManager._instance = None
    manager = LLMManager(provider)

    with patch.object(settings, "LLM_MAX_RETRIES", 3), \
         patch.object(settings, "LLM_TIMEOUT_SECONDS", 5.0), \
         patch.object(settings, "LLM_TEMPERATURE", 0.2), \
         patch.object(settings, "LLM_MAX_TOKENS", 1024), \
         patch("asyncio.sleep") as mock_sleep:

        with pytest.raises(RuntimeError, match="Transient API Error"):
            await manager.generate_with_retry([{"role": "user", "content": "hello"}])
        
        # Verify provider.generate called 3 times
        assert provider.generate.call_count == 3
        assert mock_sleep.call_count == 2  # sleep called between retries


@pytest.mark.asyncio
async def test_llm_manager_circuit_breaker_trips_to_open():
    """Asserts LLMManager trips circuit breaker to OPEN on sequential failures."""
    provider = MagicMock()
    provider.generate = AsyncMock(side_effect=RuntimeError("API Failure"))
    
    # Reset singleton state
    LLMManager._instance = None
    manager = LLMManager(provider)

    with patch.object(settings, "CB_FAILURE_THRESHOLD", 2), \
         patch.object(settings, "LLM_MAX_RETRIES", 1), \
         patch.object(settings, "LLM_TIMEOUT_SECONDS", 5.0), \
         patch.object(settings, "LLM_TEMPERATURE", 0.2), \
         patch.object(settings, "LLM_MAX_TOKENS", 1024), \
         patch("asyncio.sleep") as mock_sleep:

        # First failure
        with pytest.raises(RuntimeError):
            await manager.generate_with_retry([{"role": "user", "content": "test"}])
        assert manager.state == "CLOSED"
        assert manager.failures == 1

        # Second failure -> Trips to OPEN
        with pytest.raises(RuntimeError):
            await manager.generate_with_retry([{"role": "user", "content": "test"}])
        assert manager.state == "OPEN"
        assert manager.failures == 2

        # Subsequent requests reject instantly
        with pytest.raises(CircuitBreakerOpenException):
            await manager.generate_with_retry([{"role": "user", "content": "test"}])


@pytest.mark.asyncio
async def test_llm_manager_circuit_breaker_recovery():
    """Asserts LLMManager recovers from OPEN to HALF_OPEN and then CLOSED."""
    provider = MagicMock()
    provider.generate = AsyncMock(side_effect=RuntimeError("API Failure"))
    
    # Reset singleton state
    LLMManager._instance = None
    manager = LLMManager(provider)

    with patch.object(settings, "CB_FAILURE_THRESHOLD", 1), \
         patch.object(settings, "CB_RECOVERY_TIMEOUT_SECONDS", 0.05), \
         patch.object(settings, "CB_HALF_OPEN_LIMIT", 2), \
         patch.object(settings, "LLM_MAX_RETRIES", 1), \
         patch.object(settings, "LLM_TIMEOUT_SECONDS", 5.0), \
         patch.object(settings, "LLM_TEMPERATURE", 0.2), \
         patch.object(settings, "LLM_MAX_TOKENS", 1024):

        # 1. Trip breaker to OPEN
        with pytest.raises(RuntimeError):
            await manager.generate_with_retry([{"role": "user", "content": "test"}])
        assert manager.state == "OPEN"

        # 2. Cool down timeout expiry (transition to HALF_OPEN)
        await asyncio.sleep(0.06)

        # 3. Successful probe calls in HALF_OPEN
        provider.generate = AsyncMock(return_value="Success response")
        
        # Probe call 1
        res1 = await manager.generate_with_retry([{"role": "user", "content": "test"}])
        assert res1 == "Success response"
        assert manager.state == "HALF_OPEN"

        # Probe call 2 -> Recovers to CLOSED
        res2 = await manager.generate_with_retry([{"role": "user", "content": "test"}])
        assert res2 == "Success response"
        assert manager.state == "CLOSED"
        assert manager.failures == 0
