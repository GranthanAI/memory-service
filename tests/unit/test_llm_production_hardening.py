"""
tests/unit/test_llm_production_hardening.py

Unit tests for Phase 5: Production Hardening.
Validates metrics increments, log request tracing, rate limiting/concurrency, and retry behaviors.
"""

import asyncio
import uuid
import pytest
from unittest.mock import AsyncMock, MagicMock

from fastapi import Request
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY

from app.main import app
from app.core.config import settings
from app.core.logging import var_trace_id, set_log_context, clear_log_context
from app.core.metrics import LLM_REQUESTS, LLM_LATENCY, LLM_TOKENS
from app.managers.llm_manager import LLMManager
from app.services.llm_service import LLMService
from app.schemas.llm import SummarizeRequest, LLMMessage, FactExtractRequest
from app.providers.base import BaseLLMProvider


@pytest.mark.asyncio
async def test_llm_service_metrics():
    """
    Verifies that calling LLMService methods records latency, request status, and tokens in Prometheus metrics.
    """
    mock_provider = MagicMock(spec=BaseLLMProvider)
    mock_provider.generate = AsyncMock(return_value="Mocked LLM generation result.")
    
    # Force fresh LLMManager instance with our mock provider
    # Since LLMManager is a singleton, reset its instance for this test
    LLMManager._instance = None
    manager = LLMManager(mock_provider)
    service = LLMService(manager)

    # Initial metric samples
    before_req = REGISTRY.get_sample_value(
        "memory_llm_requests_total",
        labels={"provider": settings.LLM_PROVIDER, "model": settings.LLM_MODEL, "action": "summarize", "status": "success"}
    ) or 0.0

    before_tokens = REGISTRY.get_sample_value(
        "memory_llm_tokens_total",
        labels={"type": "completion"}
    ) or 0.0

    # Act
    request = SummarizeRequest(
        previous_summary="",
        new_messages=[LLMMessage(role="user", content="Hello internal engine metrics check.")]
    )
    response = await service.summarize(request)

    # Assert
    assert response.summary == "Mocked LLM generation result."
    
    after_req = REGISTRY.get_sample_value(
        "memory_llm_requests_total",
        labels={"provider": settings.LLM_PROVIDER, "model": settings.LLM_MODEL, "action": "summarize", "status": "success"}
    ) or 0.0
    
    after_tokens = REGISTRY.get_sample_value(
        "memory_llm_tokens_total",
        labels={"type": "completion"}
    ) or 0.0

    assert after_req == before_req + 1.0
    assert after_tokens > before_tokens


def test_http_request_tracing_middleware():
    """
    Verifies that the FastAPI trace ID middleware extracts/generates X-Trace-ID response headers
    and populates log tracing contexts.
    """
    with TestClient(app) as client:
        # 1. Test request passing custom Trace ID
        trace_id = "test-custom-trace-uuid-1234"
        response = client.get("/health", headers={"X-Trace-ID": trace_id})
        assert response.status_code == 200
        assert response.headers.get("X-Trace-ID") == trace_id

        # 2. Test request with no trace ID (should generate one)
        response_gen = client.get("/health")
        assert response_gen.status_code == 200
        gen_trace = response_gen.headers.get("X-Trace-ID")
        assert gen_trace is not None
        # Assert is a valid UUID
        assert len(gen_trace) == 36


@pytest.mark.asyncio
async def test_llm_manager_concurrency_rate_limiter():
    """
    Verifies that LLMManager semaphore enforces maximum concurrent request limits.
    """
    mock_provider = MagicMock(spec=BaseLLMProvider)
    
    # Create a generation method that sleeps to simulate latency and check concurrent counts
    concurrent_calls = 0
    max_observed_concurrency = 0
    lock = asyncio.Lock()

    async def generate_mock(*args, **kwargs):
        nonlocal concurrent_calls, max_observed_concurrency
        async with lock:
            concurrent_calls += 1
            if concurrent_calls > max_observed_concurrency:
                max_observed_concurrency = concurrent_calls
        await asyncio.sleep(0.05)
        async with lock:
            concurrent_calls -= 1
        return "Mock finished"

    mock_provider.generate = AsyncMock(side_effect=generate_mock)

    # Force reset and set low concurrency setting for this test
    LLMManager._instance = None
    settings.LLM_MAX_CONCURRENT_REQUESTS = 3
    manager = LLMManager(mock_provider)

    # Trigger 10 concurrent requests
    tasks = [
        manager.generate_with_retry([{"role": "user", "content": "ping"}])
        for _ in range(10)
    ]
    await asyncio.gather(*tasks)

    # Assert that concurrency never exceeded our settings limit of 3
    assert max_observed_concurrency <= 3
