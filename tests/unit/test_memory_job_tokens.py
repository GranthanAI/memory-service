"""R1: memory jobs leave room to answer: reasoning_effort low (gpt-oss only), MEMORY_JOB_MAX_TOKENS, one doubled retry when empty."""

import asyncio
from types import SimpleNamespace

import pytest

from app.clients.groq_budget import MEMORY_JOB_EMPTY_OUTPUT
from app.clients.groq_client import GroqClient, supports_reasoning_effort
from app.core.config import settings
from app.managers.llm_manager import LLMManager

pytestmark = pytest.mark.asyncio
MSG = [{"role": "user", "content": "summarise this"}]


class Completions:
    def __init__(self):
        self.calls = []

    async def create(self, **kw):
        self.calls.append(kw)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2, total_tokens=12),
        )


def client_with(completions):
    c = GroqClient("k")
    c.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return c


async def test_reasoning_effort_low_is_sent_for_gpt_oss_only():
    assert supports_reasoning_effort("openai/gpt-oss-20b") and supports_reasoning_effort("openai/gpt-oss-120b")
    assert not supports_reasoning_effort("llama-3.3-70b-versatile") and not supports_reasoning_effort("qwen/qwen3-32b")
    comp = Completions()
    c = client_with(comp)
    await c.chat_completion(MSG, "openai/gpt-oss-20b", max_tokens=1024)
    await c.chat_completion(MSG, "llama-3.3-70b-versatile", max_tokens=1024)
    assert comp.calls[0]["reasoning_effort"] == "low"
    assert "reasoning_effort" not in comp.calls[1]                   # other models would reject the parameter


def test_the_default_job_cap_is_1024_and_the_retry_cap_2048():
    assert settings.MEMORY_JOB_MAX_TOKENS == 1024 and settings.MEMORY_JOB_RETRY_MAX_TOKENS == 2048


def manager(answers):
    """A manager whose Groq provider returns `answers` in order and records the max_tokens it was asked for."""
    asked = []

    class Provider:
        provider_name, default_model, meters_usage = "groq", "openai/gpt-oss-20b", False

        async def generate(self, messages, model=None, temperature=None, max_tokens=None):
            asked.append(max_tokens)
            return answers.pop(0)

    m = LLMManager.__new__(LLMManager)
    m.provider, m._state, m._failures, m._last_failure_time, m._successes_in_half_open = Provider(), "CLOSED", 0, 0.0, 0
    m._lock, m._semaphore = asyncio.Lock(), asyncio.Semaphore(2)
    return m, asked


async def test_a_memory_job_asks_for_the_job_cap_whatever_the_caller_wanted():
    m, asked = manager(["a summary"])
    assert await m.generate_with_retry(MSG, max_tokens=200, job_type="summary") == "a summary"
    assert asked == [settings.MEMORY_JOB_MAX_TOKENS]


async def test_an_empty_result_is_retried_once_with_the_cap_doubled():
    before = MEMORY_JOB_EMPTY_OUTPUT.labels(type="summary")._value.get()
    m, asked = manager(["   \n", "now there is text"])
    assert await m.generate_with_retry(MSG, job_type="summary") == "now there is text"
    assert asked == [1024, 2048]
    assert MEMORY_JOB_EMPTY_OUTPUT.labels(type="summary")._value.get() == before + 1


async def test_the_retry_cap_never_exceeds_the_limit():
    m, asked = manager(["", "ok"])
    settings_cap = settings.MEMORY_JOB_RETRY_MAX_TOKENS
    settings.MEMORY_JOB_MAX_TOKENS, old = 1800, settings.MEMORY_JOB_MAX_TOKENS
    try:
        await m.generate_with_retry(MSG, job_type="fact")
    finally:
        settings.MEMORY_JOB_MAX_TOKENS = old
    assert asked == [1800, settings_cap]                              # 3,600 is capped at 2,048


async def test_still_empty_after_the_retry_goes_back_to_the_callers_failure_path():
    m, asked = manager(["", ""])
    assert await m.generate_with_retry(MSG, job_type="summary") == ""   # llm_service then raises "summary model returned an empty result"
    assert len(asked) == 2                                               # exactly one retry, not a loop


async def test_a_non_empty_result_passes_straight_through():
    m, asked = manager(["fine"])
    assert await m.generate_with_retry(MSG, job_type="fact") == "fine"
    assert len(asked) == 1
