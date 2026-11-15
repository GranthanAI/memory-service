"""G8: readiness never makes a paid model call. It uses Groq's free GET /models (cached) or the outcome of a real call."""

import time

import pytest

from app.core.metrics import MEMORY_SYSTEM_CALLS, MEMORY_UNMETERED_CALLS
from app.providers.groq_provider import GroqProvider

pytestmark = pytest.mark.asyncio


class FakeClient:
    def __init__(self, fail_models: bool = False):
        self.completions = 0
        self.model_lists = 0
        self.fail_models = fail_models

    async def chat_completion(self, **_kw):
        self.completions += 1
        return "ok"

    async def list_models(self):
        self.model_lists += 1
        if self.fail_models:
            raise RuntimeError("groq down")
        return 12


@pytest.fixture
def clock(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])
    return now


def counts():
    return MEMORY_SYSTEM_CALLS._value.get(), MEMORY_UNMETERED_CALLS._value.get()


async def test_a_readiness_probe_makes_no_completion_and_no_usage_event(clock):
    client = FakeClient()
    p = GroqProvider(client, "openai/gpt-oss-20b")
    before = counts()
    assert await p.check_health() is True
    assert (client.completions, client.model_lists) == (0, 1)
    assert counts() == before  # nothing counted as a system or an unmetered paid call


async def test_thirty_probes_in_five_minutes_make_one_free_request(clock):
    client = FakeClient()
    p = GroqProvider(client, "m")
    for _ in range(30):
        assert await p.check_health() is True
        clock[0] += 10  # 5 minutes in total
    assert client.completions == 0
    assert client.model_lists == 1  # the answer is reused for the whole 5 minutes
    clock[0] += 10
    assert await p.check_health() is True and client.model_lists == 2  # then asked again, once


async def test_a_recent_successful_real_call_is_the_health_answer(clock):
    client = FakeClient()
    p = GroqProvider(client, "m")
    await p.generate([{"role": "user", "content": "x"}])
    clock[0] += 60
    assert await p.check_health() is True
    assert client.model_lists == 0  # not even the free call: a real call just worked
    clock[0] += 400  # the proof is stale: one free probe
    assert await p.check_health() is True and client.model_lists == 1 and client.completions == 1


async def test_a_failed_real_call_does_not_count_as_proof_and_a_failed_probe_is_retried_soon(clock):
    client = FakeClient(fail_models=True)
    p = GroqProvider(client, "m")
    assert await p.check_health() is False
    assert await p.check_health() is False and client.model_lists == 1  # cached briefly
    clock[0] += 31
    client.fail_models = False
    assert await p.check_health() is True and client.model_lists == 2  # recovers within 30 s, not 5 minutes
    assert client.completions == 0
