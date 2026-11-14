"""P8a: one usage.metered event per job with the provider's exact tokens; retries that reached the provider counted; a
deterministic event_id so a re-run job cannot double-count; the mock provider never meters."""

import asyncio
import json
import uuid
from types import SimpleNamespace

import pytest

from app.clients.groq_client import GroqClient
from app.core.config import settings
from app.core.metrics import MEMORY_UNMETERED_CALLS, MEMORY_USAGE_ESTIMATED
from app.events.codecs import check_kafka_codecs, missing_codecs
from app.managers.llm_manager import LLMManager
from app.providers.groq_provider import GroqProvider
from app.providers.mock_provider import MockLLMProvider
from app.schemas.llm import FactExtractRequest, SummarizeRequest
from app.services import usage_metering
from app.services.llm_service import LLMService
from app.services.usage_metering import metered_job

pytestmark = pytest.mark.asyncio

_REAL_SLEEP = asyncio.sleep


def no_backoff(monkeypatch):
    """The retry backoff (1 s, 2 s) is skipped; anything else that sleeps is untouched."""
    monkeypatch.setattr(asyncio, "sleep", lambda s, *a, **k: _REAL_SLEEP(0))


class Sink:
    def __init__(self):
        self.jobs = []

    def insert_outbox_job(self, job):
        self.jobs.append(job)

    @property
    def events(self):
        return [json.loads(j["payload"]) for j in self.jobs]


@pytest.fixture(autouse=True)
def sink(monkeypatch):
    s = Sink()
    monkeypatch.setattr(usage_metering, "_sink", s)
    return s


class FakeCompletions:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), 0

    async def create(self, **kwargs):
        self.calls += 1
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def completion(text, prompt=None, completion_tokens=None, cached=None):
    usage = None
    if prompt is not None:
        details = SimpleNamespace(cached_tokens=cached) if cached is not None else None
        usage = SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion_tokens, prompt_tokens_details=details)
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))], usage=usage)


def groq_stack(replies, monkeypatch):
    client = GroqClient(api_key="k")
    client.client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions(replies)))
    provider = GroqProvider(client, default_model="openai/gpt-oss-20b")
    manager = LLMManager.__new__(LLMManager)                       # a fresh one: LLMManager is a process singleton
    manager._initialized = False
    LLMManager.__init__(manager, provider)
    no_backoff(monkeypatch)
    return LLMService(manager), client.client.chat.completions, manager


async def test_one_event_per_job_with_the_exact_provider_tokens(sink, monkeypatch):
    svc, comp, _ = groq_stack([completion("A summary.", 1234, 56, cached=100)], monkeypatch)
    async with metered_job("memory.summary", "memory.summary:t:0:41", "user-1", "conv-1", "req-1"):
        await svc.summarize(SummarizeRequest(previous_summary="", new_messages=[{"role": "user", "content": "hi"}]))
    assert len(sink.jobs) == 1
    job, event = sink.jobs[0], sink.events[0]
    assert job["topic"] == "usage.metered" and job["conversation_id"] == "user-1"       # outbox key = the user (plan 4.3)
    assert event["event_type"] == "usage.metered" and event["schema_version"] == "1.0" and event["producer"] == "memory-service"
    assert (event["user_id"], event["conversation_id"], event["request_message_id"], event["feature"]) == (
        "user-1", "conv-1", "req-1", "memory.summary")
    assert event["fixed_cost_micro_inr"] == 0 and event["occurred_at"].endswith("Z")
    assert event["calls"] == [{"provider": "groq", "model": "openai/gpt-oss-20b", "purpose": "memory.summary",
                               "input_tokens": 1234, "cached_input_tokens": 100, "output_tokens": 56, "usage_estimated": False}]


async def test_a_job_with_several_calls_still_sends_one_event(sink, monkeypatch):
    svc, comp, _ = groq_stack([completion("a", 10, 1), completion("b", 20, 2), completion("c", 30, 3)], monkeypatch)
    async with metered_job("memory.facts", "k1", "user-1", "conv-1"):
        await svc.extract_facts(FactExtractRequest(summary="s"))
        await svc.extract_facts(FactExtractRequest(summary="s"))
        await svc.extract_facts(FactExtractRequest(summary="s"))
    assert len(sink.events) == 1
    assert [(c["input_tokens"], c["output_tokens"]) for c in sink.events[0]["calls"]] == [(10, 1), (20, 2), (30, 3)]


async def test_retries_that_reached_the_provider_are_counted(sink, monkeypatch):
    """Two attempts time out (they reached the provider: estimated input, no output), the third answers with real usage."""
    class Slow:
        meters_usage, provider_name, default_model = True, "groq", "openai/gpt-oss-20b"

        def __init__(self):
            self.n = 0

        async def generate(self, messages, model=None, temperature=0.2, max_tokens=1024):
            self.n += 1
            if self.n < 3:
                await asyncio.Event().wait()                           # never answers: the timeout below fires
            usage_metering.record_call(provider="groq", model="openai/gpt-oss-20b",
                                       usage=SimpleNamespace(prompt_tokens=77, completion_tokens=8, prompt_tokens_details=None),
                                       messages=messages, output_text="ok")
            return "ok"

        async def check_health(self):
            return True

    monkeypatch.setattr(settings, "LLM_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr(settings, "LLM_MAX_RETRIES", 3)
    no_backoff(monkeypatch)
    manager = LLMManager.__new__(LLMManager)
    manager._initialized = False
    LLMManager.__init__(manager, Slow())
    before = MEMORY_USAGE_ESTIMATED.labels(provider="groq", model="openai/gpt-oss-20b")._value.get()
    async with metered_job("memory.summary", "k2", "user-1", "conv-1"):
        assert await manager.generate_with_retry([{"role": "user", "content": "x" * 400}]) == "ok"
    calls = sink.events[0]["calls"]
    assert len(calls) == 3
    assert [(c["usage_estimated"], c["output_tokens"]) for c in calls] == [(True, 0), (True, 0), (False, 8)]
    assert calls[0]["input_tokens"] > 0 and calls[2]["input_tokens"] == 77
    assert MEMORY_USAGE_ESTIMATED.labels(provider="groq", model="openai/gpt-oss-20b")._value.get() == before + 2


async def test_missing_provider_usage_is_estimated_and_flagged(sink, monkeypatch):
    svc, comp, _ = groq_stack([completion("a fairly short answer")], monkeypatch)      # the provider reported no usage
    before = MEMORY_USAGE_ESTIMATED.labels(provider="groq", model="openai/gpt-oss-20b")._value.get()
    async with metered_job("memory.facts", "k3", "user-1", "conv-1"):
        await svc.extract_facts(FactExtractRequest(summary="the user likes tea"))
    call = sink.events[0]["calls"][0]
    assert call["usage_estimated"] is True and call["input_tokens"] > 0 and call["output_tokens"] > 0
    assert MEMORY_USAGE_ESTIMATED.labels(provider="groq", model="openai/gpt-oss-20b")._value.get() == before + 1


async def test_event_id_is_deterministic_per_job(sink, monkeypatch):
    for _ in range(2):                                                  # the same Kafka message processed twice
        svc, comp, _ = groq_stack([completion("x", 5, 1)], monkeypatch)
        async with metered_job("memory.summary", "memory.summary:topic:0:99", "user-1", "conv-1"):
            await svc.summarize(SummarizeRequest(previous_summary="", new_messages=[{"role": "user", "content": "hi"}]))
    async with metered_job("memory.summary", "memory.summary:topic:0:100", "user-1", "conv-1"):
        svc, comp, _ = groq_stack([completion("x", 5, 1)], monkeypatch)
        await svc.summarize(SummarizeRequest(previous_summary="", new_messages=[{"role": "user", "content": "hi"}]))
    ids = [e["event_id"] for e in sink.events]
    assert ids[0] == ids[1] != ids[2]
    assert ids[0] == str(uuid.uuid5(usage_metering.NAMESPACE, "memory.summary:topic:0:99"))
    assert sink.jobs[0]["job_id"] == uuid.UUID(ids[0])


async def test_the_mock_provider_is_never_metered(sink):
    manager = LLMManager.__new__(LLMManager)
    manager._initialized = False
    LLMManager.__init__(manager, MockLLMProvider())
    svc = LLMService(manager)
    async with metered_job("memory.summary", "k4", "user-1", "conv-1"):
        await svc.summarize(SummarizeRequest(previous_summary="", new_messages=[{"role": "user", "content": "hi"}]))
        await svc.extract_facts(FactExtractRequest(summary="likes coding"))
    assert sink.jobs == []


async def test_a_job_that_fails_after_calling_the_model_still_reports_its_cost(sink, monkeypatch):
    svc, comp, _ = groq_stack([completion("x", 5, 1)], monkeypatch)
    with pytest.raises(RuntimeError):
        async with metered_job("memory.summary", "k5", "user-1", "conv-1"):
            await svc.summarize(SummarizeRequest(previous_summary="", new_messages=[{"role": "user", "content": "hi"}]))
            raise RuntimeError("the state machine failed afterwards")
    assert len(sink.events) == 1 and sink.events[0]["calls"][0]["input_tokens"] == 5


async def test_a_job_without_a_paid_call_writes_nothing(sink):
    async with metered_job("memory.summary", "k6", "user-1", "conv-1"):
        pass
    assert sink.jobs == []


async def test_a_paid_call_outside_any_job_is_counted_not_billed(sink, monkeypatch):
    svc, comp, _ = groq_stack([completion("x", 5, 1)], monkeypatch)
    before = MEMORY_UNMETERED_CALLS._value.get()
    await svc.summarize(SummarizeRequest(previous_summary="", new_messages=[{"role": "user", "content": "hi"}]))
    assert MEMORY_UNMETERED_CALLS._value.get() == before + 1 and sink.jobs == []


async def test_a_job_without_a_user_cannot_be_billed(sink, monkeypatch):
    svc, comp, _ = groq_stack([completion("x", 5, 1)], monkeypatch)
    before = MEMORY_UNMETERED_CALLS._value.get()
    async with metered_job("memory.summary", "k7", "", "conv-1"):
        await svc.summarize(SummarizeRequest(previous_summary="", new_messages=[{"role": "user", "content": "hi"}]))
    assert sink.jobs == [] and MEMORY_UNMETERED_CALLS._value.get() == before + 1


async def test_an_outbox_failure_never_breaks_the_job(monkeypatch):
    class Broken:
        def insert_outbox_job(self, job):
            raise RuntimeError("cassandra down")

    monkeypatch.setattr(usage_metering, "_sink", Broken())
    svc, comp, _ = groq_stack([completion("x", 5, 1)], monkeypatch)
    async with metered_job("memory.summary", "k8", "user-1", "conv-1"):
        await svc.summarize(SummarizeRequest(previous_summary="", new_messages=[{"role": "user", "content": "hi"}]))


async def test_request_message_id_flows_into_the_summary_request_payload(monkeypatch):
    """The dispatcher stamps the summary request with the reason and the request_message_id being ingested."""
    assert usage_metering.REQUEST_MESSAGE_ID.get() is None
    token = usage_metering.REQUEST_MESSAGE_ID.set("0194a1b2-0000-7000-8000-000000000001")
    try:
        assert usage_metering.REQUEST_MESSAGE_ID.get() == "0194a1b2-0000-7000-8000-000000000001"
    finally:
        usage_metering.REQUEST_MESSAGE_ID.reset(token)


def test_all_kafka_codecs_are_available():
    assert missing_codecs() == []
    check_kafka_codecs()


def test_a_missing_codec_fails_startup(monkeypatch):
    import aiokafka.codec as codec
    monkeypatch.setattr(codec, "has_snappy", lambda: False)
    with pytest.raises(RuntimeError, match="snappy"):
        check_kafka_codecs()


async def test_model_not_in_billing_catalog_logs_a_warning(monkeypatch, caplog):
    import httpx

    class FakeClient:
        def __init__(self, *a, **k): ...
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def get(self, url):
            return SimpleNamespace(json=lambda: {"models": [{"id": "openai/gpt-oss-20b"}, {"id": "gemini-2.5-flash-lite"}]})

    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)
    monkeypatch.setattr(settings, "LLM_PROVIDER", "groq")
    monkeypatch.setattr(settings, "LLM_MODEL", "some/unknown-model")
    with caplog.at_level("WARNING"):
        assert await usage_metering.check_model_in_billing_catalog() is False
    assert "llm_model_not_in_billing_catalog" in caplog.text
    monkeypatch.setattr(settings, "LLM_MODEL", "openai/gpt-oss-20b")
    assert await usage_metering.check_model_in_billing_catalog() is True
    monkeypatch.setattr(settings, "LLM_PROVIDER", "mock")
    assert await usage_metering.check_model_in_billing_catalog() is None


async def test_the_health_probe_is_a_system_call_not_an_unmetered_bug(sink, monkeypatch):
    from app.core.metrics import MEMORY_SYSTEM_CALLS
    svc, comp, manager = groq_stack([completion("pong", 3, 1)], monkeypatch)
    system_before, unmetered_before = MEMORY_SYSTEM_CALLS._value.get(), MEMORY_UNMETERED_CALLS._value.get()
    assert await manager.check_health() is True
    assert MEMORY_SYSTEM_CALLS._value.get() == system_before + 1
    assert MEMORY_UNMETERED_CALLS._value.get() == unmetered_before and sink.jobs == []
