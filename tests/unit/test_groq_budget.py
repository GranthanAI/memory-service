"""B1/B2: memory jobs trust Groq's headers; busy Groq DEFERS a job (INFO, no attempt used) until it succeeds."""

import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from app.clients import groq_budget as gb
from app.clients.groq_budget import GroqBudget, GroqBudgetDeferred, MEMORY_JOBS_DEFERRED, MEMORY_JOBS_DROPPED, estimate_tokens
from app.core.config import settings
from app.managers.llm_manager import LLMManager
from app.services.memory_service import MemoryService
from app.workers.retry_support import defer_job, schedule_retry_or_park

M20 = "openai/gpt-oss-20b"


class FakeRedis:
    def __init__(self):
        self.kv, self.ttls, self.z = {}, {}, {}

    async def get(self, k):
        return self.kv.get(k)

    async def set(self, k, v, ex=None):
        self.kv[k] = str(v).encode()
        self.ttls[k] = ex

    async def ttl(self, k):
        return self.ttls.get(k, -2) if k in self.kv else -2

    async def exists(self, k):
        return 1 if k in self.kv else 0

    async def decrby(self, k, n):
        self.kv[k] = str(int(self.kv[k]) - n).encode()

    async def incrby(self, k, n):
        self.kv[k] = str(int(self.kv[k]) + n).encode()

    async def expire(self, k, s):
        return 1

    async def zadd(self, k, mapping):
        self.z.setdefault(k, {}).update(mapping)

    async def zrem(self, k, m):
        self.z.get(k, {}).pop(m, None)

    async def zremrangebyscore(self, k, lo, hi):
        self.z[k] = {m: s for m, s in self.z.get(k, {}).items() if not lo <= s <= hi}

    async def zrange(self, k, a, b, withscores=False):
        return sorted(((m.encode(), s) for m, s in self.z.get(k, {}).items()), key=lambda x: x[1])


def hdr(remaining, reset, retry=None):
    h = {"x-ratelimit-remaining-tokens": str(remaining), "x-ratelimit-reset-tokens": reset}
    if retry:
        h["retry-after"] = str(retry)
    return h


@pytest.fixture
def budget(monkeypatch):
    b = GroqBudget(FakeRedis(), tpm_limit=8000)
    monkeypatch.setattr(gb, "_BUDGET", b)
    return b


def manager():
    return SimpleNamespace(provider=SimpleNamespace(provider_name="groq", default_model=M20))


MSG = [{"role": "user", "content": "x" * 3000}]   # about 1,050 tokens


# --- B1 ------------------------------------------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_remaining_and_reset_come_from_groqs_headers(budget):
    await budget.record_headers(M20, hdr(6000, "20s"))
    assert (await budget.known_remaining(M20))[0] == 6000
    await budget.record_headers(M20, hdr(100, "7.5s"))
    wait = await budget.acquire(M20, 3000, "memory")
    assert 7 <= wait <= 8.5                                    # Groq's reset, not a minute


def test_the_estimate_counts_a_realistic_output_not_max_tokens():
    assert estimate_tokens(3367, 512) == 3367 + 512
    assert estimate_tokens(3367, 2048) == 3367 + 600
    assert settings.GROQ_OUTPUT_ESTIMATE_TOKENS == 600


@pytest.mark.asyncio
async def test_there_is_no_answer_reserve_a_memory_job_may_use_what_groq_says_is_left(budget):
    assert not hasattr(settings, "GROQ_ANSWER_RESERVE_TOKENS")
    await budget.record_headers(M20, hdr(1700, "30s"))
    tokens = await LLMManager._groq_gate(manager(), MSG, None, 2048)
    assert tokens == settings.MEMORY_JOB_MAX_TOKENS             # every memory job gets the same room
    left = (await budget.known_remaining(M20))[0]
    assert 1700 - left == estimate_tokens(int(3000 / 3.0) + 50, settings.MEMORY_JOB_MAX_TOKENS)   # input + min(cap, 600), nothing held back


@pytest.mark.asyncio
async def test_settle_to_real_usage_gives_the_difference_back(budget):
    await budget.record_headers(M20, hdr(5000, "30s"))
    assert await budget.acquire(M20, 2000, "memory") == 0.0
    await budget.settle(M20, 2000, 1200)
    assert (await budget.known_remaining(M20))[0] == 5000 - 1200


# --- B2 ------------------------------------------------------------------------------------------------------------------

class Repo:
    def __init__(self):
        self.jobs = []

    def insert_retry_job(self, job):
        self.jobs.append(job)


@pytest.mark.asyncio
async def test_a_short_budget_defers_the_job_with_groqs_reset_logged_as_deferred_not_failed(budget, caplog):
    await budget.record_headers(M20, hdr(200, "14s"))
    with pytest.raises(GroqBudgetDeferred) as e:
        await LLMManager._groq_gate(manager(), MSG, None, 512)
    assert 13 <= e.value.wait_s <= 15
    repo = Repo()
    before = MEMORY_JOBS_DEFERRED.labels(type="summary")._value.get()
    with caplog.at_level(logging.INFO):
        out = schedule_retry_or_park(repo, "summary", {"conversation_id": "c1", "attempt_count": 3}, b"{}", e.value)
    assert out == "deferred"
    (job,) = repo.jobs
    delay = (job["next_retry"] - datetime.now(timezone.utc)).total_seconds()
    assert 13 <= delay <= 18.5                                   # reset + 0-3 s jitter
    assert job["status"] == "PENDING" and job["retry_count"] == 2   # the scheduler re-dispatches as attempt 3: no attempt used
    assert json.loads(job["payload"])["deferred_since"]
    msgs = [r for r in caplog.records if "memory_job_deferred" in r.getMessage()]
    assert msgs and msgs[0].levelno == logging.INFO and "conversation_id=c1" in msgs[0].getMessage() and "attempt=3" in msgs[0].getMessage()
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert MEMORY_JOBS_DEFERRED.labels(type="summary")._value.get() == before + 1


@pytest.mark.asyncio
async def test_handle_failure_defers_without_failing_the_snapshot():
    repo = Repo()
    svc = MemoryService.__new__(MemoryService)
    svc.cassandra_repo = repo
    svc.transition_state = None                                   # would raise if the FAILED path were taken
    await svc.handle_failure("c1", None, "fact", {"conversation_id": "c1", "attempt_count": 5}, "busy", 5, defer_seconds=9.0)
    (job,) = repo.jobs
    assert job["status"] == "PENDING" and job["retry_count"] == 4 and job["job_type"] == "fact"


def test_a_deferred_job_keeps_retrying_until_it_succeeds_attempts_never_run_out():
    repo = Repo()
    payload = {"conversation_id": "c1"}
    for _ in range(12):                                           # far more than MAX_ATTEMPTS (5)
        attempt = repo.jobs[-1]["retry_count"] + 1 if repo.jobs else 1
        stored = json.loads(repo.jobs[-1]["payload"]) if repo.jobs else payload
        stored["attempt_count"] = attempt                         # what RetryScheduler does on re-dispatch
        assert defer_job(repo, "summary", stored, 5.0, "c1") == "deferred"
    assert repo.jobs[-1]["retry_count"] == 0                      # still attempt 1: never counted


def test_after_the_max_age_the_job_is_dropped_with_a_warning_and_a_metric(caplog):
    repo = Repo()
    old = (datetime.now(timezone.utc) - timedelta(minutes=settings.MEMORY_JOB_MAX_AGE_MIN + 1)).isoformat()
    before = MEMORY_JOBS_DROPPED.labels(type="summary")._value.get()
    with caplog.at_level(logging.INFO):
        out = defer_job(repo, "summary", {"conversation_id": "c9", "attempt_count": 1, "deferred_since": old}, 10.0, "c9")
    assert out == "dropped" and repo.jobs == []
    warn = [r for r in caplog.records if r.levelno == logging.WARNING and "memory_job_dropped" in r.getMessage()]
    assert warn and "conversation_id=c9" in warn[0].getMessage()
    assert MEMORY_JOBS_DROPPED.labels(type="summary")._value.get() == before + 1


def test_a_real_failure_keeps_its_current_handling():
    repo = Repo()
    out = schedule_retry_or_park(repo, "summary", {"conversation_id": "c1", "attempt_count": 0}, b"{}", RuntimeError("bad output"))
    assert out == "retry_scheduled" and repo.jobs[0]["retry_count"] == 1 and repo.jobs[0]["last_error"].startswith("RuntimeError")
    out = schedule_retry_or_park(repo, "summary", {"conversation_id": "c1", "attempt_count": 0}, b"{}", ValueError("parse error"))
    assert out == "dead_lettered"


@pytest.mark.asyncio
async def test_a_429_from_the_sdk_becomes_a_deferral_with_groqs_retry_after(budget):
    from app.clients.groq_client import GroqClient

    class Err(Exception):
        status_code = 429
        response = SimpleNamespace(headers={"retry-after": "11", "x-ratelimit-remaining-tokens": "0", "x-ratelimit-reset-tokens": "11s"})

    class Completions:
        async def create(self, **kw):
            raise Err("429 rate limit")

    client = GroqClient("k")
    client.client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    with pytest.raises(GroqBudgetDeferred) as e:
        await client.chat_completion([{"role": "user", "content": "hi"}], M20)
    assert 10.5 <= e.value.wait_s <= 11.5
    assert (await budget.known_remaining(M20))[0] == 0           # the 429's headers were stored for everyone


@pytest.mark.asyncio
async def test_the_manager_does_not_retry_or_trip_the_breaker_on_a_deferral(budget):
    calls = []

    class Provider:
        provider_name, default_model, meters_usage = "groq", M20, False

        async def generate(self, messages, **kw):
            calls.append(1)
            raise GroqBudgetDeferred(5.0, M20)

    mgr = LLMManager.__new__(LLMManager)
    mgr.provider, mgr._state, mgr._failures, mgr._last_failure_time, mgr._successes_in_half_open = Provider(), "CLOSED", 0, 0.0, 0
    mgr._lock, mgr._semaphore = asyncio.Lock(), asyncio.Semaphore(2)
    with pytest.raises(GroqBudgetDeferred):
        await mgr.generate_with_retry([{"role": "user", "content": "hi"}])
    assert len(calls) == 1 and mgr._failures == 0 and mgr._state == "CLOSED"
