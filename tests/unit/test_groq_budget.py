"""C1: memory jobs share the per-model Groq budget with llm-service: background priority, reserve kept, deferred not failed."""

import asyncio
import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from app.clients import groq_budget as gb
from app.clients.groq_budget import GroqBudget, GroqBudgetDeferred, MEMORY_JOBS_DEFERRED
from app.core.config import settings
from app.managers.llm_manager import LLMManager
from app.services.memory_service import MemoryService
from app.workers.retry_support import schedule_retry_or_park

M20 = "openai/gpt-oss-20b"


class FakeRedis:
    def __init__(self):
        self.kv, self.ttls, self.z = {}, {}, {}

    async def get(self, k):
        return self.kv.get(k)

    async def set(self, k, v, ex=None):
        self.kv[k] = str(v).encode(); self.ttls[k] = ex

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

    async def zremrangebyscore(self, k, lo, hi):
        self.z[k] = {m: s for m, s in self.z.get(k, {}).items() if not lo <= s <= hi}

    async def zrange(self, k, a, b, withscores=False):
        return sorted(((m.encode(), s) for m, s in self.z.get(k, {}).items()), key=lambda x: x[1])


def hdr(remaining, reset):
    return {"x-ratelimit-remaining-tokens": str(remaining), "x-ratelimit-reset-tokens": reset}


@pytest.fixture
def budget(monkeypatch):
    b = GroqBudget(FakeRedis(), tpm_limit=8000, answer_reserve=4000)
    monkeypatch.setattr(gb, "_BUDGET", b)
    return b


def manager():
    return SimpleNamespace(provider=SimpleNamespace(provider_name="groq", default_model=M20))


@pytest.mark.asyncio
async def test_budget_reads_the_headers_llm_service_stored(budget):
    await budget.record_headers(M20, hdr(6000, "20s"))
    assert (await budget.known_remaining(M20))[0] == 6000


@pytest.mark.asyncio
async def test_a_memory_job_caps_its_output_at_512_and_claims_tokens(budget):
    await budget.record_headers(M20, hdr(7000, "30s"))
    tokens = await LLMManager._groq_gate(manager(), [{"role": "user", "content": "x" * 300}], None, 2048)
    assert tokens == settings.GROQ_MEMORY_MAX_TOKENS == 512
    assert (await budget.known_remaining(M20))[0] < 7000                     # claimed for everyone else to see


@pytest.mark.asyncio
async def test_the_answer_reserve_is_respected_and_the_job_is_deferred_not_failed(budget):
    await budget.record_headers(M20, hdr(4300, "25s"))
    with pytest.raises(GroqBudgetDeferred) as e:
        await LLMManager._groq_gate(manager(), [{"role": "user", "content": "x" * 3000}], None, 512)
    assert 20 <= e.value.wait_s <= 27                                       # until the reset
    assert (await budget.known_remaining(M20))[0] == 4300                   # nothing was taken
    assert await budget.acquire(M20, 4000, "answer") == 0.0                 # an answer still gets the reserve


class Repo:
    def __init__(self):
        self.jobs = []

    def insert_retry_job(self, job):
        self.jobs.append(job)


@pytest.mark.asyncio
async def test_handle_failure_defers_without_using_an_attempt():
    repo = Repo()
    svc = MemoryService.__new__(MemoryService)
    svc.cassandra_repo = repo
    before = MEMORY_JOBS_DEFERRED.labels(job_type="summary")._value.get()
    await svc.handle_failure("c1", None, "summary", {"conversation_id": "c1", "attempt_count": 2}, "budget", 2, defer_seconds=18.0)
    (job,) = repo.jobs
    assert job["status"] == "PENDING" and job["retry_count"] == 2 and job["last_error"].startswith("deferred")
    assert 17 <= (job["next_retry"] - datetime.now(timezone.utc)).total_seconds() <= 19
    assert MEMORY_JOBS_DEFERRED.labels(job_type="summary")._value.get() == before + 1


def test_schedule_retry_or_park_defers_a_budget_exception_without_counting_an_attempt():
    repo = Repo()
    payload = {"conversation_id": "c1", "attempt_count": 3}
    outcome = schedule_retry_or_park(repo, "branch_summary", payload, b"{}", GroqBudgetDeferred(12.0, M20))
    assert outcome == "deferred"
    assert repo.jobs[0]["status"] == "PENDING" and json.loads(repo.jobs[0]["payload"])["attempt_count"] == 3
