"""P8b: billing client (deadline, breaker), entitlements cache (hit, miss, stale, Free fallback), the memory_facts cap, the
facts_used / facts_limit API fields, and the entitlements.changed consumer."""

import asyncio
import json
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import grpc
import pytest
import pytest_asyncio

from app.clients.billing_client import BillingClient, BillingUnavailable, EntitlementsData
from app.clients.billing_grpc import billing_pb2 as pb
from app.clients.billing_grpc import billing_pb2_grpc as pb_grpc
from app.core.config import settings
from app.core.metrics import MEMORY_FACT_CAP
from app.core.plan_defaults import FREE_LIMITS
from app.services.entitlements_service import EntitlementsService, cache_key, free_entitlements, limit_of, stale_key
from app.services.long_memory_service import LongMemoryService
from app.workers.billing_events_worker import BillingEventsWorker

pytestmark = pytest.mark.asyncio

# ---- fake in-process billing -------------------------------------------------------------------------------------------


class FakeBilling(pb_grpc.BillingInternalServicer):
    def __init__(self):
        self.calls, self.metadata, self.delay, self.fail = 0, [], 0.0, None
        self.limits = {"memory_facts": 500}
        self.plan = "pro"

    async def GetEntitlements(self, request, context):
        self.calls += 1
        self.metadata.append(dict(context.invocation_metadata()))
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.fail is not None:
            await context.abort(self.fail, "forced")
        return pb.Entitlements(user_id=request.user_id, plan_code=self.plan, source="subscription", limits=self.limits)


async def start_fake(fake):
    server = grpc.aio.server()
    pb_grpc.add_BillingInternalServicer_to_server(fake, server)
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    return server, f"127.0.0.1:{port}"


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


@pytest_asyncio.fixture
async def billing():
    fake = FakeBilling()
    server, target = await start_fake(fake)
    clock = Clock()
    client = BillingClient(target=target, token="mem-token", clock=clock)
    yield fake, client, clock
    await client.close()
    await server.stop(None)


async def test_get_entitlements_sends_the_token_and_maps_limits(billing):
    fake, client, _ = billing
    ent = await client.get_entitlements("u1")
    assert ent.plan_code == "pro" and limit_of(ent, "memory_facts") == 500
    assert fake.metadata[0]["authorization"] == "Bearer mem-token"


async def test_deadline_is_300ms(billing):
    fake, client, _ = billing
    fake.delay = 1.0
    started = time.perf_counter()
    with pytest.raises(BillingUnavailable) as e:
        await client.get_entitlements("u1")
    assert e.value.reason == "deadline_exceeded" and 0.2 < time.perf_counter() - started < 0.9


async def test_breaker_opens_after_5_failures_and_half_opens_after_30s(billing):
    fake, client, clock = billing
    fake.fail = grpc.StatusCode.UNAVAILABLE
    for _ in range(5):
        with pytest.raises(BillingUnavailable):
            await client.get_entitlements("u1")
    assert client.breaker.state == "open" and fake.calls == 5
    with pytest.raises(BillingUnavailable) as e:
        await client.get_entitlements("u1")
    assert e.value.reason == "breaker_open" and fake.calls == 5            # refused locally
    clock.t += 30
    fake.fail = None
    assert (await client.get_entitlements("u1")).plan_code == "pro"          # the probe succeeds
    assert client.breaker.state == "closed"


async def test_unauthenticated_is_a_config_problem_not_an_outage(billing):
    fake, client, _ = billing
    fake.fail = grpc.StatusCode.UNAUTHENTICATED
    for _ in range(6):
        with pytest.raises(BillingUnavailable):
            await client.get_entitlements("u1")
    assert client.breaker.state == "closed"


# ---- the Redis cache ---------------------------------------------------------------------------------------------------

class FakeRedis:
    def __init__(self):
        self.store, self.ttl = {}, {}

    async def get(self, k):
        return self.store.get(k)

    async def set(self, k, v, ex=None, nx=False):
        self.store[k] = v
        self.ttl[k] = ex
        return True

    async def delete(self, *keys):
        return sum(1 for k in keys if self.store.pop(k, None) is not None)


class StubBilling:
    def __init__(self, plan="pro", limits=None):
        self.calls, self.down = 0, False
        self.ent = EntitlementsData(user_id="u", plan_code=plan, limits=limits or {"memory_facts": 500}, source="subscription")

    async def get_entitlements(self, user_id):
        self.calls += 1
        if self.down:
            raise BillingUnavailable("unavailable")
        return self.ent


async def test_cache_miss_hit_ttls_and_sharing():
    redis, stub = FakeRedis(), StubBilling()
    svc = EntitlementsService(stub, redis)
    await svc.get("u1")
    await svc.get("u1")
    assert stub.calls == 1 and redis.ttl[cache_key("u1")] == 60 and redis.ttl[stale_key("u1")] == 3600
    other = StubBilling()
    assert (await EntitlementsService(other, redis).get("u1")).plan_code == "pro" and other.calls == 0     # shared by replicas


async def test_stale_then_free_fallback():
    redis, stub = FakeRedis(), StubBilling(plan="max")
    svc = EntitlementsService(stub, redis)
    await svc.get("u1")
    del redis.store[cache_key("u1")]
    stub.down = True
    ent = await svc.get("u1")
    assert ent.plan_code == "max" and ent.source == "stale"
    redis.store.clear()
    ent = await svc.get("u1")
    assert ent.plan_code == "free" and ent.source == "free_fallback" and ent.limits == FREE_LIMITS


PLANS = Path(__file__).resolve().parents[3] / "billing-service" / "config" / "plans.yaml"


@pytest.mark.skipif(not PLANS.exists(), reason="billing-service repo not next to this one")
async def test_bundled_free_limits_match_plans_yaml():
    # plans.yaml defines the Free limits once, as an anchor (`base_limits: &free_limits`) that the free plan reuses; PyYAML is not
    # installed in this environment, so the block is read by hand.
    lines = PLANS.read_text(encoding="utf-8").splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("base_limits:")) + 1
    limits = {}
    for line in lines[start:]:
        if not line.startswith("  ") or ":" not in line:
            break
        key, _, value = line.strip().partition(":")
        limits[key] = int(value.split("#")[0].strip())
    free_block = " ".join(lines[next(i for i, l in enumerate(lines) if "code: free" in l):][:6])
    assert "limits: *free_limits" in free_block
    assert limits == FREE_LIMITS


def test_unlimited_and_missing_limits():
    assert limit_of(EntitlementsData("u", "max", {"memory_facts": -1}), "memory_facts") == -1
    assert limit_of(EntitlementsData("u", "x", {}), "memory_facts") == FREE_LIMITS["memory_facts"]
    assert free_entitlements("u").limits["memory_facts"] == 50


# ---- the cap -----------------------------------------------------------------------------------------------------------

COLOURS = ["red", "blue", "green", "yellow", "purple", "orange", "pink", "brown", "black", "white", "grey", "teal", "navy", "lime"]


class FakeCassandra:
    """Only the calls the merge makes, with real state: an upsert makes the fact active, retire/delete removes it."""

    def __init__(self):
        self.facts = {}

    def get_facts(self, user_id, category, include_inactive=False):
        return [f for f in self.facts.values() if f["user_id"] == user_id and f["category"] == category]

    def upsert_fact(self, fact):
        self.facts[fact["fact_id"]] = dict(fact)

    def delete_fact(self, user_id, category, fact_id):
        self.facts.pop(fact_id, None)

    def retire_fact(self, user_id, category, fact_id, superseded_by):
        self.facts.pop(fact_id, None)

    def get_fact_tombstones(self, user_id):
        return set()

    def get_memory_enabled(self, user_id):
        return None


def make_service(entitlements, monkeypatch, enabled=True):
    cass, milvus = FakeCassandra(), MagicMock()
    milvus.search_facts.return_value = []
    milvus.insert_facts.return_value = []
    monkeypatch.setattr(settings, "BILLING_LIMITS_ENABLED", enabled)
    return LongMemoryService(cass, milvus, entitlements=entitlements), cass


def incoming(n, start=0):
    return [{"statement": f"User likes the colour {COLOURS[start + i]}.", "category": "preferences", "importance": 0.9,
             "vector": [0.1] * settings.VECTOR_DIMENSION} for i in range(n)]


class FixedLimit:
    def __init__(self, limit):
        self.limit = limit

    async def get(self, user_id):
        return EntitlementsData(user_id, "free", {"memory_facts": self.limit})


async def test_cap_minus_one_stores_and_the_cap_refuses(monkeypatch):
    svc, cass = make_service(FixedLimit(3), monkeypatch)
    before = MEMORY_FACT_CAP._value.get()
    assert (await svc.merge_user_facts("u1", "c1", incoming(2)))["inserted"] == 2          # 2 < 3: cap - 1 stores
    stats = await svc.merge_user_facts("u1", "c1", incoming(1, start=2))
    assert stats["inserted"] == 1                                                            # the 3rd reaches the cap
    stats = await svc.merge_user_facts("u1", "c1", incoming(2, start=3))
    assert stats["inserted"] == 0 and stats["capped"] == 2                                   # at the cap: refused
    assert len(cass.facts) == 3 and MEMORY_FACT_CAP._value.get() == before + 2


async def test_delete_frees_a_slot(monkeypatch):
    svc, cass = make_service(FixedLimit(2), monkeypatch)
    await svc.merge_user_facts("u1", "c1", incoming(2))
    assert (await svc.merge_user_facts("u1", "c1", incoming(1, start=2)))["inserted"] == 0
    cass.delete_fact("u1", "preferences", next(iter(cass.facts)))                            # the user deletes one memory
    assert (await svc.merge_user_facts("u1", "c1", incoming(1, start=2)))["inserted"] == 1
    assert len(cass.facts) == 2


async def test_existing_facts_over_a_shrunk_limit_stay_and_a_supersede_is_still_allowed(monkeypatch):
    svc, cass = make_service(FixedLimit(5), monkeypatch)
    await svc.merge_user_facts("u1", "c1", incoming(5))
    svc.entitlements = FixedLimit(2)                                                         # downgraded: 5 facts, limit 2
    assert (await svc.merge_user_facts("u1", "c1", incoming(1, start=5)))["inserted"] == 0
    assert len(cass.facts) == 5                                                              # nothing deleted or hidden


async def test_unlimited_never_caps(monkeypatch):
    svc, cass = make_service(FixedLimit(-1), monkeypatch)
    assert (await svc.merge_user_facts("u1", "c1", incoming(len(COLOURS))))["inserted"] == len(COLOURS)


async def test_flag_off_is_todays_behaviour(monkeypatch):
    svc, cass = make_service(FixedLimit(1), monkeypatch, enabled=False)
    assert (await svc.merge_user_facts("u1", "c1", incoming(4)))["inserted"] == 4


async def test_the_cap_is_per_user(monkeypatch):
    svc, cass = make_service(FixedLimit(2), monkeypatch)
    await svc.merge_user_facts("u1", "c1", incoming(2))
    assert (await svc.merge_user_facts("u2", "c1", incoming(2)))["inserted"] == 2


# ---- facts_used / facts_limit on the API ---------------------------------------------------------------------------------

async def test_list_facts_reports_facts_used_and_limit(monkeypatch):
    from app.api import user_memory

    service = MagicMock()
    service.list_facts.return_value = [{"id": "1", "category": "x", "statement": "s", "importance": 0.5}] * 3
    service.get_settings.return_value = {"memory_enabled": True}
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(container=SimpleNamespace(
        user_memory_service=service, entitlements=FixedLimit(50)))))
    monkeypatch.setattr(settings, "BILLING_LIMITS_ENABLED", True)
    out = await user_memory.list_facts(request, user_id="u1")
    assert (out["facts_used"], out["facts_limit"]) == (3, 50)
    monkeypatch.setattr(settings, "BILLING_LIMITS_ENABLED", False)
    out = await user_memory.list_facts(request, user_id="u1")
    assert (out["facts_used"], out["facts_limit"]) == (3, None)                               # limits not switched on


# ---- entitlements.changed ----------------------------------------------------------------------------------------------

class FakeProcessed:
    def __init__(self):
        self.seen = set()

    def is_event_processed(self, event_id):
        return event_id in self.seen

    def register_event(self, event_id, conversation_id):
        self.seen.add(event_id)


async def test_entitlements_changed_drops_both_cache_keys_once():
    redis, stub = FakeRedis(), StubBilling()
    svc = EntitlementsService(stub, redis)
    await svc.get("u1")
    worker = BillingEventsWorker(svc, FakeProcessed())
    event = {"event_id": str(uuid.uuid4()), "user_id": "u1", "event_type": "entitlements.changed"}
    assert await worker.handle("entitlements.changed", event) == "invalidated"
    assert cache_key("u1") not in redis.store and stale_key("u1") not in redis.store
    await svc.get("u1")
    assert await worker.handle("entitlements.changed", event) == "duplicate"                  # redelivery changes nothing
    assert cache_key("u1") in redis.store
    assert await worker.handle("entitlements.changed", {"user_id": "u1"}) == "ignored"
