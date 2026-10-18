"""User memory controls and user-facing memory management: pause is enforced on reads,
writes and extraction; deleted/corrected memories are never re-learned or used; every
operation is confined to the authenticated user; tokens are verified strictly."""

import base64
import hashlib
import hmac
import json
import time
import uuid
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.config import settings
from app.core.security import verify_user_access_token
from app.services.long_memory_service import LongMemoryService
from app.services.memory_controls import MemoryControls, statement_hash
from app.services.retrieval_service import RetrievalService
from app.services.user_memory_service import FactNotFoundError, UserMemoryService

ALICE = str(uuid.uuid4())
BOB = str(uuid.uuid4())
VEC = [0.1] * settings.VECTOR_DIMENSION


class FakeCassandra:
    """Just the parts of CassandraRepository these features use, with real semantics."""

    def __init__(self):
        self.settings: dict = {}
        self.tombs: dict = {}
        self.facts: dict = {}  # (user, category, fact_id) -> row
        self.fail_settings = False

    def get_memory_enabled(self, user_id):
        if self.fail_settings:
            raise RuntimeError("cassandra down")
        return self.settings.get(user_id)

    def set_memory_enabled(self, user_id, enabled):
        self.settings[user_id] = enabled

    def add_fact_tombstone(self, user_id, h):
        self.tombs.setdefault(user_id, set()).add(h)

    def get_fact_tombstones(self, user_id):
        return set(self.tombs.get(user_id, set()))

    def get_facts(self, user_id, category):
        return [r for (u, c, _), r in self.facts.items() if u == user_id and c == category]

    def get_fact(self, user_id, category, fact_id):
        return self.facts.get((user_id, category, fact_id))

    def upsert_fact(self, fact):
        self.facts[(fact["user_id"], fact["category"], fact["fact_id"])] = dict(fact)

    def delete_fact(self, user_id, category, fact_id):
        self.facts.pop((user_id, category, fact_id), None)


class FakeMilvus:
    def __init__(self):
        self.rows: dict = {}  # fact_id -> row
        self.calls: list = []

    def insert_facts(self, records):
        for r in records:
            self.rows[r["fact_id"]] = dict(r)
        return [r["fact_id"] for r in records]

    def search_facts(self, user_id, query_vector, limit=5, category=None, **_):
        self.calls.append(("search", user_id))
        hits = [dict(r, distance=0.5) for r in self.rows.values() if r["user_id"] == user_id
                and (category is None or r["category"] == category)]
        return hits[:limit]

    def list_user_facts(self, user_id, limit=500):
        return [dict(r) for r in self.rows.values() if r["user_id"] == user_id][:limit]

    def delete_fact(self, user_id, fact_id):
        self.calls.append(("delete", fact_id))
        self.rows.pop(str(fact_id), None)

    def delete_user_facts(self, user_id):
        for fid in [k for k, r in self.rows.items() if r["user_id"] == user_id]:
            self.rows.pop(fid)


def _seed(cass, milvus, user, statement, category="preferences"):
    fid = uuid.uuid4()
    now = datetime.now(timezone.utc)
    cass.upsert_fact({"user_id": user, "category": category, "fact_id": fid, "conversation_id": "c1",
                      "statement": statement, "importance": 0.8, "fact_version": 1, "embedding_version": "v1",
                      "created_at": now, "updated_at": now})
    milvus.insert_facts([{"fact_id": str(fid), "user_id": user, "conversation_id": "c1", "category": category,
                          "statement": statement, "importance": 0.8, "fact_version": 1, "embedding_ver": "v1",
                          "created_at": now.timestamp(), "vector": VEC}])
    return fid


@pytest.fixture
def stores():
    return FakeCassandra(), FakeMilvus()


def _incoming(statement):
    return [{"statement": statement, "category": "preferences", "importance": 0.9, "vector": VEC}]


# ---------------------------------------------------------------- hashing / controls


def test_statement_hash_ignores_case_spacing_and_trailing_punctuation():
    assert statement_hash("User likes  Python.") == statement_hash("  user likes python ")
    assert statement_hash("User likes Python") != statement_hash("User likes Rust")


def test_memory_defaults_on_and_fails_closed(stores):
    cass, _ = stores
    controls = MemoryControls(cass)
    assert controls.is_enabled(ALICE) is True
    controls.set_enabled(ALICE, False)
    assert controls.is_enabled(ALICE) is False
    assert controls.is_enabled(BOB) is True
    assert controls.is_enabled("unknown_user") is False
    cass.fail_settings = True
    assert controls.is_enabled(BOB) is False, "an unreadable setting must not silently mean ON"


# ---------------------------------------------------------------- write path


@pytest.mark.asyncio
async def test_paused_memory_stores_nothing(stores):
    cass, milvus = stores
    MemoryControls(cass).set_enabled(ALICE, False)
    stats = await LongMemoryService(cass, milvus).merge_user_facts(ALICE, "c1", _incoming("Likes tea"))
    assert stats["inserted"] == 0 and stats["skipped"] == 1
    assert cass.facts == {} and milvus.rows == {}


@pytest.mark.asyncio
async def test_deleted_memory_is_not_relearned_by_background_extraction(stores):
    cass, milvus = stores
    fid = _seed(cass, milvus, ALICE, "Likes tea")
    svc = UserMemoryService(cass, milvus, AsyncMock(), MemoryControls(cass))
    svc.delete_fact(ALICE, fid, "preferences")
    # a later summary produces the same statement (different case/punctuation)
    stats = await LongMemoryService(cass, milvus).merge_user_facts(ALICE, "c2", _incoming("likes tea."))
    assert stats["inserted"] == 0 and milvus.rows == {}


@pytest.mark.asyncio
async def test_unpaused_memory_still_learns(stores):
    cass, milvus = stores
    stats = await LongMemoryService(cass, milvus).merge_user_facts(ALICE, "c1", _incoming("Likes tea"))
    assert stats["inserted"] == 1 and len(milvus.rows) == 1


# ---------------------------------------------------------------- read path


@pytest.mark.asyncio
async def test_paused_memory_is_not_used_in_answers(stores):
    cass, milvus = stores
    _seed(cass, milvus, ALICE, "Likes tea")
    controls = MemoryControls(cass)
    retrieval = RetrievalService(MagicMock(), milvus, controls=controls)
    assert len(await retrieval.retrieve_relevant_facts(ALICE, VEC)) == 1
    controls.set_enabled(ALICE, False)
    milvus.calls.clear()
    assert await retrieval.retrieve_relevant_facts(ALICE, VEC) == []
    assert milvus.calls == [], "no vector search at all while paused"


@pytest.mark.asyncio
async def test_forgotten_statement_is_filtered_even_if_its_vector_survived(stores):
    cass, milvus = stores
    _seed(cass, milvus, ALICE, "Likes tea")
    controls = MemoryControls(cass)
    controls.forget(ALICE, "Likes tea")  # e.g. Milvus delete did not complete
    assert await RetrievalService(MagicMock(), milvus, controls=controls).retrieve_relevant_facts(ALICE, VEC) == []


# ---------------------------------------------------------------- extraction


@pytest.mark.asyncio
async def test_fact_worker_skips_the_llm_call_when_memory_is_off(stores):
    from app.workers.fact_worker import FactWorker

    cass, _ = stores
    worker = FactWorker.__new__(FactWorker)
    worker.controls = MemoryControls(cass)
    worker.llm_service = MagicMock()
    worker.llm_service.extract_facts = AsyncMock(return_value=SimpleNamespace(
        facts=[SimpleNamespace(category="preferences", importance=0.8, statement="Likes tea")]))
    assert await worker._extract_facts(ALICE, "c1", object()) == ["preferences:0.8:Likes tea"]
    worker.controls.set_enabled(ALICE, False)
    worker.llm_service.extract_facts.reset_mock()
    assert await worker._extract_facts(ALICE, "c1", object()) == []
    worker.llm_service.extract_facts.assert_not_called()


# ---------------------------------------------------------------- user-facing management


def test_list_is_scoped_to_the_user(stores):
    cass, milvus = stores
    _seed(cass, milvus, ALICE, "Alice likes tea")
    _seed(cass, milvus, BOB, "Bob likes coffee")
    svc = UserMemoryService(cass, milvus, AsyncMock(), MemoryControls(cass))
    assert [f["statement"] for f in svc.list_facts(ALICE)] == ["Alice likes tea"]


def test_delete_order_tombstone_then_vector_then_row(stores):
    cass, milvus = stores
    fid = _seed(cass, milvus, ALICE, "Likes tea")
    svc = UserMemoryService(cass, milvus, AsyncMock(), MemoryControls(cass))
    svc.delete_fact(ALICE, fid, "preferences")
    assert milvus.rows == {} and cass.facts == {}
    assert statement_hash("Likes tea") in cass.tombs[ALICE]


def test_cannot_delete_or_edit_another_users_memory(stores):
    cass, milvus = stores
    fid = _seed(cass, milvus, BOB, "Bob likes coffee")
    svc = UserMemoryService(cass, milvus, AsyncMock(), MemoryControls(cass))
    with pytest.raises(FactNotFoundError):
        svc.delete_fact(ALICE, fid, "preferences")
    assert len(milvus.rows) == 1 and len(cass.facts) == 1


@pytest.mark.asyncio
async def test_edit_reembeds_keeps_id_and_forgets_old_wording(stores):
    cass, milvus = stores
    fid = _seed(cass, milvus, ALICE, "Lives in Pune")
    emb = MagicMock()
    emb.generate_embedding = AsyncMock(return_value=[0.2] * settings.VECTOR_DIMENSION)
    svc = UserMemoryService(cass, milvus, emb, MemoryControls(cass))
    out = await svc.update_fact(ALICE, fid, "preferences", "Lives in Mumbai")
    assert out["id"] == str(fid) and out["statement"] == "Lives in Mumbai"
    assert milvus.rows[str(fid)]["statement"] == "Lives in Mumbai" and milvus.rows[str(fid)]["fact_version"] == 2
    assert cass.facts[(ALICE, "preferences", fid)]["statement"] == "Lives in Mumbai"
    assert statement_hash("Lives in Pune") in cass.tombs[ALICE]
    emb.generate_embedding.assert_awaited_once_with("Lives in Mumbai")


def test_clear_all_removes_only_this_users_memories(stores):
    cass, milvus = stores
    _seed(cass, milvus, ALICE, "a1")
    _seed(cass, milvus, ALICE, "a2", category="goals")
    _seed(cass, milvus, BOB, "b1")
    svc = UserMemoryService(cass, milvus, AsyncMock(), MemoryControls(cass))
    assert svc.clear_all(ALICE) == 2
    assert [r["user_id"] for r in milvus.rows.values()] == [BOB]
    assert all(u == BOB for (u, _, _) in cass.facts)


# ---------------------------------------------------------------- ownership in context building


@pytest.mark.asyncio
async def test_context_for_someone_elses_conversation_is_empty():
    from app.services.context_builder import ContextBuilder

    retrieval = MagicMock()
    retrieval.get_or_hydrate_snapshot = AsyncMock(return_value={"user_id": BOB})
    retrieval.get_or_hydrate_summary = AsyncMock(return_value="Bob's private summary")
    retrieval.get_or_hydrate_recent_messages = AsyncMock(return_value=[{"content": "bob secret"}])
    retrieval.retrieve_relevant_facts = AsyncMock(return_value=[{"statement": "bob fact"}])
    builder = ContextBuilder(retrieval, MagicMock())
    builder.get_parent_summaries = AsyncMock(return_value=([], True))
    ctx = await builder.build_context(ALICE, "bobs-conversation", query_vector=VEC)
    assert ctx["current_summary"] == "" and ctx["short_term_messages"] == [] and ctx["relevant_facts"] == []
    assert ctx["metadata"]["denied"] is True


# ---------------------------------------------------------------- token verification + API


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _token(sub=ALICE, exp_in=300, iss=None, alg="HS256", secret=None, drop=()):
    header = {"alg": alg, "typ": "JWT"}
    payload = {"sub": sub, "exp": int(time.time()) + exp_in, "iss": iss or settings.USER_JWT_ISSUER}
    for k in drop:
        payload.pop(k, None)
    signing = f"{_b64(json.dumps(header).encode())}.{_b64(json.dumps(payload).encode())}"
    sig = hmac.new((secret or settings.USER_JWT_SECRET_KEY).encode(), signing.encode(), hashlib.sha256).digest()
    return f"{signing}.{_b64(sig)}"


@pytest.fixture
def user_secret(monkeypatch):
    monkeypatch.setattr(settings, "USER_JWT_SECRET_KEY", "test-user-secret-key-for-unit-tests")


def test_token_verification_is_strict(user_secret):
    assert verify_user_access_token(_token()) == ALICE
    assert verify_user_access_token(_token(exp_in=-5)) is None
    assert verify_user_access_token(_token(drop=("exp",))) is None
    assert verify_user_access_token(_token(iss="https://evil.example")) is None
    assert verify_user_access_token(_token(alg="none")) is None
    assert verify_user_access_token(_token(secret="another-secret")) is None
    assert verify_user_access_token(_token(sub="not-a-uuid")) is None
    assert verify_user_access_token(_token()[:-3] + "abc") is None


def test_public_routes_fail_closed_without_a_configured_key(monkeypatch):
    monkeypatch.setattr(settings, "USER_JWT_SECRET_KEY", "")
    assert verify_user_access_token(_token(secret="x")) is None


@pytest.fixture
def api(stores, user_secret):
    from app.api.user_memory import router

    cass, milvus = stores
    app = FastAPI()
    app.include_router(router)
    app.state.container = SimpleNamespace(
        user_memory_service=UserMemoryService(cass, milvus, AsyncMock(), MemoryControls(cass)))
    return TestClient(app), cass, milvus


def test_api_requires_a_valid_user_token(api):
    client, _, _ = api
    assert client.get("/memory/facts").status_code == 401
    assert client.get("/memory/facts", headers={"Authorization": f"Bearer {_token(exp_in=-1)}"}).status_code == 401


def test_api_uses_only_the_token_subject(api):
    client, cass, milvus = api
    _seed(cass, milvus, ALICE, "Alice fact")
    bob_fid = _seed(cass, milvus, BOB, "Bob fact")
    ha = {"Authorization": f"Bearer {_token(ALICE)}"}
    listed = client.get("/memory/facts", headers=ha).json()
    assert [f["statement"] for f in listed["items"]] == ["Alice fact"] and listed["memory_enabled"] is True
    # Alice cannot delete Bob's memory, even knowing its id and category
    assert client.delete(f"/memory/facts/{bob_fid}", params={"category": "preferences"}, headers=ha).status_code == 404
    assert str(bob_fid) in milvus.rows


def test_api_pause_and_clear(api):
    client, cass, milvus = api
    _seed(cass, milvus, ALICE, "Alice fact")
    ha = {"Authorization": f"Bearer {_token(ALICE)}"}
    assert client.put("/memory/settings", json={"memory_enabled": False}, headers=ha).json() == {"memory_enabled": False}
    assert client.get("/memory/settings", headers=ha).json() == {"memory_enabled": False}
    assert client.delete("/memory/facts", headers=ha).json() == {"deleted": 1}
    assert client.get("/memory/facts", headers=ha).json()["items"] == []
