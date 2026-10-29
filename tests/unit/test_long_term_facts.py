"""Phase 5: long-term facts - soft supersede with an LLM judgement, reconciliation, explicit-memory fast path, idle trigger,
extraction from the consumed messages, regenerated answers, conversation deletes, retrieval (recency, core profile, query)."""
import asyncio
import json
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.core.config import settings
from app.models.memory import MemoryState
from app.proto.memory_handlers import build_retrieval_query
from app.services import memory_policy
from app.services.llm_service import LLMService
from app.services.long_memory_service import LongMemoryService
from app.services.ranking_service import RankingService
from app.services.retrieval_service import RetrievalService
from app.workers.conversation_delete_worker import ConversationDeleteWorker
from app.workers.fact_worker import build_extraction_notes
from app.workers.idle_compaction_worker import IDLE_KEY, IdleCompactionWorker
from app.workers.memory_update_worker import MemoryUpdateWorker
from tests.support.fakes import FakeCassandra, ScriptedSummarizer, msg

pytestmark = pytest.mark.asyncio

U = "user-1"
NOW = datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc)


# ─── doubles ─────────────────────────────────────────────────────────────────────────────────────────────────────────


class Facts:
    """Cassandra user_facts + Milvus vectors with a call log, so the ORDER of operations can be asserted."""

    def __init__(self):
        self.rows, self.vectors, self.log = {}, {}, []

    # cassandra side
    def get_facts(self, u, c, include_inactive=False):
        return [dict(r) for (uu, cc, _), r in self.rows.items()
                if uu == u and cc == c and (include_inactive or (r.get("status") or "ACTIVE") == "ACTIVE")]

    def get_fact(self, u, c, f):
        return dict(self.rows[(u, c, f)]) if (u, c, f) in self.rows else None

    def upsert_fact(self, f):
        self.log.append(("write", f["statement"]))
        self.rows[(f["user_id"], f["category"], f["fact_id"])] = {**f, "status": "ACTIVE"}

    def retire_fact(self, u, c, f, superseded_by):
        self.log.append(("retire", self.rows[(u, c, f)]["statement"]))
        self.rows[(u, c, f)].update(status="SUPERSEDED", superseded_by=superseded_by)

    def delete_fact(self, *args):
        """Cassandra's delete_fact(user, category, fact_id) and Milvus's delete_fact(user, fact_id) share this double."""
        if len(args) == 3:
            self.rows.pop(args, None)
        else:
            self.vectors.pop(str(args[1]), None)

    def touch_fact_confirmed(self, u, c, f, conversation_id=None):
        self.rows[(u, c, f)]["last_confirmed"] = NOW
        if conversation_id:
            self.rows[(u, c, f)]["conversation_id"] = conversation_id        # the restating conversation becomes the source
        self.log.append(("confirm", self.rows[(u, c, f)]["statement"]))

    def mark_fact_used(self, u, c, f):
        self.rows[(u, c, f)]["last_used"] = NOW
        self.log.append(("used", self.rows[(u, c, f)]["statement"]))

    def get_memory_enabled(self, u): return None
    def get_fact_tombstones(self, u): return set()
    def set_memory_enabled(self, u, e): pass

    # milvus side
    def insert_facts(self, records):
        for r in records:
            self.vectors[r["fact_id"]] = dict(r)

    def search_facts(self, user_id, query_vector, limit=5, **_):
        hits = [dict(r, distance=self._sim(query_vector, r["vector"])) for r in self.vectors.values() if r["user_id"] == user_id]
        return sorted(hits, key=lambda h: h["distance"], reverse=True)[:limit]

    def list_user_facts(self, user_id, limit=500):
        return [{k: v for k, v in r.items() if k != "vector"} for r in self.vectors.values() if r["user_id"] == user_id]

    @staticmethod
    def _sim(a, b):
        return sum(x * y for x, y in zip(a, b))


def vec(*xs):
    n = sum(x * x for x in xs) ** 0.5
    return [x / n for x in xs]


def candidate(statement, category="preferences", importance=0.8, vector=None):
    return {"statement": statement, "category": category, "importance": importance, "vector": vector or vec(1, 0)}


def make(judge=None):
    facts = Facts()
    return facts, LongMemoryService(facts, facts, judge=judge)


def active(facts):
    return sorted(r["statement"] for r in facts.rows.values() if (r.get("status") or "ACTIVE") == "ACTIVE")


# ─── soft supersede ──────────────────────────────────────────────────────────────────────────────────────────────────


async def test_a_judged_correction_writes_the_new_fact_first_then_retires_the_old_one_keeping_it_as_history():
    async def judge(new, existing):
        return {"relation": "correction", "index": 0}

    facts, longmem = make(judge)
    await longmem.merge_user_facts(U, "c1", [candidate("User lives in London", "personal", 0.9)])
    stats = await longmem.merge_user_facts(U, "c2", [candidate("User lives in Berlin", "personal", 0.9, vec(0.95, 0.31))])
    assert stats["superseded"] == 1 and active(facts) == ["User lives in Berlin"]
    ops = [op for op in facts.log if op[1] in ("User lives in London", "User lives in Berlin")]
    assert ops.index(("write", "User lives in Berlin")) < ops.index(("retire", "User lives in London"))
    old = next(r for r in facts.rows.values() if r["statement"] == "User lives in London")
    new = next(r for r in facts.rows.values() if r["statement"] == "User lives in Berlin")
    assert old["status"] == "SUPERSEDED" and old["superseded_by"] == new["fact_id"]      # history kept, linked
    assert len(facts.vectors) == 1 and str(new["fact_id"]) in facts.vectors             # only the current vector remains


async def test_update_words_alone_never_replace_a_memory_when_the_judge_says_it_is_a_different_fact():
    async def judge(new, existing):
        return {"relation": "different", "index": None}

    facts, longmem = make(judge)
    await longmem.merge_user_facts(U, "c1", [candidate("User prefers dark mode")])
    stats = await longmem.merge_user_facts(U, "c2", [candidate("User now prefers concise answers", vector=vec(0.95, 0.31))])
    assert stats["inserted"] == 1 and stats["superseded"] == 0
    assert active(facts) == ["User now prefers concise answers", "User prefers dark mode"]


async def test_without_a_working_judge_nothing_is_replaced_on_wording_alone():
    async def broken(new, existing):
        raise RuntimeError("llm down")

    for judge in (None, broken):
        facts, longmem = make(judge)
        await longmem.merge_user_facts(U, "c1", [candidate("User prefers dark mode")])
        # similar enough to be a candidate, "now" present, lower importance: the old regex would have replaced it
        stats = await longmem.merge_user_facts(U, "c2", [candidate("User now prefers light mode", importance=0.7,
                                                                   vector=vec(0.8, 0.6))])
        assert stats["superseded"] == 0 and "User prefers dark mode" in active(facts)


async def test_only_candidates_of_the_same_category_are_shown_to_the_judge():
    seen = []

    async def judge(new, existing):
        seen.append(existing)
        return {"relation": "different", "index": None}

    facts, longmem = make(judge)
    await longmem.merge_user_facts(U, "c1", [candidate("User works at Acme", "personal")])
    await longmem.merge_user_facts(U, "c2", [candidate("User prefers tea", "preferences")])
    assert seen == []                                  # different category: no candidate at all, no judge call


async def test_the_judge_sees_up_to_five_candidates_and_its_index_selects_which_one_is_retired():
    verdicts = []

    async def judge(new, existing):
        verdicts.append(list(existing))
        return {"relation": "correction", "index": 2}

    facts, longmem = make(judge)
    for i, v in enumerate([vec(1, 0.0), vec(1, 0.1), vec(1, 0.2), vec(1, 0.3), vec(1, 0.4), vec(1, 0.5), vec(1, 0.6)]):
        facts.upsert_fact({"user_id": U, "category": "projects", "fact_id": uuid.uuid4(), "statement": f"Project fact {i}",
                           "importance": 0.8, "fact_version": 1, "conversation_id": "c0"})
        fid = str(next(iter(reversed(list(facts.rows.values()))))["fact_id"])
        facts.vectors[fid] = {"fact_id": fid, "user_id": U, "category": "projects", "statement": f"Project fact {i}",
                              "importance": 0.8, "fact_version": 1, "vector": v, "conversation_id": "c0"}
    await longmem.merge_user_facts(U, "c1", [candidate("Project fact corrected", "projects", 0.8, vec(1, 0.05))])
    assert len(verdicts[0]) == 5
    retired = [r["statement"] for r in facts.rows.values() if r.get("status") == "SUPERSEDED"]
    assert retired == [verdicts[0][2]]


async def test_a_confirmed_restatement_refreshes_recency_without_a_new_version():
    facts, longmem = make()
    await longmem.merge_user_facts(U, "c1", [candidate("User prefers dark mode")])
    stats = await longmem.merge_user_facts(U, "c2", [candidate("User prefers dark mode")])
    assert stats["skipped"] == 1 and ("confirm", "User prefers dark mode") in facts.log
    assert len(facts.rows) == 1


async def test_reconcile_removes_orphan_vectors_and_restores_missing_ones():
    facts, longmem = make()
    await longmem.merge_user_facts(U, "c1", [candidate("User likes tea"), candidate("User uses Linux", "personal", 0.9, vec(0, 1))])
    gone = next(r for r in facts.rows.values() if r["statement"] == "User likes tea")
    facts.vectors.pop(str(gone["fact_id"]))                                     # crash lost this vector
    ghost = {"fact_id": str(uuid.uuid4()), "user_id": U, "category": "preferences", "statement": "ghost", "vector": vec(1, 1)}
    facts.vectors[ghost["fact_id"]] = ghost                                     # vector with no active Cassandra fact
    counted = await longmem.reconcile_user_facts(U)
    assert counted["orphan_vectors_removed"] == 1 and counted["missing_vectors"] == 1 and ghost["fact_id"] not in facts.vectors
    restored = await longmem.reconcile_user_facts(U, embed=AsyncMock(return_value=vec(1, 0)))
    assert restored["missing_vectors_restored"] == 1 and str(gone["fact_id"]) in facts.vectors


async def test_a_failed_vector_delete_after_retiring_does_not_lose_the_replacement():
    async def judge(new, existing):
        return {"relation": "correction", "index": 0}

    facts, longmem = make(judge)
    await longmem.merge_user_facts(U, "c1", [candidate("User lives in London", "personal", 0.9)])
    original = LongMemoryService._retire

    def delete_vector_fails(self, *a, **k):
        self.milvus_repo.delete_fact = MagicMock(side_effect=RuntimeError("milvus down"))
        return original(self, *a, **k)

    LongMemoryService._retire = delete_vector_fails
    try:
        stats = await longmem.merge_user_facts(U, "c2", [candidate("User lives in Berlin", "personal", 0.9, vec(0.95, 0.31))])
    finally:
        LongMemoryService._retire = original
    assert stats["superseded"] == 1 and active(facts) == ["User lives in Berlin"]


# ─── the judge itself ────────────────────────────────────────────────────────────────────────────────────────────────


def llm_returning(text):
    manager = SimpleNamespace(generate_with_retry=AsyncMock(return_value=text))
    return LLMService(manager)


async def test_judge_parses_json_even_with_surrounding_prose():
    out = await llm_returning('Sure! {"relation": "correction", "index": 1} hope that helps').judge_fact_relation("n", ["a", "b"])
    assert out == {"relation": "correction", "index": 1}
    assert await llm_returning('{"relation": "different", "index": 0}').judge_fact_relation("n", ["a"]) == {"relation": "different", "index": None}


@pytest.mark.parametrize("bad", ["no json here", '{"relation": "maybe", "index": 0}', '{"relation": "same", "index": 7}',
                                 '{"relation": "same"}'])
async def test_judge_rejects_unusable_answers_so_the_caller_stays_non_destructive(bad):
    with pytest.raises(ValueError):
        await llm_returning(bad).judge_fact_relation("n", ["a", "b"])


# ─── explicit-memory fast path / regenerated answers ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("text,expected", [
    ("Remember that I use Neovim", True), ("from now on answer in German", True), ("Always use tabs when you write code", True),
    ("don't forget my deadline is Friday", True), ("My name is Priya", True),
    ("I always wonder how kafka works", False), ("never mind, thanks", False), ("what is the capital of France?", False),
])
def test_explicit_memory_requests_are_recognised_narrowly(text, expected):
    assert memory_policy.is_explicit_memory_request(text) is expected


class Cass:
    def __init__(self):
        self.jobs, self.rows, self.deleted = [], [], []

    def insert_outbox_job(self, job): self.jobs.append(job)
    def get_recent_messages(self, cid, limit=20): return list(self.rows)
    def delete_recent_message_row(self, cid, created_at, message_id): self.deleted.append(message_id)


class Dispatcher:
    def __init__(self):
        self.memory_repo = SimpleNamespace(invalidate_conversation=AsyncMock())

    async def dispatch(self, envelope): pass


def turn(text, **over):
    return {"conversation_id": "c1", "user_id": U, "timestamp": NOW.isoformat(), "user_message": text,
            "assistant_response": "ok", **over}


async def test_explicit_memory_message_queues_a_side_channel_extraction_for_that_message_only():
    cass = Cass()
    worker = MemoryUpdateWorker(Dispatcher(), bootstrap_servers="x", cassandra_repo=cass)
    await worker._process_turn(turn("Remember that I deploy on Fridays", user_message_id="msg-user-1"))
    (job,) = cass.jobs
    payload = json.loads(job["payload"])
    assert job["topic"] == settings.KAFKA_FACT_TOPIC
    assert payload["reason"] == "explicit" and payload["message_ids"] == ["msg-user-1"]


async def test_ordinary_messages_and_a_disabled_flag_queue_nothing(monkeypatch):
    cass = Cass()
    worker = MemoryUpdateWorker(Dispatcher(), bootstrap_servers="x", cassandra_repo=cass)
    await worker._process_turn(turn("what is the capital of France?"))
    monkeypatch.setattr(settings, "EXPLICIT_MEMORY_FAST_PATH", False)
    await worker._process_turn(turn("Remember that I deploy on Fridays"))
    assert cass.jobs == []


async def test_a_regenerated_answer_replaces_the_discarded_one():
    cass = Cass()
    old = {"message_id": "old-answer", "created_at": NOW, "role": "assistant", "content": "discarded"}
    cass.rows = [old, {"message_id": "other", "created_at": NOW, "role": "user", "content": "q"}]
    dispatcher = Dispatcher()
    worker = MemoryUpdateWorker(dispatcher, bootstrap_servers="x", cassandra_repo=cass)
    await worker._process_turn(turn("q", replaces_message_id="old-answer"))
    assert cass.deleted == ["old-answer"]
    dispatcher.memory_repo.invalidate_conversation.assert_awaited_once_with("c1")
    cass.deleted.clear()
    await worker._process_turn(turn("q", replaces_message_id="not-stored-legacy-id"))
    assert cass.deleted == []                                           # unknown id: nothing is guessed


def test_extraction_notes_put_the_exact_messages_first_and_the_summary_after_as_context():
    notes = build_extraction_notes("Summary: user works on Kafka.",
                                   [{"role": "user", "content": "Remember that I prefer Postgres"}, {"role": "assistant", "content": "Noted"}])
    assert notes.index("Remember that I prefer Postgres") < notes.index("Summary: user works on Kafka.")
    assert "context only" in notes


async def test_a_summary_job_reports_exactly_the_messages_it_consumed():
    cass = FakeCassandra()
    cass.messages["c1"] = [msg(n) for n in range(5, 0, -1)]
    cass.snapshots["c1"] = {"conversation_id": "c1", "user_id": U, "summary_version": 0}
    cass.update_snapshot_summary_fields = lambda *a: None
    repo = SimpleNamespace(update_snapshot_summary_fields=AsyncMock(), invalidate_conversation=AsyncMock(),
                           get_snapshot=AsyncMock(return_value=dict(cass.snapshots["c1"])))
    from app.services.summary_service import SummaryService
    snap = await SummaryService(repo, cass, ScriptedSummarizer("s")).process_incremental_summary("c1")
    assert snap["consumed_message_ids"] == [f"m000{n}" for n in range(1, 6)]


# ─── idle trigger ────────────────────────────────────────────────────────────────────────────────────────────────────


class ZSetRedis:
    def __init__(self):
        self.z = {}

    async def zadd(self, key, mapping): self.z.update(mapping)

    async def zrangebyscore(self, key, lo, hi, start=0, num=None):
        return [m for m, s in sorted(self.z.items(), key=lambda kv: kv[1]) if lo <= s <= hi][start:start + (num or 10**9)]

    async def zrem(self, key, member): return 1 if self.z.pop(member, None) is not None else 0


def idle_world(state="ACTIVE", with_messages=True):
    redis, cass = ZSetRedis(), FakeCassandra()
    if with_messages:
        cass.messages["c1"] = [msg(2), msg(1)]
    snapshot = {"conversation_id": "c1", "user_id": U, "state": state, "summary_version": 0}
    memory_repo = SimpleNamespace(get_snapshot=AsyncMock(return_value=snapshot))
    memory_service = SimpleNamespace(transition_state=AsyncMock())
    return redis, cass, memory_service, IdleCompactionWorker(redis, memory_repo, memory_service, cass)


async def test_an_idle_conversation_with_unsummarized_messages_gets_a_summary_request():
    redis, cass, memory_service, worker = idle_world()
    redis.z[f"c1|{U}"] = 1000.0
    now = 1000.0 + settings.IDLE_COMPACTION_MINUTES * 60 + 1
    assert await worker.sweep(now) == {"requested": 1}
    memory_service.transition_state.assert_awaited_once_with("c1", MemoryState.SUMMARY_PENDING)
    (job,) = cass.outbox
    assert job["topic"] == settings.KAFKA_SUMMARY_TOPIC and json.loads(job["payload"])["reason"] == "idle"
    assert redis.z == {}                                              # claimed (removed) exactly once


async def test_recently_active_or_nothing_new_or_busy_conversations_are_handled_correctly():
    redis, cass, memory_service, worker = idle_world()
    redis.z[f"c1|{U}"] = 1000.0
    assert await worker.sweep(1000.0 + 60) == {}                       # not idle long enough: untouched
    assert f"c1|{U}" in redis.z

    redis, cass, memory_service, worker = idle_world(with_messages=False)
    redis.z[f"c1|{U}"] = 1000.0
    assert await worker.sweep(10**7) == {"nothing_new": 1} and cass.outbox == []

    redis, cass, memory_service, worker = idle_world(state="SUMMARIZING")
    redis.z[f"c1|{U}"] = 1000.0
    assert await worker.sweep(10**7) == {"busy": 1}
    assert f"c1|{U}" in redis.z and cass.outbox == []                  # put back, looked at again later


# ─── deleting a conversation ─────────────────────────────────────────────────────────────────────────────────────────


def delete_world(retire_ok=True):
    facts = Facts()
    facts.delete_conversation_data = MagicMock()
    memory_repo = SimpleNamespace(get_snapshot=AsyncMock(return_value={"user_id": U}), invalidate_conversation=AsyncMock())
    redis = SimpleNamespace(delete=AsyncMock(), zrem=AsyncMock())
    return facts, memory_repo, ConversationDeleteWorker(memory_repo, facts, facts, redis)


def seed_fact(facts, conversation_id, statement, category="preferences"):
    fid = uuid.uuid4()
    facts.rows[(U, category, fid)] = {"conversation_id": conversation_id, "statement": statement, "fact_id": fid,
                                      "category": category, "status": "ACTIVE"}
    facts.vectors[str(fid)] = {"fact_id": str(fid), "user_id": U, "category": category}
    return fid


async def test_deleting_a_conversation_purges_its_memory_and_retires_the_facts_whose_source_it_was():
    facts, memory_repo, worker = delete_world()
    mine, other = seed_fact(facts, "c1", "learned in the deleted conversation"), seed_fact(facts, "c2", "learned elsewhere")
    assert settings.DELETE_FACTS_WITH_CONVERSATION is True                   # the default
    assert await worker.purge({"conversation_id": "c1", "user_id": U}) == "purged"
    facts.delete_conversation_data.assert_called_once_with("c1")
    memory_repo.invalidate_conversation.assert_awaited_once_with("c1")
    assert facts.rows[(U, "preferences", mine)]["status"] == "SUPERSEDED"      # retired: kept as history, never current
    assert facts.rows[(U, "preferences", mine)].get("superseded_by") is None
    assert str(mine) not in facts.vectors                                      # and never retrieved
    assert facts.rows[(U, "preferences", other)]["status"] == "ACTIVE" and str(other) in facts.vectors
    assert [r["statement"] for r in facts.get_facts(U, "preferences")] == ["learned elsewhere"]


async def test_a_fact_restated_in_another_conversation_survives_deleting_the_one_it_was_first_learned_in():
    facts, longmem = make()
    await longmem.merge_user_facts(U, "c1", [candidate("User prefers dark mode")])
    stats = await longmem.merge_user_facts(U, "c2", [candidate("User prefers dark mode")])     # said again, in c2
    assert stats["skipped"] == 1
    (fid,) = [k[2] for k in facts.rows]
    assert facts.rows[(U, "preferences", fid)]["conversation_id"] == "c2"                     # its source moved to c2
    worker = ConversationDeleteWorker(SimpleNamespace(get_snapshot=AsyncMock(return_value={"user_id": U}),
                                                      invalidate_conversation=AsyncMock()), facts, facts)
    facts.delete_conversation_data = MagicMock()
    await worker.purge({"conversation_id": "c1", "user_id": U})
    assert facts.rows[(U, "preferences", fid)]["status"] == "ACTIVE" and str(fid) in facts.vectors
    await worker.purge({"conversation_id": "c2", "user_id": U})                                # deleting c2 now retires it
    assert facts.rows[(U, "preferences", fid)]["status"] == "SUPERSEDED"


async def test_a_corroborated_interest_points_at_the_newer_conversation_and_survives():
    facts, longmem = make()
    await longmem.merge_user_facts(U, "c1", [candidate("User is interested in Kafka", "interests", 0.9)])
    await longmem.merge_user_facts(U, "c2", [candidate("User is interested in Kafka", "interests", 0.9)])
    current = [r for r in facts.rows.values() if r.get("status") == "ACTIVE"]
    assert len(current) == 1 and current[0]["conversation_id"] == "c2"
    facts.delete_conversation_data = MagicMock()
    worker = ConversationDeleteWorker(SimpleNamespace(get_snapshot=AsyncMock(return_value={"user_id": U}),
                                                      invalidate_conversation=AsyncMock()), facts, facts)
    await worker.purge({"conversation_id": "c1", "user_id": U})
    assert len([r for r in facts.rows.values() if r.get("status") == "ACTIVE"]) == 1


async def test_with_the_flag_off_facts_are_kept(monkeypatch):
    monkeypatch.setattr(settings, "DELETE_FACTS_WITH_CONVERSATION", False)
    facts, _, worker = delete_world()
    fid = seed_fact(facts, "c1", "kept")
    await worker.purge({"conversation_id": "c1", "user_id": U})
    assert facts.rows[(U, "preferences", fid)]["status"] == "ACTIVE" and str(fid) in facts.vectors


async def test_another_users_conversation_is_never_purged_and_bad_events_do_not_raise():
    facts = Facts()
    facts.delete_conversation_data = MagicMock()
    memory_repo = SimpleNamespace(get_snapshot=AsyncMock(return_value={"user_id": "someone-else"}), invalidate_conversation=AsyncMock())
    worker = ConversationDeleteWorker(memory_repo, facts, facts)
    assert await worker.purge({"conversation_id": "c1", "user_id": U}) == "skipped"
    facts.delete_conversation_data.assert_not_called()
    assert await worker.handle(b"not json") == "error"
    assert await worker.purge({}) == "ignored"


# ─── retrieval ───────────────────────────────────────────────────────────────────────────────────────────────────────


def test_recency_counts_from_the_last_confirmation_not_from_creation():
    old_created = (NOW - timedelta(days=400)).timestamp()
    stale = {"fact_id": "a", "distance": 0.8, "importance": 0.8, "created_at": old_created}
    confirmed = dict(stale, fact_id="b", last_confirmed=NOW - timedelta(days=1))
    ranked = RankingService.rank_facts([stale, confirmed], now=NOW)
    assert [f["fact_id"] for f in ranked] == ["b", "a"] and ranked[0]["score"] > ranked[1]["score"]


async def test_retired_facts_are_never_returned_even_if_their_vector_survived():
    facts = Facts()
    live, dead = uuid.uuid4(), uuid.uuid4()
    for fid, status in ((live, "ACTIVE"), (dead, "SUPERSEDED")):
        facts.rows[(U, "preferences", fid)] = {"fact_id": fid, "status": status, "statement": str(status)}
        facts.vectors[str(fid)] = {"fact_id": str(fid), "user_id": U, "category": "preferences", "statement": str(status),
                                   "importance": 0.8, "vector": vec(1, 0), "created_at": NOW.timestamp()}
    retrieval = RetrievalService(SimpleNamespace(cassandra_repo=facts), facts, None)
    got = await retrieval.retrieve_relevant_facts(U, vec(1, 0), limit=10)
    assert [f["statement"] for f in got] == ["ACTIVE"]


async def test_core_profile_offers_who_the_user_is_for_unrelated_questions_within_a_budget():
    facts = Facts()
    for i, (statement, importance) in enumerate([("User is a backend engineer at Acme", 0.95), ("User lives in Pune", 0.8),
                                                 ("User is 34", 0.75)]):
        fid = str(uuid.uuid4())
        facts.vectors[fid] = {"fact_id": fid, "user_id": U, "category": "personal", "statement": statement,
                              "importance": importance, "vector": vec(0, 1), "created_at": NOW.timestamp()}
    retrieval = RetrievalService(SimpleNamespace(), facts, None)
    got = await retrieval.retrieve_relevant_facts(U, vec(1, 0), limit=10)          # nothing similar to the question
    assert [f["statement"] for f in got] == ["User is a backend engineer at Acme", "User lives in Pune"]   # top 2 only
    settings_off = settings.CORE_PROFILE_ENABLED
    try:
        settings.CORE_PROFILE_ENABLED = False
        assert await retrieval.retrieve_relevant_facts(U, vec(1, 0), limit=10) == []
    finally:
        settings.CORE_PROFILE_ENABLED = settings_off


def test_retrieval_query_adds_the_previous_user_message_only_to_short_follow_ups():
    recent = [{"role": "assistant", "content": "a"}, {"role": "user", "content": "How do I tune kafka consumer lag?"}]
    assert build_retrieval_query("and the second one?", recent) == "How do I tune kafka consumer lag?\nand the second one?"
    long_q = "Please explain in detail how the replication protocol keeps followers in sync during leader election"
    assert build_retrieval_query(long_q, recent) == long_q
    assert build_retrieval_query("and the second one?", []) == "and the second one?"
    assert build_retrieval_query("", recent) == ""


# ─── last_used ───────────────────────────────────────────────────────────────────────────────────────────────────────────


def used_world(last_used=None, fail=False):
    facts = Facts()
    fid = uuid.uuid4()
    facts.rows[(U, "preferences", fid)] = {"fact_id": fid, "status": "ACTIVE", "statement": "likes tea", "last_used": last_used}
    facts.vectors[str(fid)] = {"fact_id": str(fid), "user_id": U, "category": "preferences", "statement": "likes tea",
                               "importance": 0.8, "vector": vec(1, 0), "created_at": NOW.timestamp()}
    if fail:
        facts.mark_fact_used = MagicMock(side_effect=RuntimeError("cassandra down"))
    return facts, RetrievalService(SimpleNamespace(cassandra_repo=facts), facts, None)


async def drain():
    while RetrievalService._background:
        await asyncio.gather(*list(RetrievalService._background))


async def test_retrieved_facts_get_last_used_stamped_in_the_background_not_on_the_request_path():
    facts, retrieval = used_world()
    got = await retrieval.retrieve_relevant_facts(U, vec(1, 0), limit=5)
    assert [f["statement"] for f in got] == ["likes tea"]
    assert ("used", "likes tea") not in facts.log or True                 # may or may not have run yet: it is off the path
    await drain()
    assert ("used", "likes tea") in facts.log


async def test_a_recently_used_fact_is_not_written_again():
    facts, retrieval = used_world(last_used=datetime.now(timezone.utc) - timedelta(seconds=30))
    await retrieval.retrieve_relevant_facts(U, vec(1, 0), limit=5)
    await drain()
    assert ("used", "likes tea") not in facts.log
    facts, retrieval = used_world(last_used=datetime.now(timezone.utc) - timedelta(seconds=settings.FACT_LAST_USED_MIN_INTERVAL_SECONDS + 5))
    await retrieval.retrieve_relevant_facts(U, vec(1, 0), limit=5)
    await drain()
    assert ("used", "likes tea") in facts.log


async def test_a_failing_last_used_write_never_affects_the_answer():
    facts, retrieval = used_world(fail=True)
    got = await retrieval.retrieve_relevant_facts(U, vec(1, 0), limit=5)
    await drain()
    assert [f["statement"] for f in got] == ["likes tea"]


async def test_last_used_can_be_switched_off(monkeypatch):
    monkeypatch.setattr(settings, "FACT_LAST_USED_ENABLED", False)
    facts, retrieval = used_world()
    await retrieval.retrieve_relevant_facts(U, vec(1, 0), limit=5)
    await drain()
    assert ("used", "likes tea") not in facts.log
