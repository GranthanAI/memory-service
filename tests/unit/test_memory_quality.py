"""Memory quality regression: a multi-turn scenario run through the real policy, merge and retrieval
code (only the stores and the embedding are in-memory fakes). Asserts on what is STORED and what is
RETRIEVED, not on API success."""

import math
import re
import uuid
from collections import Counter
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from app.services import memory_policy
from app.services.long_memory_service import LongMemoryService
from app.services.memory_controls import MemoryControls
from app.services.retrieval_service import RetrievalService
from app.services.user_memory_service import UserMemoryService

pytestmark = pytest.mark.asyncio(loop_scope="function")

ALICE = str(uuid.uuid4())
BOB = str(uuid.uuid4())


def embed(text: str) -> dict:
    """Deterministic bag-of-words 'embedding' (unit length) over the normalized statement."""
    counts = Counter(re.findall(r"\w+", memory_policy.normalize_statement(text)))
    norm = math.sqrt(sum(v * v for v in counts.values())) or 1.0
    return {k: v / norm for k, v in counts.items()}


def cosine(a: dict, b: dict) -> float:
    return sum(v * b.get(k, 0.0) for k, v in a.items())


class Cassandra:
    def __init__(self):
        self.enabled, self.tombs, self.facts = {}, {}, {}

    def get_memory_enabled(self, u):
        return self.enabled.get(u)

    def set_memory_enabled(self, u, e):
        self.enabled[u] = e

    def add_fact_tombstone(self, u, h):
        self.tombs.setdefault(u, set()).add(h)

    def get_fact_tombstones(self, u):
        return set(self.tombs.get(u, set()))

    def get_facts(self, u, c, include_inactive=False):
        return [r for (uu, cc, _), r in self.facts.items()
                if uu == u and cc == c and (include_inactive or (r.get("status") or "ACTIVE") == "ACTIVE")]

    def retire_fact(self, u, c, f, superseded_by):
        self.facts[(u, c, f)]["status"] = "SUPERSEDED"
        self.facts[(u, c, f)]["superseded_by"] = superseded_by

    def touch_fact_confirmed(self, u, c, f):
        pass

    def get_fact(self, u, c, f):
        return self.facts.get((u, c, f))

    def upsert_fact(self, f):
        self.facts[(f["user_id"], f["category"], f["fact_id"])] = dict(f)

    def delete_fact(self, u, c, f):
        self.facts.pop((u, c, f), None)


class Milvus:
    def __init__(self):
        self.rows = {}

    def insert_facts(self, records):
        for r in records:
            self.rows[r["fact_id"]] = dict(r)

    def search_facts(self, user_id, query_vector, limit=5, category=None, **_):
        hits = [dict(r, distance=cosine(query_vector, r["vector"])) for r in self.rows.values()
                if r["user_id"] == user_id and (category is None or r["category"] == category)]
        return sorted(hits, key=lambda h: h["distance"], reverse=True)[:limit]

    def list_user_facts(self, user_id, limit=500):
        return [{k: v for k, v in r.items() if k != "vector"} for r in self.rows.values() if r["user_id"] == user_id]

    def delete_fact(self, user_id, fact_id):
        self.rows.pop(str(fact_id), None)


async def judge(new_statement, existing):
    """Stands in for the LLM judgement: a statement that says it replaces something is a correction of the closest
    memory; anything else that got this far is a restatement."""
    replaces = " instead of " in new_statement or " no longer " in new_statement
    return {"relation": "correction" if replaces else "same", "index": 0}


@pytest.fixture
def world():
    cass, milvus = Cassandra(), Milvus()
    return cass, milvus, LongMemoryService(cass, milvus, judge=judge), RetrievalService(None, milvus, MemoryControls(cass))


def cand(category, importance, statement):
    return {"category": category, "importance": importance, "statement": statement, "vector": embed(statement)}


def stored(cass, user=ALICE):
    return sorted((r["category"], r["statement"], r["importance"], r["fact_version"])
                  for (u, _c, _f), r in cass.facts.items() if u == user and (r.get("status") or "ACTIVE") == "ACTIVE")


async def recall(retrieval, query, user=ALICE):
    return [f["statement"] for f in await retrieval.retrieve_relevant_facts(user, embed(query), limit=10)]


# --------------------------------------------------------------------------- the scenario


async def test_one_off_questions_and_temporary_context_are_not_memories(world):
    cass, _, longmem, _ = world
    stats = await longmem.merge_user_facts(ALICE, "c1", [
        cand("interests", 0.7, "User asked about Kafka consumer groups"),
        cand("personal", 0.6, "User is debugging this error today"),
        cand("interests", 0.7, "User wants to know how partitions work"),
        cand("personal", 0.3, "User mentioned a movie"),
    ])
    assert stats["rejected"] == 4 and stats["inserted"] == 0 and stored(cass) == []


async def test_stable_preference_and_explicit_goal_are_kept_with_high_importance(world):
    cass, _, longmem, _ = world
    await longmem.merge_user_facts(ALICE, "c1", [
        cand("preferences", 0.8, "User prefers step-by-step explanations"),
        cand("goals", 0.9, "User is preparing for a distributed systems interview"),
    ])
    rows = {(c, s): imp for c, s, imp, _v in stored(cass)}
    assert rows[("preferences", "User prefers step-by-step explanations")] >= 0.8
    assert rows[("goals", "User is preparing for a distributed systems interview")] >= 0.9


async def test_rewording_does_not_create_a_second_memory(world):
    cass, _, longmem, _ = world
    await longmem.merge_user_facts(ALICE, "c1", [cand("preferences", 0.8, "User prefers step-by-step explanations")])
    stats = await longmem.merge_user_facts(ALICE, "c2", [
        cand("preferences", 0.8, "user prefers step by step explanations."),
        cand("personal", 0.8, "User prefers step-by-step explanations"),     # even under another category
    ])
    assert stats["inserted"] == 0 and len(stored(cass)) == 1


async def test_repeated_questions_about_a_topic_consolidate_into_one_capped_interest(world):
    cass, _, longmem, _ = world
    await longmem.merge_user_facts(ALICE, "c1", [cand("interests", 0.9, "User is interested in Kafka and event streaming")])
    first = stored(cass)
    assert len(first) == 1 and first[0][2] <= memory_policy.INTEREST_CAP          # frequency != importance
    # the same conversation producing it again changes nothing
    await longmem.merge_user_facts(ALICE, "c1", [cand("interests", 0.9, "User is interested in Kafka and streaming events")])
    assert stored(cass) == first
    # a different conversation corroborates it: still ONE memory, now more important, new version
    await longmem.merge_user_facts(ALICE, "c2", [cand("interests", 0.9, "User is interested in Kafka and streaming events")])
    after = stored(cass)
    assert len(after) == 1 and after[0][2] > first[0][2] and after[0][3] == 2
    assert after[0][2] <= memory_policy.CORROBORATED_INTEREST_CAP


async def test_a_changed_preference_replaces_the_old_one(world):
    cass, milvus, longmem, _ = world
    await longmem.merge_user_facts(ALICE, "c1", [cand("preferences", 0.8, "User prefers step-by-step explanations")])
    await longmem.merge_user_facts(ALICE, "c3", [
        cand("preferences", 0.7, "User now prefers concise answers instead of step-by-step explanations")])
    rows = stored(cass)
    assert [s for _c, s, _i, _v in rows] == ["User now prefers concise answers instead of step-by-step explanations"]
    assert rows[0][3] == 2 and len(milvus.rows) == 1                               # old vector removed too
    old = [r for r in cass.facts.values() if r["statement"] == "User prefers step-by-step explanations"]
    assert len(old) == 1 and old[0]["status"] == "SUPERSEDED"                      # kept as history, not deleted
    assert old[0]["superseded_by"] == next(r["fact_id"] for r in cass.facts.values() if r.get("status") != "SUPERSEDED")


async def test_without_update_wording_a_weaker_near_duplicate_does_not_replace(world):
    cass, _, longmem, _ = world
    await longmem.merge_user_facts(ALICE, "c1", [cand("preferences", 0.9, "User prefers short code examples for backend work")])
    stats = await longmem.merge_user_facts(ALICE, "c2", [cand("preferences", 0.6, "User prefers short code examples for backend tasks")])
    assert stats["ignored"] == 1 and len(stored(cass)) == 1


async def test_deleted_fact_is_never_retrieved_or_relearned(world):
    cass, milvus, longmem, retrieval = world
    await longmem.merge_user_facts(ALICE, "c1", [cand("goals", 0.9, "User is preparing for a distributed systems interview")])
    fid = next(iter(cass.facts))[2]
    UserMemoryService(cass, milvus, AsyncMock(), MemoryControls(cass)).delete_fact(ALICE, fid, "goals")
    assert await recall(retrieval, "distributed systems interview") == []
    stats = await longmem.merge_user_facts(ALICE, "c9", [cand("goals", 0.9, "User is preparing for a distributed systems interview.")])
    assert stats["inserted"] == 0 and stored(cass) == []


async def test_pausing_stops_learning_and_using_memory_midway(world):
    cass, _, longmem, retrieval = world
    await longmem.merge_user_facts(ALICE, "c1", [cand("preferences", 0.8, "User prefers step-by-step explanations")])
    MemoryControls(cass).set_enabled(ALICE, False)
    stats = await longmem.merge_user_facts(ALICE, "c2", [cand("goals", 0.9, "User wants to learn Rust this year")])
    assert stats["inserted"] == 0 and len(stored(cass)) == 1
    assert await recall(retrieval, "explain things") == []                         # not used while paused
    MemoryControls(cass).set_enabled(ALICE, True)
    assert await recall(retrieval, "explain things")                               # back on: kept memories return


async def test_stable_preferences_apply_to_unrelated_questions_and_other_users_never_leak(world):
    cass, _, longmem, retrieval = world
    await longmem.merge_user_facts(ALICE, "c1", [
        cand("preferences", 0.8, "User prefers step-by-step explanations"),
        cand("projects", 0.7, "User is building a RAG chatbot with FastAPI"),
    ])
    await longmem.merge_user_facts(BOB, "b1", [cand("preferences", 0.9, "User prefers answers in Spanish")])
    unrelated = await recall(retrieval, "recipe for pasta carbonara")
    assert "User prefers step-by-step explanations" in unrelated                  # topic-independent preference
    assert "User is building a RAG chatbot with FastAPI" not in unrelated          # topical project is not forced in
    related = await recall(retrieval, "how should I structure my FastAPI RAG chatbot")
    assert "User is building a RAG chatbot with FastAPI" in related
    assert all("Spanish" not in s for s in unrelated + related)                    # Bob's memory never reaches Alice
    assert any("Spanish" in s for s in await recall(retrieval, "answers", user=BOB))


async def test_injected_memory_respects_the_character_budget(world, monkeypatch):
    from app.core.config import settings

    cass, _, longmem, retrieval = world
    monkeypatch.setattr(settings, "RETRIEVAL_FACT_CHAR_BUDGET", 120)
    await longmem.merge_user_facts(ALICE, "c1", [
        cand("preferences", 0.9, "User prefers detailed answers with worked examples and short summaries at the end"),
        cand("preferences", 0.9, "User prefers answers written in British English spelling and formal tone"),
        cand("preferences", 0.9, "User prefers tables for comparisons between several alternatives"),
    ])
    facts = await retrieval.retrieve_relevant_facts(ALICE, embed("anything"), limit=10)
    assert 1 <= len(facts) < 3 and sum(len(f["statement"]) for f in facts) <= 120 + len(facts[0]["statement"])


@pytest.mark.parametrize("raw,expected", [("habits", "preferences"), ("plans", "goals"), ("Project", "projects"),
                                           ("response style", "preferences"), ("unknown-thing", "personal")])
def test_categories_map_onto_the_canonical_set(raw, expected):
    assert memory_policy.canonical_category(raw) == expected


def test_policy_rejects_weak_or_conversation_specific_candidates_and_accepts_durable_ones():
    assert not memory_policy.evaluate("preferences", 0.9, "short").accepted
    assert not memory_policy.evaluate("personal", 0.4, "User works as a backend engineer").accepted
    assert memory_policy.evaluate("personal", 0.7, "User works as a backend engineer").accepted
    assert memory_policy.evaluate("interests", 0.95, "User follows Formula 1 closely").importance == memory_policy.INTEREST_CAP
