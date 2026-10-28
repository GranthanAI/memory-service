"""Phase 4: branch-safe summaries - immutable summary versions, version choice at the branch point, raw-message bridge,
bounded summaries and the branch-creation check. UUIDv7 ids like conversation-service mints; in-memory stores."""
import json
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.repositories.memory_repository import MemoryRepository
from app.services.context_builder import ContextBuilder
from app.services.summary_service import SummaryService
from app.workers.branch_summary_worker import BranchSummaryWorker
from tests.support.fakes import FakeCassandra, ScriptedSummarizer

pytestmark = pytest.mark.asyncio

T0 = datetime(2026, 10, 1, 9, 0, tzinfo=timezone.utc)
USER = "u1"


def v7(ts: datetime, rand: int = 1) -> str:
    ms = int(ts.timestamp() * 1000)
    return str(uuid.UUID(int=(ms << 80) | (0x7 << 76) | (0b10 << 62) | rand))


def message(n: int, conv: str = "A"):
    ts = T0 + timedelta(seconds=10 * n)
    return {"message_id": v7(ts, n), "role": "user" if n % 2 else "assistant", "content": f"{conv} message {n}", "created_at": ts}


class Redis:
    async def get_snapshot(self, cid): return None
    async def set_snapshot(self, s): pass
    async def invalidate_conversation(self, cid): pass


class World:
    def __init__(self):
        self.cass = FakeCassandra()
        self.repo = MemoryRepository(self.cass, Redis())
        self.cass.update_snapshot_summary_fields = lambda *a: None
        self.summarizer = ScriptedSummarizer()
        self.service = SummaryService(self.repo, self.cass, self.summarizer)

    def add_conversation(self, cid, count):
        self.cass.snapshots[cid] = {"conversation_id": cid, "user_id": USER, "message_count": count, "state": "ACTIVE",
                                    "summary_version": 0, "fact_version": 0, "snapshot_version": 1}
        self.cass.messages[cid] = []
        self.add_messages(cid, 1, count)

    def add_messages(self, cid, first, last):
        for n in range(first, last + 1):
            self.cass.append_recent_message(cid, message(n, cid))

    async def summarize(self, cid, **kw):
        self.cass.snapshots[cid].setdefault("summary_version", 0)
        await self.service.process_incremental_summary(cid, **kw)

    def builder(self):
        retrieval = SimpleNamespace(memory_repo=SimpleNamespace(
            cassandra_repo=self.cass, get_snapshot=AsyncMock(side_effect=lambda cid: self.cass.get_snapshot(cid))))
        return ContextBuilder(retrieval, graph_client=None)

    async def inherited(self, *refs):
        return await self.builder().ancestor_summaries(USER, [{"conversation_id": c, "upto_message_id": message(n, c)["message_id"]}
                                                              for c, n in refs])


def summary_text_for(world, cid):
    return world.cass.summaries[cid]["summary_text"]


async def test_superseded_summaries_are_archived_and_never_rewritten():
    w = World()
    w.add_conversation("A", 4)
    await w.summarize("A")
    w.add_messages("A", 5, 8)
    await w.summarize("A")
    assert w.cass.summaries["A"]["summary_version"] == 2
    archived = w.cass.list_summary_versions("A")
    assert [r["summary_version"] for r in archived] == [1]                 # v1 kept; v2 is the current row
    v1_text = archived[0]["summary_text"]
    w.cass.archive_summary_version({**archived[0], "summary_text": "TAMPERED"})
    assert w.cass.list_summary_versions("A")[0]["summary_text"] == v1_text   # immutable (IF NOT EXISTS)


async def test_child_branched_before_the_parent_kept_summarizing_gets_the_earlier_version_without_leaks():
    w = World()
    w.add_conversation("A", 4)
    await w.summarize("A")                                   # v1 through message 4
    v1 = summary_text_for(w, "A")
    w.add_messages("A", 5, 8)
    await w.summarize("A")                                   # v2 through message 8 (past the branch point)
    assert w.cass.summaries["A"]["summary_version"] == 2

    (inherited,) = await w.inherited(("A", 4))               # child branched at message 4
    assert inherited["summary"] == v1 and inherited["summary_version"] == 1
    assert "message 5" not in inherited["summary"] and "message 8" not in inherited["summary"]
    assert summary_text_for(w, "A") != v1                    # the parent's current summary is NOT what the child gets


async def test_gap_between_the_chosen_version_and_the_branch_point_is_bridged_with_raw_messages_only_up_to_the_branch():
    w = World()
    w.add_conversation("A", 4)
    await w.summarize("A")                                   # v1 through 4
    w.add_messages("A", 5, 10)
    await w.summarize("A")                                   # v2 through 10
    (inherited,) = await w.inherited(("A", 7))               # branch at message 7: v1 is the latest valid version
    text = inherited["summary"]
    assert text.startswith(w.cass.list_summary_versions("A")[0]["summary_text"])
    for n in (5, 6, 7):
        assert f"A message {n}" in text                      # bridged
    for n in (8, 9, 10):
        assert f"A message {n}" not in text                  # nothing said after the branch point
    assert inherited["covers_through_message_id"] == message(7)["message_id"]


async def test_bridge_is_limited_to_its_token_budget_keeping_the_messages_closest_to_the_branch(monkeypatch):
    from app.core.config import settings
    w = World()
    w.add_conversation("A", 2)
    await w.summarize("A")
    w.add_messages("A", 3, 40)
    await w.summarize("A")
    monkeypatch.setattr(settings, "ANCESTOR_BRIDGE_MAX_TOKENS", 60)
    (inherited,) = await w.inherited(("A", 30))
    assert "A message 30" in inherited["summary"] and "A message 3\n" not in inherited["summary"]
    assert "A message 31" not in inherited["summary"]


async def test_grandparent_parent_child_lineage_returns_each_ancestors_summary_valid_at_its_own_branch_point():
    w = World()
    w.add_conversation("G", 6)
    await w.summarize("G")                                   # G v1 through 6
    g1 = summary_text_for(w, "G")
    w.add_messages("G", 7, 9)
    await w.summarize("G")                                   # G moved on (v2)
    w.add_conversation("P", 5)
    await w.summarize("P")                                   # P v1 through 5
    p1 = summary_text_for(w, "P")
    w.add_messages("P", 6, 9)
    await w.summarize("P")                                   # P moved on too
    out = await w.inherited(("G", 6), ("P", 5))              # C branched from P at 5, P branched from G at 6
    assert [(a["conversation_id"], a["summary"], a["summary_version"]) for a in out] == [("G", g1, 1), ("P", p1, 1)]


async def test_a_summary_that_cannot_be_proven_valid_and_has_no_archive_is_omitted():
    w = World()
    w.add_conversation("A", 8)
    w.cass.summaries["A"] = {"conversation_id": "A", "summary_text": "legacy summary", "summary_version": 3,
                             "through_message_id": message(8)["message_id"], "through_created_at": message(8)["created_at"]}
    assert await w.inherited(("A", 4)) == []                 # past the branch point and nothing archived: never guess


async def test_legacy_current_row_without_any_versions_still_works():
    w = World()
    w.add_conversation("A", 8)
    w.cass.summaries["A"] = {"conversation_id": "A", "summary_text": "legacy summary", "summary_version": 1,
                             "through_message_id": message(4)["message_id"], "through_created_at": message(4)["created_at"]}
    (out,) = await w.inherited(("A", 6))
    assert out["summary"].startswith("legacy summary") and "A message 5" in out["summary"]


async def test_an_archive_failure_aborts_the_commit_so_no_version_is_lost():
    w = World()
    w.add_conversation("A", 4)
    await w.summarize("A")
    before = dict(w.cass.summaries["A"])
    w.add_messages("A", 5, 6)

    def boom(row):
        raise RuntimeError("cassandra down")

    w.cass.archive_summary_version = boom
    with pytest.raises(RuntimeError):
        await w.summarize("A")
    assert w.cass.summaries["A"] == before


async def test_summary_can_be_bounded_at_a_branch_point():
    w = World()
    w.add_conversation("A", 10)
    await w.summarize("A", upto_message_id=message(6)["message_id"])
    row = w.cass.summaries["A"]
    assert row["through_message_id"] == message(6)["message_id"]
    sent = [m.content for call in w.summarizer.calls for m in call.new_messages]
    assert "A message 6" in sent and "A message 7" not in sent
    await w.summarize("A")                                   # the next unbounded job continues from the watermark
    assert w.cass.summaries["A"]["through_message_id"] == message(10)["message_id"]


# ─── branch creation ─────────────────────────────────────────────────────────────────────────────────────────────────


def branch_worker(w):
    return BranchSummaryWorker(w.repo, w.cass, scheduler=None, bootstrap_servers="unused")


def created(parent, upto_n):
    return {"conversation_id": "child", "user_id": USER, "parent_conversation_id": parent,
            "branch_message_id": message(upto_n, parent)["message_id"]}


async def test_branch_creation_requests_a_bounded_summary_when_the_parent_is_behind_the_branch_point():
    w = World()
    w.add_conversation("A", 8)
    await w.summarize("A", upto_message_id=message(3)["message_id"])
    assert await branch_worker(w).ensure_branch_summary(created("A", 6)) == "requested"
    (job,) = w.cass.outbox
    payload = json.loads(job["payload"])
    assert payload["upto_message_id"] == message(6, "A")["message_id"] and payload["reason"] == "branch"


async def test_branch_creation_with_no_parent_summary_yet_requests_one():
    w = World()
    w.add_conversation("A", 8)
    assert await branch_worker(w).ensure_branch_summary(created("A", 5)) == "requested"


async def test_branch_creation_is_a_no_op_when_a_version_already_ends_at_the_branch_message():
    w = World()
    w.add_conversation("A", 8)
    await w.summarize("A", upto_message_id=message(5)["message_id"])
    assert await branch_worker(w).ensure_branch_summary(created("A", 5)) == "present"
    assert w.cass.outbox == []


async def test_branch_creation_reports_unavailable_when_the_parent_is_past_the_branch_point_with_no_archive():
    w = World()
    w.add_conversation("A", 8)
    w.cass.summaries["A"] = {"conversation_id": "A", "summary_text": "s", "summary_version": 2,
                             "through_message_id": message(8)["message_id"], "through_created_at": message(8)["created_at"]}
    assert await branch_worker(w).ensure_branch_summary(created("A", 4)) == "unavailable"
    assert w.cass.outbox == []


async def test_branch_creation_ignores_non_branches_and_other_users_conversations():
    w = World()
    w.add_conversation("A", 8)
    worker = branch_worker(w)
    assert await worker.ensure_branch_summary({"conversation_id": "x", "user_id": USER}) == "not_a_branch"
    assert await worker.ensure_branch_summary({**created("A", 3), "user_id": "someone-else"}) == "skipped"
    assert await worker.handle(b"not json") == "error"       # best effort: never raises
