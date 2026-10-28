"""Reproductions of the memory-remediation findings (see docs/memory-remediation.md).

Each test asserts the CORRECT behaviour and is marked xfail(strict=True) while the bug exists. When the fix lands the
test starts passing, strict xfail turns that into a failure, and the fix commit removes the marker. Run
`pytest --runxfail tests/unit/test_memory_remediation_repro.py` to see the raw failures.
"""

import random
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from app.services.context_builder import ContextBuilder
from app.services.summary_service import SummaryService, message_key
from app.workers.memory_update_worker import MemoryUpdateWorker, _derive_id

pytestmark = pytest.mark.asyncio(loop_scope="function")

T0 = datetime(2026, 10, 1, 9, 0, tzinfo=timezone.utc)


def uuid7_at(ts: datetime, rand: int = 0) -> str:
    """A real UUIDv7: 48-bit unix-ms timestamp, version 7, variant 10."""
    ms = int(ts.timestamp() * 1000)
    value = (ms << 80) | (0x7 << 76) | ((rand & 0xFFF) << 64) | (0b10 << 62) | (rand & ((1 << 62) - 1))
    return str(uuid.UUID(int=value))


# ─── P0-1: user and assistant of one turn share created_at and have random uuid5 ids ──────────────────────────────────


class RecordingDispatcher:
    def __init__(self):
        self.envelopes = []

    async def dispatch(self, envelope):
        self.envelopes.append(envelope)


async def stored_order(turn_timestamps):
    dispatcher = RecordingDispatcher()
    worker = MemoryUpdateWorker(dispatcher, bootstrap_servers="unused")
    for i, ts in enumerate(turn_timestamps):
        await worker._process_turn({
            "conversation_id": "c1", "user_id": "u1", "timestamp": ts.isoformat(),
            "user_message": f"question {i}", "assistant_response": f"answer {i}",
        })
    rows = [{**e["payload"], "message_id": e["payload"]["message_id"]} for e in dispatcher.envelopes]
    rows.sort(key=message_key)                       # the order the Cassandra clustering key / summarizer use
    return [r["role"] for r in rows]


async def test_legacy_event_turns_are_stored_user_then_assistant():
    roles = await stored_order([T0 + timedelta(seconds=10 * i) for i in range(200)])
    assert roles == ["user", "assistant"] * 200


# ─── P0-2: uuid5 watermark compared with a UUIDv7 branch point ────────────────────────────────────────────────────────


class Repo:
    def __init__(self):
        self.snapshots, self.summaries = {}, {}
        self.cassandra_repo = SimpleNamespace(get_summary=lambda cid: self.summaries.get(cid))

    async def get_snapshot(self, cid):
        return self.snapshots.get(cid)


def ancestor_case(through_ts, branch_ts, seed):
    """Parent summary through a legacy (uuid5) message at through_ts; child branched at branch_ts (UUIDv7)."""
    repo = Repo()
    through = _derive_id(f"c1:{through_ts.isoformat()}", "assistant", "msg")
    repo.snapshots["A"] = {"conversation_id": "A", "user_id": "u1"}
    repo.summaries["A"] = {"summary_text": "parent summary", "through_message_id": through,
                           "through_created_at": through_ts, "summary_version": 1}
    builder = ContextBuilder(SimpleNamespace(memory_repo=repo), graph_client=None)
    return builder.ancestor_summaries("u1", [{"conversation_id": "A", "upto_message_id": uuid7_at(branch_ts, seed)}])


async def test_valid_legacy_ancestor_summary_is_accepted():
    rng = random.Random(7)
    accepted = 0
    for i in range(200):
        through_ts = T0 + timedelta(seconds=60 * i)
        out = await ancestor_case(through_ts, through_ts + timedelta(seconds=30), rng.getrandbits(62))
        accepted += bool(out)
    assert accepted == 200                            # today about 0.6% pass


async def test_legacy_ancestor_summary_past_the_branch_point_is_rejected():
    """Safety half: a summary covering messages after the branch point must never be shown to the child."""
    rng = random.Random(8)
    for i in range(200):
        through_ts = T0 + timedelta(seconds=60 * i)
        out = await ancestor_case(through_ts, through_ts - timedelta(seconds=30), rng.getrandbits(62))
        assert out == []


# ─── P0-3: snapshot overwritten after the LLM call ────────────────────────────────────────────────────────────────────


class Cassandra:
    def __init__(self, messages):
        self.messages, self.row = messages, None

    def get_summary(self, cid):
        return dict(self.row) if self.row else None

    def get_recent_messages(self, cid, limit=20):
        return sorted(self.messages, key=lambda m: (m["created_at"], m["message_id"]), reverse=True)[:limit]

    def write_summary_if_current(self, summary, expected_version):
        if (self.row["summary_version"] if self.row else None) != expected_version:
            return False
        self.row = dict(summary)
        return True


class SnapshotStore:
    """Cassandra snapshot row; the dispatcher keeps incrementing message_count while the LLM call is in flight."""

    def __init__(self):
        self.snapshot = {"conversation_id": "c1", "summary_version": 0, "last_summary_msg_id": None,
                         "message_count": 9, "snapshot_version": 9}

    async def get_snapshot(self, cid):
        return dict(self.snapshot)

    async def save_snapshot(self, snap):
        self.snapshot = dict(snap)

    async def invalidate_conversation(self, cid):
        pass


@pytest.mark.xfail(strict=True, reason="P0-3: save_snapshot overwrites the whole row read before the LLM call")
async def test_summary_commit_does_not_reset_counters_bumped_during_the_llm_call():
    store = SnapshotStore()
    messages = [{"message_id": f"m{i}", "role": "user", "content": f"message {i}",
                 "created_at": T0 + timedelta(seconds=i)} for i in range(3)]

    async def summarize(request):
        store.snapshot["message_count"] += 1          # dispatcher ingests a message meanwhile
        store.snapshot["snapshot_version"] += 1
        return SimpleNamespace(summary="a summary")

    service = SummaryService(store, Cassandra(messages), SimpleNamespace(summarize=summarize))
    await service.process_incremental_summary("c1")
    assert store.snapshot["message_count"] == 10      # not rolled back to 9
    assert store.snapshot["summary_version"] == 1
