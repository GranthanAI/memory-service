"""Phase 1: message identity and ordering (new event shape + legacy shape) and branch-validity of summaries."""
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from app.services.context_builder import ContextBuilder, summary_valid_at_branch
from app.services.summary_service import message_key, select_new_messages
from app.utils.message_ids import is_uuid7, uuid7_datetime
from app.workers.memory_update_worker import (
    LEGACY_ASSISTANT_OFFSET, MemoryUpdateWorker, _derive_id, turn_identity,
)

pytestmark = pytest.mark.asyncio(loop_scope="function")

T0 = datetime(2026, 10, 1, 9, 0, tzinfo=timezone.utc)


def uuid7_at(ts: datetime, rand: int = 0) -> str:
    ms = int(ts.timestamp() * 1000)
    value = (ms << 80) | (0x7 << 76) | ((rand & 0xFFF) << 64) | (0b10 << 62) | (rand & ((1 << 62) - 1))
    return str(uuid.UUID(int=value))


class RecordingDispatcher:
    def __init__(self):
        self.envelopes = []

    async def dispatch(self, envelope):
        self.envelopes.append(envelope)


async def ingest(payloads):
    dispatcher = RecordingDispatcher()
    worker = MemoryUpdateWorker(dispatcher, bootstrap_servers="unused")
    for p in payloads:
        await worker._process_turn(p)
    return [{**e["payload"], "event_id": e["event_id"]} for e in dispatcher.envelopes]


def legacy(i, ts=None):
    return {"conversation_id": "c1", "user_id": "u1", "timestamp": (ts or T0 + timedelta(seconds=10 * i)).isoformat(),
            "user_message": f"q{i}", "assistant_response": f"a{i}"}


def new_shape(i, user_ts, assistant_ts, **over):
    return {**legacy(i, assistant_ts), "user_message_id": uuid7_at(user_ts, 11 + i),
            "assistant_message_id": uuid7_at(user_ts, 99 + i), "user_created_at": user_ts.isoformat(),
            "assistant_created_at": assistant_ts.isoformat(), **over}


# ─── ordering ────────────────────────────────────────────────────────────────────────────────────────────────────────


async def test_legacy_events_are_stored_user_then_assistant_strictly():
    rows = await ingest([legacy(i) for i in range(300)])
    for user, assistant in zip(rows[0::2], rows[1::2]):
        assert user["role"] == "user" and assistant["role"] == "assistant"
        assert message_key(user) < message_key(assistant)
        assert user["created_at"] != assistant["created_at"]
    assert [r["role"] for r in sorted(rows, key=message_key)] == ["user", "assistant"] * 300


async def test_legacy_ids_and_event_ids_are_unchanged_so_redelivery_stays_idempotent():
    rows = await ingest([legacy(0), legacy(0)])
    assert rows[0] == rows[2] and rows[1] == rows[3]                 # same ids, event ids and timestamps on redelivery
    ts = legacy(0)["timestamp"]
    assert rows[0]["message_id"] == _derive_id(f"c1:{ts}", "user", "msg")
    assert rows[1]["message_id"] == _derive_id(f"c1:{ts}", "assistant", "msg")
    assert rows[0]["event_id"] == _derive_id(f"c1:{ts}", "user")


async def test_new_events_use_the_real_ids_and_timestamps():
    user_ts, assistant_ts = T0, T0 + timedelta(seconds=7)
    payload = new_shape(0, user_ts, assistant_ts)
    user, assistant = await ingest([payload])
    assert user["message_id"] == payload["user_message_id"] and assistant["message_id"] == payload["assistant_message_id"]
    assert datetime.fromisoformat(user["created_at"]) == user_ts
    assert datetime.fromisoformat(assistant["created_at"]) == assistant_ts


@pytest.mark.parametrize("assistant_offset", [timedelta(0), timedelta(milliseconds=-5), timedelta(seconds=-3)])
async def test_new_events_with_equal_or_inverted_times_still_sort_user_first(assistant_offset):
    payload = new_shape(0, T0, T0 + assistant_offset)
    user, assistant = await ingest([payload])
    assert message_key(user) < message_key(assistant)
    assert datetime.fromisoformat(assistant["created_at"]) - datetime.fromisoformat(user["created_at"]) >= LEGACY_ASSISTANT_OFFSET


async def test_new_events_without_timestamps_derive_them_from_the_uuid7_ids():
    payload = {**legacy(0), "user_message_id": uuid7_at(T0, 5), "assistant_message_id": uuid7_at(T0, 1)}   # same ms
    user, assistant = await ingest([payload])
    assert datetime.fromisoformat(user["created_at"]) == uuid7_datetime(payload["user_message_id"])
    assert message_key(user) < message_key(assistant)


async def test_consecutive_new_turns_interleave_correctly():
    payloads = [new_shape(i, T0 + timedelta(seconds=20 * i), T0 + timedelta(seconds=20 * i + 8)) for i in range(50)]
    rows = await ingest(payloads)
    assert [r["role"] for r in sorted(rows, key=message_key)] == ["user", "assistant"] * 50


def test_turn_identity_mixes_new_and_old_fields_per_message():
    ident = turn_identity({**legacy(0), "user_message_id": uuid7_at(T0, 3)})     # only the user id is known
    assert ident["user_message_id"] == str(uuid.UUID(uuid7_at(T0, 3)))
    assert ident["assistant_message_id"] == _derive_id(f"c1:{legacy(0)['timestamp']}", "assistant", "msg")
    assert ident["user_created_at"] < ident["assistant_created_at"]


# ─── watermark / summarizer input stay consistent with the ordering ────────────────────────────────────────────────────


@pytest.mark.parametrize("make", [lambda i: legacy(i),
                                  lambda i: new_shape(i, T0 + timedelta(seconds=20 * i), T0 + timedelta(seconds=20 * i + 8))])
async def test_watermark_through_an_assistant_leaves_exactly_the_next_turn_as_new(make):
    rows = await ingest([make(i) for i in range(5)])
    through = rows[5]                                            # assistant of turn 2 (0-based)
    assert through["role"] == "assistant"
    new, gap = select_new_messages(rows, through["message_id"], datetime.fromisoformat(through["created_at"]))
    assert [m["content"] for m in new] == ["q3", "a3", "q4", "a4"] and not gap


# ─── summary_valid_at_branch ───────────────────────────────────────────────────────────────────────────────────────────


def test_uuid7_both_sides_compare_by_time():
    upto = uuid7_at(T0 + timedelta(seconds=30), 777)
    assert summary_valid_at_branch(uuid7_at(T0, 123456), upto)                                   # earlier: valid
    assert summary_valid_at_branch(upto, upto)                                                   # exactly the branch msg
    assert not summary_valid_at_branch(uuid7_at(T0 + timedelta(seconds=31), 1), upto)            # past the branch point


def test_uuid7_inside_the_same_millisecond_is_not_provable():
    a, b = uuid7_at(T0, 1), uuid7_at(T0, 2)
    assert not summary_valid_at_branch(a, b) and not summary_valid_at_branch(b, a)


def test_legacy_watermark_uses_created_at_against_the_time_in_the_branch_id():
    upto = uuid7_at(T0 + timedelta(seconds=30), 5)
    legacy_id = _derive_id("x", "assistant", "msg")
    assert summary_valid_at_branch(legacy_id, upto, T0)                                          # summarized before
    assert not summary_valid_at_branch(legacy_id, upto, T0 + timedelta(seconds=31))              # covers later messages
    assert not summary_valid_at_branch(legacy_id, upto, T0 + timedelta(seconds=30))              # same ms: unprovable
    assert not summary_valid_at_branch(legacy_id, upto, None)                                    # no timestamp: fail closed
    assert summary_valid_at_branch(legacy_id, upto, (T0).isoformat())                            # ISO strings are accepted


@pytest.mark.parametrize("through,upto", [(None, "x"), ("x", None), ("garbage", "garbage"),
                                          (uuid7_at(T0), str(uuid.uuid5(uuid.NAMESPACE_DNS, "a")))])
def test_everything_unprovable_fails_closed(through, upto):
    assert not summary_valid_at_branch(through, upto, T0)


def test_is_uuid7_rejects_uuid5_and_garbage():
    assert is_uuid7(uuid7_at(T0)) and not is_uuid7(_derive_id("a")) and not is_uuid7("nope") and not is_uuid7(None)


class Repo:
    def __init__(self):
        self.snapshots, self.summaries = {}, {}
        self.cassandra_repo = SimpleNamespace(get_summary=lambda cid: self.summaries.get(cid))

    async def get_snapshot(self, cid):
        return self.snapshots.get(cid)


async def ancestor(through_id, through_ts, upto_id):
    repo = Repo()
    repo.snapshots["A"] = {"conversation_id": "A", "user_id": "u1"}
    repo.summaries["A"] = {"summary_text": "parent summary", "through_message_id": through_id,
                           "through_created_at": through_ts, "summary_version": 1}
    builder = ContextBuilder(SimpleNamespace(memory_repo=repo), graph_client=None)
    return await builder.ancestor_summaries("u1", [{"conversation_id": "A", "upto_message_id": upto_id}])


async def test_new_style_parent_summary_valid_at_branch_is_accepted_and_one_past_it_is_rejected():
    branch = uuid7_at(T0 + timedelta(seconds=30), 9)
    accepted = await ancestor(uuid7_at(T0 + timedelta(seconds=20), 4), T0 + timedelta(seconds=20), branch)
    rejected = await ancestor(uuid7_at(T0 + timedelta(seconds=45), 4), T0 + timedelta(seconds=45), branch)
    assert len(accepted) == 1 and rejected == []


async def test_payload_as_serialized_by_llm_service_json_z_timestamps_and_nulls():
    """llm-service dumps datetimes as ISO strings ending in Z and unset optional fields as null."""
    payload = {
        "event_type": "memory.update.requested", "conversation_id": "c1", "user_id": "u1", "mode": "default",
        "user_message": "q", "assistant_response": "a", "timestamp": "2026-10-01T09:00:08.500000Z",
        "user_message_id": uuid7_at(T0, 1), "assistant_message_id": None,
        "user_created_at": "2026-10-01T09:00:00.123000Z", "assistant_created_at": "2026-10-01T09:00:08.500000Z",
    }
    user, assistant = await ingest([payload])
    assert user["message_id"] == str(uuid.UUID(payload["user_message_id"]))
    assert assistant["message_id"] == _derive_id(f"c1:{payload['timestamp']}", "assistant", "msg")   # null id: uuid5 kept
    assert datetime.fromisoformat(user["created_at"]) == datetime(2026, 10, 1, 9, 0, 0, 123000, tzinfo=timezone.utc)
    assert message_key(user) < message_key(assistant)
