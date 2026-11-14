"""P8c: user.deleted removes the user's facts (active and retired), vectors, settings, tombstones and idle compaction, purges
the conversations it can attribute to them through the existing path, and is idempotent / deduped / retried on failure."""

import uuid
from types import SimpleNamespace

import pytest

from app.services.user_deletion import IDLE_KEY, UserDeletionService
from app.workers.billing_events_worker import BillingEventsWorker

pytestmark = pytest.mark.asyncio


class FakeCassandra:
    def __init__(self):
        self.rows = {}                       # (user, category, fact_id) -> row
        self.controls = {}                   # user -> True
        self.deleted_controls = []

    def add(self, user, category, conversation_id, status="ACTIVE"):
        fid = uuid.uuid4()
        self.rows[(user, category, fid)] = {"user_id": user, "category": category, "fact_id": fid,
                                            "conversation_id": conversation_id, "status": status}
        return fid

    def get_facts(self, user_id, category, include_inactive=False):
        rows = [r for (u, c, _), r in self.rows.items() if u == user_id and c == category]
        return rows if include_inactive else [r for r in rows if r["status"] == "ACTIVE"]

    def delete_fact(self, user_id, category, fact_id):
        self.rows.pop((user_id, category, fact_id), None)

    def delete_user_memory_controls(self, user_id):
        self.deleted_controls.append(user_id)


class FakeMilvus:
    def __init__(self):
        self.users = {"u1": 3, "u2": 2}
        self.fail = False

    def delete_user_facts(self, user_id):
        if self.fail:
            raise RuntimeError("milvus down")
        self.users.pop(user_id, None)


class FakeRedis:
    def __init__(self):
        self.idle = {}

    async def zscan(self, key, cursor=0, match=None, count=None):
        suffix = match.lstrip("*")
        return 0, [(m, 1.0) for m in self.idle if m.endswith(suffix)]

    async def zrem(self, key, member):
        return 1 if self.idle.pop(member, None) is not None else 0


def build():
    cass, milvus, redis, purged = FakeCassandra(), FakeMilvus(), FakeRedis(), []

    async def purge(payload):
        purged.append(payload)
        return "purged"

    return UserDeletionService(cass, milvus, redis, purge_conversation=purge), cass, milvus, redis, purged


async def test_all_of_the_users_memory_data_is_removed_and_other_users_are_untouched():
    svc, cass, milvus, redis, purged = build()
    cass.add("u1", "preferences", "c-a")
    cass.add("u1", "goals", "c-b")
    cass.add("u1", "interests", "c-a", status="SUPERSEDED")                 # retired history goes too
    cass.add("u2", "preferences", "c-x")
    redis.idle.update({"c-idle|u1": 1.0, "c-other|u2": 1.0})
    stats = await svc.delete_user("u1")
    assert stats == {"facts": 3, "conversations": 3, "idle_removed": 1}
    assert not [k for k in cass.rows if k[0] == "u1"] and len([k for k in cass.rows if k[0] == "u2"]) == 1
    assert "u1" not in milvus.users and milvus.users["u2"] == 2
    assert cass.deleted_controls == ["u1"]
    assert list(redis.idle) == ["c-other|u2"]                               # u1's scheduled compaction is cancelled
    assert sorted(p["conversation_id"] for p in purged) == ["c-a", "c-b", "c-idle"]
    assert all(p["user_id"] == "u1" for p in purged)                        # the existing conversation-delete path, owner-scoped


async def test_replay_is_idempotent():
    svc, cass, milvus, redis, purged = build()
    cass.add("u1", "preferences", "c-a")
    await svc.delete_user("u1")
    again = await svc.delete_user("u1")
    assert again == {"facts": 0, "conversations": 0, "idle_removed": 0}


async def test_a_failing_vector_delete_raises_so_the_event_is_retried_and_no_row_is_lost_first():
    svc, cass, milvus, redis, purged = build()
    cass.add("u1", "preferences", "c-a")
    milvus.fail = True
    with pytest.raises(RuntimeError):
        await svc.delete_user("u1")
    assert len([k for k in cass.rows if k[0] == "u1"]) == 1                 # Cassandra rows kept: the retry finds them
    milvus.fail = False
    assert (await svc.delete_user("u1"))["facts"] == 1


class FakeProcessed:
    def __init__(self):
        self.seen = set()

    def is_event_processed(self, event_id):
        return event_id in self.seen

    def register_event(self, event_id, conversation_id):
        self.seen.add(event_id)


class FakeEntitlements:
    def __init__(self):
        self.invalidated = []

    async def invalidate(self, user_id):
        self.invalidated.append(user_id)


async def test_the_consumer_dedupes_drops_the_cache_and_retries_a_failed_delete():
    svc, cass, milvus, redis, purged = build()
    cass.add("u1", "preferences", "c-a")
    ent, processed = FakeEntitlements(), FakeProcessed()
    worker = BillingEventsWorker(ent, processed, user_deleter=svc)
    event = {"event_id": str(uuid.uuid4()), "user_id": "u1", "event_type": "user.deleted"}
    milvus.fail = True
    with pytest.raises(RuntimeError):
        await worker.handle("user.deleted", event)
    assert event["event_id"] not in processed.seen                          # not recorded: Kafka redelivers it
    milvus.fail = False
    assert await worker.handle("user.deleted", event) == "user_deleted"
    assert ent.invalidated.count("u1") == 2 and event["event_id"] in processed.seen
    assert await worker.handle("user.deleted", event) == "duplicate"        # a replay does nothing
    assert purged and len(purged) == 1
