"""When a summary (compaction) is requested: the original every-N-messages rule is kept, and context PRESSURE (long messages)
triggers it earlier; small conversations do not; duplicates are not created. Redis, Cassandra and the clock are in-memory."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.core.config import settings
from app.events.dispatcher import EventDispatcher
from app.services.compaction_scheduler import CompactionScheduler
from app.utils.token_counter import TokenCounter
from tests.support.fakes import FakeCassandra, FakeRedis

pytestmark = pytest.mark.asyncio
CONV, USER = "conv-1", "user-1"
LONG = " ".join(["context"] * 900)            # ~900 tokens


def make(threshold: int = 20):
    redis, cass = FakeRedis(), FakeCassandra()
    scheduler = CompactionScheduler(redis, cass, TokenCounter())
    snapshot = {"conversation_id": CONV, "user_id": USER, "message_count": 0, "state": "ACTIVE", "summary_version": 0,
                "fact_version": 0, "snapshot_version": 0, "last_summary_msg_id": None}
    state = {"snapshot": snapshot, "commits": []}

    memory_repo = MagicMock()
    memory_repo.get_snapshot = AsyncMock(side_effect=lambda cid: dict(state["snapshot"]))
    processed = MagicMock()
    processed.is_event_processed.return_value = False
    snapshots = MagicMock()

    def commit(snapshot, event_id, outbox_topic=None, outbox_payload=None, message=None):
        state["snapshot"] = dict(snapshot)
        state["commits"].append({"topic": outbox_topic, "payload": outbox_payload, "count": snapshot["message_count"]})

    snapshots.commit_snapshot_and_outbox.side_effect = commit
    snapshots.post_commit_invalidation = AsyncMock()
    dispatcher = EventDispatcher(processed, memory_repo, snapshots, summary_threshold=threshold, scheduler=scheduler)
    return dispatcher, scheduler, state, redis, cass


async def send(dispatcher, n: int, content: str = "short message"):
    await dispatcher.dispatch({
        "event_id": f"evt-{n}", "event_type": "chat.message.created", "conversation_id": CONV, "user_id": USER,
        "payload": {"message_id": f"m{n:04d}", "role": "user", "content": content}})


def requested(state):
    return [c["count"] for c in state["commits"] if c["topic"]]


async def test_a_small_conversation_never_triggers_compaction_before_the_threshold():
    dispatcher, _, state, *_ = make()
    for n in range(1, 20):
        await send(dispatcher, n)
    assert requested(state) == []


async def test_the_original_every_n_messages_rule_still_fires():
    dispatcher, _, state, *_ = make()
    for n in range(1, 41):
        await send(dispatcher, n)
    assert requested(state) == [20, 40]


async def test_long_messages_trigger_compaction_long_before_twenty_messages():
    dispatcher, _, state, *_ = make()
    for n in range(1, 11):
        await send(dispatcher, n, LONG)
    fired = requested(state)
    assert fired and fired[0] < 20                               # pressure, not count
    assert fired[0] >= settings.COMPACTION_MIN_NEW_MESSAGES      # but never after just one big message


async def test_one_huge_message_alone_does_not_trigger_a_summary_before_the_minimum_message_count():
    dispatcher, _, state, *_ = make()
    await send(dispatcher, 1, " ".join(["context"] * 20000))
    assert requested(state) == []


async def test_only_one_request_is_in_flight_while_pressure_stays_high():
    dispatcher, _, state, *_ = make()
    for n in range(1, 19):
        await send(dispatcher, n, LONG)
    assert len(requested(state)) == 1                           # the in-flight guard stops one request per message


async def test_pressure_restarts_after_the_summary_is_committed():
    dispatcher, scheduler, state, redis, _ = make()
    for n in range(1, 11):
        await send(dispatcher, n, LONG)
    assert len(requested(state)) == 1
    await scheduler.reset(CONV)                                  # what SummaryService does after a commit
    for n in range(11, 22):
        await send(dispatcher, n, LONG)
    assert len(requested(state)) >= 2


async def test_without_the_scheduler_behaviour_is_exactly_the_original_rule():
    dispatcher, _, state, *_ = make()
    dispatcher.scheduler = None
    for n in range(1, 11):
        await send(dispatcher, n, LONG)
    assert requested(state) == []


async def test_lost_redis_counters_fall_back_to_the_count_rule():
    dispatcher, scheduler, state, redis, _ = make()

    async def broken(*a, **k):
        raise ConnectionError("redis down")

    redis.incrby = broken
    for n in range(1, 21):
        await send(dispatcher, n, LONG)
    assert requested(state) == [20]


async def test_the_stale_read_trigger_creates_one_outbox_job_and_deduplicates():
    _, scheduler, _, _, cass = make()
    assert await scheduler.request_from_read(CONV, USER, 3) is True
    assert await scheduler.request_from_read(CONV, USER, 3) is False
    assert len(cass.outbox) == 1 and cass.outbox[0]["topic"] == settings.KAFKA_SUMMARY_TOPIC
