"""Phase 2: pipeline integrity - conditional snapshot writes, ingestion retry/DLQ, trigger accounting relative to the
watermark, the in-flight TTL, summary metadata and one-object summary caching. In-memory doubles only."""
import json
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.core.config import SystemSettings, settings
from app.models.memory import MemoryState
from app.repositories.memory_repository import MemoryRepository
from app.repositories.redis_repository import RedisRepository
from app.services.compaction_scheduler import CompactionScheduler
from app.services.context_builder import ContextBuilder
from app.services.memory_service import MemoryService
from app.services.summary_service import SummaryService
from app.utils.token_counter import TokenCounter
from app.workers.memory_update_worker import MAX_INGEST_ATTEMPTS, MemoryUpdateWorker
from tests.support.fakes import FakeCassandra, FakeRedis, ScriptedSummarizer, msg

pytestmark = pytest.mark.asyncio


# ─── snapshot writes ─────────────────────────────────────────────────────────────────────────────────────────────────


class MemoryRedis:
    """RedisRepository surface used by MemoryRepository, backed by dicts (including the summary record)."""

    def __init__(self):
        self.snapshots, self.summary_records, self.invalidations = {}, {}, 0

    async def get_snapshot(self, cid):
        return self.snapshots.get(cid)

    async def set_snapshot(self, snap):
        self.snapshots[snap["conversation_id"]] = dict(snap)

    async def invalidate_conversation(self, cid):
        self.invalidations += 1
        self.snapshots.pop(cid, None)
        self.summary_records.pop(cid, None)

    async def get_summary_record(self, cid):
        return self.summary_records.get(cid)

    async def set_summary_record(self, cid, record):
        self.summary_records[cid] = dict(record)


def snapshot(**over):
    base = {"conversation_id": "c1", "user_id": "u1", "message_count": 9, "state": "ACTIVE", "summary_version": 0,
            "fact_version": 0, "snapshot_version": 9, "last_summary_msg_id": None}
    return {**base, **over}


async def test_state_transition_does_not_roll_back_counters_bumped_concurrently():
    cass, redis = FakeCassandra(), MemoryRedis()
    cass.snapshots["c1"] = snapshot()
    service = MemoryService(MemoryRepository(cass, redis), cass)
    real_get = cass.get_snapshot
    state = {"reads": 0}

    def get_then_dispatcher_ingests(cid):
        row = real_get(cid)
        state["reads"] += 1
        if state["reads"] == 1:                       # the dispatcher commits one more message right after our read
            cass.snapshots[cid]["message_count"] += 1
            cass.snapshots[cid]["snapshot_version"] += 1
        return row

    cass.get_snapshot = get_then_dispatcher_ingests
    result = await service.transition_state("c1", MemoryState.SUMMARY_PENDING)
    stored = cass.snapshots["c1"]
    assert stored["message_count"] == 10                          # the increment survived
    assert stored["state"] == MemoryState.SUMMARY_PENDING.value
    assert stored["snapshot_version"] == 11 and result["snapshot_version"] == 11   # retried on the fresh version


async def test_state_transition_gives_up_loudly_when_the_snapshot_never_settles():
    cass, redis = FakeCassandra(), MemoryRedis()
    cass.snapshots["c1"] = snapshot()
    service = MemoryService(MemoryRepository(cass, redis), cass)
    cass.update_snapshot_state_if_current = lambda *a: False
    with pytest.raises(RuntimeError, match="kept changing"):
        await service.transition_state("c1", MemoryState.SUMMARY_PENDING)


async def test_summary_commit_writes_only_summary_columns():
    cass, redis = FakeCassandra(), MemoryRedis()
    cass.snapshots["c1"] = snapshot()
    cass.messages["c1"] = [msg(2), msg(1)]
    service = SummaryService(MemoryRepository(cass, redis), cass, ScriptedSummarizer("a summary"))
    cass.upsert_snapshot = MagicMock(side_effect=AssertionError("whole-row snapshot upsert"))
    cass.snapshots["c1"]["message_count"] = 12                      # bumped while the model call was in flight
    await service.process_incremental_summary("c1")
    assert cass.snapshots["c1"]["message_count"] == 12
    assert cass.snapshots["c1"]["summary_version"] == 1 and cass.snapshots["c1"]["last_summary_msg_id"] == "m0002"


# ─── ingestion retries / dead letters ────────────────────────────────────────────────────────────────────────────────


class Dispatcher:
    """Idempotent by event id like EventDispatcher; can fail once on the assistant side."""

    def __init__(self, fail_assistant_times=0, always_fail=False):
        self.done, self.stored, self.fail_assistant_times, self.always_fail = set(), [], fail_assistant_times, always_fail

    async def dispatch(self, envelope):
        if envelope["event_id"] in self.done:
            return
        if self.always_fail or (envelope["payload"]["role"] == "assistant" and self.fail_assistant_times > 0):
            self.fail_assistant_times -= 1
            raise RuntimeError("cassandra write timeout")
        self.done.add(envelope["event_id"])
        self.stored.append(envelope["payload"]["role"])


class Repo:
    def __init__(self, fail_insert=False):
        self.jobs, self.fail_insert = [], fail_insert

    def insert_retry_job(self, job):
        if self.fail_insert:
            raise RuntimeError("cassandra down")
        self.jobs.append(job)


def turn(**over):
    return {"conversation_id": "c1", "user_id": "u1", "timestamp": "2026-10-01T09:00:00+00:00",
            "user_message": "q", "assistant_response": "a", **over}


def worker_with(dispatcher, repo):
    w = MemoryUpdateWorker(dispatcher, bootstrap_servers="unused", cassandra_repo=repo)
    w._consumer = SimpleNamespace(seek=MagicMock(), commit=AsyncMock())
    return w


def kafka_msg(payload, offset=5):
    raw = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    return SimpleNamespace(value=raw, offset=offset)


async def test_a_failed_second_dispatch_schedules_a_retry_and_the_retry_completes_the_turn_without_duplicating():
    dispatcher, repo = Dispatcher(fail_assistant_times=1), Repo()
    w = worker_with(dispatcher, repo)
    assert await w._handle_message("tp", kafka_msg(turn())) is True       # safe to commit: a retry job is durable
    assert dispatcher.stored == ["user"]                                  # half a turn so far
    (job,) = repo.jobs
    assert job["status"] == "PENDING" and job["job_type"] == "memory_update" and job["retry_count"] == 1
    retried = json.loads(job["payload"])
    assert retried["attempt_count"] == 1 and retried["user_message"] == "q"      # the ORIGINAL payload comes back

    assert await w._handle_message("tp", kafka_msg(retried)) is True      # RetryScheduler republishes it
    assert dispatcher.stored == ["user", "assistant"]                     # completed; the user side was a duplicate no-op
    assert len(repo.jobs) == 1


async def test_retries_stop_at_the_limit_and_leave_a_failed_dead_letter_record():
    dispatcher, repo = Dispatcher(always_fail=True), Repo()
    w = worker_with(dispatcher, repo)
    payload = turn(attempt_count=MAX_INGEST_ATTEMPTS - 1)
    assert await w._handle_message("tp", kafka_msg(payload)) is True
    (job,) = repo.jobs
    assert job["status"] == "FAILED" and "Dead-lettered" in job["last_error"]
    assert json.loads(job["payload"])["user_message"] == "q"


@pytest.mark.parametrize("raw", [b"\xff\xfe not json", b"[1, 2]", json.dumps({"no": "conversation"}).encode()])
async def test_undecodable_or_foreign_payloads_are_dead_lettered_not_silently_skipped(raw):
    repo = Repo()
    w = worker_with(Dispatcher(), repo)
    assert await w._handle_message("tp", kafka_msg(raw)) is True
    (job,) = repo.jobs
    assert job["status"] == "FAILED" and job["job_type"] == "memory_update"
    assert job["payload"]                                                  # the raw bytes are kept for inspection


async def test_if_the_failure_cannot_even_be_recorded_the_offset_is_not_committed_and_the_consumer_rewinds(monkeypatch):
    monkeypatch.setattr("app.workers.memory_update_worker.asyncio.sleep", AsyncMock())
    dispatcher, repo = Dispatcher(always_fail=True), Repo(fail_insert=True)
    w = worker_with(dispatcher, repo)
    assert await w._handle_message("tp", kafka_msg(turn(), offset=42)) is False
    w._consumer.seek.assert_called_once_with("tp", 42)


async def test_a_successful_turn_stores_user_then_assistant_and_records_nothing():
    dispatcher, repo = Dispatcher(), Repo()
    assert await worker_with(dispatcher, repo)._handle_message("tp", kafka_msg(turn())) is True
    assert dispatcher.stored == ["user", "assistant"] and repo.jobs == []


# ─── trigger accounting ──────────────────────────────────────────────────────────────────────────────────────────────


async def test_scheduler_reset_keeps_messages_that_arrived_during_the_job():
    redis = FakeRedis()
    scheduler = CompactionScheduler(redis, None, TokenCounter())
    for _ in range(25):
        await scheduler.note_message("c1", "hello world")
    await scheduler.claim("c1")
    await scheduler.reset("c1", pending_messages=5, pending_tokens=40)
    assert int(redis.kv["compact:msgs:c1"]) == 5 and int(redis.kv["compact:tokens:c1"]) == 40
    assert "compact:inflight:c1" not in redis.kv                         # guard released
    await scheduler.reset("c1")                                          # nothing pending: counters are cleared
    assert "compact:msgs:c1" not in redis.kv


async def test_summary_commit_resets_pressure_to_what_is_pending_after_the_new_watermark():
    cass = FakeCassandra()
    cass.messages["c1"] = [msg(n) for n in range(50, 0, -1)]            # 50 messages; the batch covers the first 40
    cass.snapshots["c1"] = snapshot()
    scheduler = SimpleNamespace(reset=AsyncMock(), renew=AsyncMock())
    service = SummaryService(MemoryRepository(cass, MemoryRedis()), cass, ScriptedSummarizer("s"), scheduler=scheduler)
    await service.process_incremental_summary("c1")
    # BATCH_SIZE=40 per model call, MAX_BATCHES=5: all 50 are summarized across two batches -> nothing pending...
    pending, tokens = scheduler.reset.await_args.args[1:3]
    assert pending == 0 and tokens == 0
    # ...and messages added after the commit are what the next reset reports
    cass.messages["c1"].insert(0, msg(51))
    cass.messages["c1"].insert(0, msg(52))
    assert service._pending_after_watermark("c1")[0] == 2


async def test_dispatcher_count_rule_is_relative_to_the_watermark_not_the_overall_message_count():
    from app.events.dispatcher import EventDispatcher
    redis = FakeRedis()
    scheduler = CompactionScheduler(redis, FakeCassandra(), TokenCounter())
    state = {"snapshot": snapshot(message_count=100, snapshot_version=100, state="ACTIVE"), "requests": 0}
    memory_repo = MagicMock()
    memory_repo.get_snapshot = AsyncMock(side_effect=lambda cid: dict(state["snapshot"]))
    processed = MagicMock()
    processed.is_event_processed.return_value = False
    snaps = MagicMock()
    snaps.post_commit_invalidation = AsyncMock()

    def commit(snapshot, event_id, outbox_topic=None, **kw):
        state["snapshot"] = dict(snapshot)
        state["requests"] += 1 if outbox_topic else 0

    snaps.commit_snapshot_and_outbox.side_effect = commit
    dispatcher = EventDispatcher(processed, memory_repo, snaps, summary_threshold=20, scheduler=scheduler)
    # The last summary left 15 messages pending (e.g. they arrived during the job); overall count is 100 (not a multiple of 20).
    await scheduler.reset("c1", pending_messages=15, pending_tokens=100)
    for n in range(1, 6):
        await dispatcher.dispatch({"event_id": f"e{n}", "event_type": "chat.message.created", "conversation_id": "c1",
                                   "user_id": "u1", "payload": {"message_id": f"m{n}", "role": "user", "content": "hi"}})
    assert state["requests"] == 1                                       # 15 + 5 = 20 pending since the watermark


# ─── in-flight TTL, summary metadata, cache coherence ────────────────────────────────────────────────────────────────


def test_inflight_ttl_covers_a_worst_case_job():
    cfg = SystemSettings(_env_file=None, LLM_TIMEOUT_SECONDS=60.0, LLM_MAX_RETRIES=3)
    assert cfg.COMPACTION_INFLIGHT_TTL_SECONDS == 120                    # the old, too-short value is only a floor now
    assert cfg.compaction_inflight_ttl > 60 * 4 * 3                      # one batch: retries x (model + refinement + consolidation)
    assert cfg.compaction_inflight_ttl >= 60 * 4 * 3 * 5                 # ... for every batch of a job


async def test_the_inflight_guard_uses_the_derived_ttl_and_is_renewed_per_batch():
    redis = FakeRedis()
    scheduler = CompactionScheduler(redis, None, TokenCounter(), cfg=SystemSettings(_env_file=None))
    await scheduler.claim("c1")
    assert redis.expiry["compact:inflight:c1"] == scheduler.cfg.compaction_inflight_ttl
    cass = FakeCassandra()
    cass.messages["c1"] = [msg(n) for n in range(60, 0, -1)]
    cass.snapshots["c1"] = snapshot()
    renewals = []
    sched = SimpleNamespace(reset=AsyncMock(), renew=AsyncMock(side_effect=lambda cid: renewals.append(cid)))
    service = SummaryService(MemoryRepository(cass, MemoryRedis()), cass, ScriptedSummarizer("s"), scheduler=sched)
    await service.process_incremental_summary("c1")
    assert len(renewals) >= 2                                            # 60 messages = two batches, renewed around each


async def test_summary_row_records_model_prompt_version_and_provider_separately():
    cass = FakeCassandra()
    cass.messages["c1"] = [msg(2), msg(1)]
    cass.snapshots["c1"] = snapshot()
    service = SummaryService(MemoryRepository(cass, MemoryRedis()), cass, ScriptedSummarizer("s"))
    await service.process_incremental_summary("c1")
    row = cass.summaries["c1"]
    assert row["model_name"] == settings.LLM_MODEL
    assert row["model_version"] != settings.LLM_PROVIDER                  # used to hold the provider name by mistake
    assert row["llm_provider"] == settings.LLM_PROVIDER
    assert row["prompt_version"] == settings.SUMMARY_PROMPT_VERSION


async def test_summary_text_version_and_watermark_come_from_one_cache_record():
    cass, redis = FakeCassandra(), MemoryRedis()
    t = datetime(2026, 10, 1, 9, 0, tzinfo=timezone.utc)
    cass.summaries["c1"] = {"conversation_id": "c1", "summary_text": "v2 text", "summary_version": 2,
                            "through_message_id": "m0020", "through_created_at": t}
    repo = MemoryRepository(cass, redis)
    record = await repo.get_summary_record("c1")                          # miss: hydrated from Cassandra and cached whole
    assert record == {"text": "v2 text", "summary_version": 2, "through_message_id": "m0020", "through_created_at": t}
    # Cassandra moves on, the cache is still the v2 record: text and watermark stay a matching pair (never mixed).
    cass.summaries["c1"] = {**cass.summaries["c1"], "summary_text": "v3 text", "summary_version": 3,
                            "through_message_id": "m0030"}
    again = await repo.get_summary_record("c1")
    assert (again["text"], again["summary_version"], again["through_message_id"]) == ("v2 text", 2, "m0020")
    await repo.invalidate_conversation("c1")                               # what a committed summary does
    fresh = await repo.get_summary_record("c1")
    assert (fresh["text"], fresh["summary_version"], fresh["through_message_id"]) == ("v3 text", 3, "m0030")


async def test_context_builder_takes_text_and_watermark_from_the_same_record():
    cass, redis = FakeCassandra(), MemoryRedis()
    t = datetime(2026, 10, 1, 9, 0, tzinfo=timezone.utc)
    redis.summary_records["c1"] = {"text": "cached v1", "summary_version": 1, "through_message_id": "m0001",
                                   "through_created_at": t}
    cass.summaries["c1"] = {"conversation_id": "c1", "summary_text": "newer v5", "summary_version": 5,
                            "through_message_id": "m0099", "through_created_at": t}   # disagrees with the cache
    repo = MemoryRepository(cass, redis)
    retrieval = SimpleNamespace(
        memory_repo=repo,
        get_or_hydrate_summary_record=lambda cid: repo.get_summary_record(cid),
        get_or_hydrate_snapshot=AsyncMock(return_value=None),
        get_or_hydrate_recent_messages=AsyncMock(return_value=[]),
    )
    builder = ContextBuilder(retrieval, graph_client=None)
    text, meta = await builder._load_summary("c1")
    assert text == "cached v1" and meta["summary_version"] == 1 and meta["through_message_id"] == "m0001"


async def test_legacy_plain_text_cache_entries_are_a_miss_for_the_record_not_a_mix():
    """An entry cached before records existed has no watermark: it must not be paired with a Cassandra watermark."""
    redis_client = AsyncMock()
    from app.utils.compression import compress_string
    import base64
    redis_client.get.return_value = base64.b64encode(compress_string("plain old summary")).decode()
    repo = RedisRepository(redis_client)
    assert await repo.get_summary("c1") == "plain old summary"            # text API unchanged
    assert await repo.get_summary_record("c1") is None                    # but not a record -> hydrated from Cassandra


async def test_record_round_trips_through_the_real_redis_repository():
    store = {}
    client = SimpleNamespace(
        get=AsyncMock(side_effect=lambda k: store.get(k)),
        set=AsyncMock(side_effect=lambda k, v, ex=None: store.__setitem__(k, v)),
    )
    repo = RedisRepository(client)
    t = datetime(2026, 10, 1, 9, 0, 0, 123000, tzinfo=timezone.utc)
    await repo.set_summary_record("c1", {"text": "héllo\nworld", "summary_version": 3, "through_message_id": "m9",
                                         "through_created_at": t})
    got = await repo.get_summary_record("c1")
    assert got == {"text": "héllo\nworld", "summary_version": 3, "through_message_id": "m9", "through_created_at": t}
    assert await repo.get_summary("c1") == "héllo\nworld"                 # the text API unwraps the record
