"""Incremental summary correctness: watermark selection, conditional (compare-and-set) commits,
retries/idempotency and output validation. Cassandra and the model are in-memory fakes."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.services.summary_service import (
    BATCH_SIZE,
    ConcurrentSummaryUpdate,
    SummaryOutputError,
    SummaryService,
    select_new_messages,
    validate_summary,
)

pytestmark = pytest.mark.asyncio(loop_scope="function")

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def msg(i, role="user", content=None):
    return {"message_id": f"m{i:04d}", "role": role, "content": content or f"message {i}",
            "created_at": T0 + timedelta(seconds=i)}


class FakeCassandra:
    def __init__(self, messages=()):
        self.messages = list(messages)
        self.row = None
        self.before_write = None          # hook: simulate another worker committing first
        self.writes = []

    def get_summary(self, cid):
        return dict(self.row) if self.row else None

    def get_recent_messages(self, cid, limit=20):
        return sorted(self.messages, key=lambda m: (m["created_at"], m["message_id"]), reverse=True)[:limit]

    def write_summary_if_current(self, summary, expected_version):
        if self.before_write:
            hook, self.before_write = self.before_write, None
            hook(self)
        current = self.row["summary_version"] if self.row else None
        if current != expected_version:
            return False
        self.row = dict(summary)
        self.writes.append(summary["summary_version"])
        return True


class FakeMemoryRepo:
    def __init__(self):
        self.snapshot = {"conversation_id": "c1", "summary_version": 0, "last_summary_msg_id": None}
        self.invalidated = 0

    async def get_snapshot(self, cid):
        return dict(self.snapshot)

    async def save_snapshot(self, snap):
        self.snapshot = dict(snap)

    async def invalidate_conversation(self, cid):
        self.invalidated += 1


def make(messages=(), llm=None):
    cass, repo = FakeCassandra(messages), FakeMemoryRepo()
    calls = []

    async def summarize(request):
        calls.append(request)
        if llm:
            return await llm(request)
        return SimpleNamespace(summary=f"summary of {len(request.new_messages)} new; prev={bool(request.previous_summary)}")

    service = SummaryService(repo, cass, SimpleNamespace(summarize=summarize))
    return service, cass, repo, calls


# ------------------------------------------------------------------ watermark selection (pure)


def test_first_summary_takes_everything_in_chronological_order():
    new, gap = select_new_messages([msg(3), msg(1), msg(2)], None, None)
    assert [m["message_id"] for m in new] == ["m0001", "m0002", "m0003"] and gap is False


def test_only_messages_after_the_watermark_are_new_even_when_the_window_overlaps():
    window = [msg(i) for i in range(1, 8)]
    new, _ = select_new_messages(window, "m0005", T0 + timedelta(seconds=5))
    assert [m["message_id"] for m in new] == ["m0006", "m0007"]


def test_equal_timestamps_are_ordered_by_message_id_not_input_order():
    same = T0
    a, b = {**msg(1), "created_at": same, "message_id": "a"}, {**msg(1), "created_at": same, "message_id": "b"}
    new, _ = select_new_messages([b, a], "a", same)
    assert [m["message_id"] for m in new] == ["b"]


def test_legacy_watermark_without_timestamp_is_located_by_id():
    window = [msg(i) for i in range(1, 6)]
    assert [m["message_id"] for m in select_new_messages(window, "m0003", None)[0]] == ["m0004", "m0005"]
    new, gap = select_new_messages(window, "m9999", None)           # not retained: everything, flagged
    assert len(new) == 5 and gap is True


# ------------------------------------------------------------------ service behaviour


async def test_new_messages_produce_exactly_one_version_and_redelivery_is_a_no_op():
    service, cass, repo, calls = make([msg(1), msg(2), msg(3)])
    await service.process_incremental_summary("c1")
    assert cass.writes == [1] and cass.row["through_message_id"] == "m0003"
    assert repo.snapshot["summary_version"] == 1 and repo.snapshot["last_summary_msg_id"] == "m0003"
    await service.process_incremental_summary("c1")                 # same event delivered again
    assert cass.writes == [1] and len(calls) == 1                   # no second version, no second model call


async def test_overlapping_window_is_not_double_counted():
    service, cass, _, calls = make([msg(i) for i in range(1, 4)])
    await service.process_incremental_summary("c1")
    cass.messages.append(msg(4))
    await service.process_incremental_summary("c1")
    assert [m.content for m in calls[1].new_messages] == ["message 4"]   # m1..m3 are already in the summary
    assert calls[1].previous_summary and cass.writes == [1, 2]


async def test_no_new_messages_creates_no_version():
    service, cass, repo, calls = make([])
    await service.process_incremental_summary("c1")
    assert calls == [] and cass.row is None and repo.invalidated == 0


async def test_a_stale_job_cannot_overwrite_a_newer_summary_or_move_the_watermark_back():
    service, cass, _, calls = make([msg(1), msg(2), msg(3)])

    def other_worker_commits_first(c):
        c.messages.extend([msg(4), msg(5)])
        c.row = {"conversation_id": "c1", "summary_text": "newer", "summary_version": 1,
                 "through_message_id": "m0005", "through_created_at": T0 + timedelta(seconds=5)}

    cass.before_write = other_worker_commits_first
    await service.process_incremental_summary("c1")                  # loses the race, re-reads, finds nothing new
    assert cass.row["summary_text"] == "newer" and cass.row["through_message_id"] == "m0005"
    assert cass.writes == [] and len(calls) == 1


async def test_a_lost_race_recomputes_what_is_new_from_the_committed_summary():
    service, cass, _, calls = make([msg(1), msg(2)])

    def other_worker(c):
        c.row = {"conversation_id": "c1", "summary_text": "w1", "summary_version": 1,
                 "through_message_id": "m0002", "through_created_at": T0 + timedelta(seconds=2)}
        c.messages.append(msg(3))

    cass.before_write = other_worker
    await service.process_incremental_summary("c1")
    assert [m.content for m in calls[-1].new_messages] == ["message 3"] and calls[-1].previous_summary == "w1"
    assert cass.row["summary_version"] == 2 and cass.row["through_message_id"] == "m0003"


async def test_endless_contention_fails_visibly_instead_of_dropping_messages():
    service, cass, _, _ = make([msg(1)])
    cass.write_summary_if_current = lambda summary, expected: False
    with pytest.raises(ConcurrentSummaryUpdate):
        await service.process_incremental_summary("c1")


async def test_model_failure_leaves_the_previous_summary_intact():
    async def boom(_):
        raise RuntimeError("model down")

    service, cass, repo, _ = make([msg(1)], llm=boom)
    cass.row = {"conversation_id": "c1", "summary_text": "old", "summary_version": 3,
                "through_message_id": "m0000", "through_created_at": T0 - timedelta(seconds=1)}
    with pytest.raises(RuntimeError):
        await service.process_incremental_summary("c1")
    assert cass.row["summary_text"] == "old" and cass.row["summary_version"] == 3 and cass.writes == []
    assert repo.snapshot["summary_version"] == 0


@pytest.mark.parametrize("bad", ["", "   \n", None])
async def test_empty_model_output_is_rejected_and_nothing_is_written(bad):
    async def respond(_):
        return SimpleNamespace(summary=bad)

    service, cass, _, _ = make([msg(1)], llm=respond)
    with pytest.raises(SummaryOutputError):
        await service.process_incremental_summary("c1")
    assert cass.row is None and cass.writes == []


async def test_large_backlogs_are_summarized_in_batches_and_the_watermark_only_moves_forward():
    total = BATCH_SIZE * 2 + 5
    service, cass, _, calls = make([msg(i) for i in range(1, total + 1)])
    await service.process_incremental_summary("c1")
    assert [len(c.new_messages) for c in calls] == [BATCH_SIZE, BATCH_SIZE, 5]
    assert cass.writes == [1, 2, 3] and cass.row["through_message_id"] == f"m{total:04d}"


def test_validate_summary_strips_and_accepts_normal_text():
    assert validate_summary("  The user wants X.  ") == "The user wants X."
