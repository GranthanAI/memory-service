"""The branch-summary and conversation-delete consumers use the same retry -> FAILED (dead-letter) path as the ingestion worker.
A lost delete is a privacy bug, so a failure is never dropped and an unrecordable failure is never committed past."""
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.core.config import settings
from app.events.retry_scheduler import RetryScheduler
from app.workers.branch_summary_worker import JOB_TYPE as BRANCH_JOB, BranchSummaryWorker
from app.workers.conversation_delete_worker import JOB_TYPE as DELETE_JOB, ConversationDeleteWorker
from app.workers.retry_support import MAX_ATTEMPTS, schedule_retry_or_park

pytestmark = pytest.mark.asyncio


class Repo:
    def __init__(self, fail_insert=False):
        self.jobs, self.fail_insert = [], fail_insert

    def insert_retry_job(self, job):
        if self.fail_insert:
            raise RuntimeError("cassandra down")
        self.jobs.append(job)

    def delete_conversation_data(self, cid):
        raise RuntimeError("cassandra timeout")


def envelope(**over):
    return {"event_type": "conversation.deleted", "payload": {"conversation_id": "c1", "user_id": "u1", **over}}


def kmsg(obj, offset=7):
    raw = obj if isinstance(obj, bytes) else json.dumps(obj).encode()
    return SimpleNamespace(value=raw, offset=offset)


def delete_worker(repo):
    memory_repo = SimpleNamespace(get_snapshot=AsyncMock(return_value={"user_id": "u1"}), invalidate_conversation=AsyncMock())
    w = ConversationDeleteWorker(memory_repo, repo, MagicMock())
    w._consumer = SimpleNamespace(seek=MagicMock())
    return w


def branch_worker(repo, fail=True):
    memory_repo = SimpleNamespace(get_snapshot=AsyncMock(side_effect=RuntimeError("cassandra down") if fail else None))
    w = BranchSummaryWorker(memory_repo, repo, scheduler=None, bootstrap_servers="x")
    w._consumer = SimpleNamespace(seek=MagicMock())
    return w


BRANCH_EVENT = {"event_type": "conversation.created",
                "payload": {"conversation_id": "child", "user_id": "u1", "parent_conversation_id": "p", "branch_message_id": "m"}}


async def test_a_failed_conversation_delete_is_scheduled_for_retry_with_the_original_event():
    repo = Repo()
    assert await delete_worker(repo).handle_with_retry("tp", kmsg(envelope())) is True      # safe to commit: a retry is durable
    (job,) = repo.jobs
    assert job["status"] == "PENDING" and job["job_type"] == DELETE_JOB == "conversation_delete" and job["retry_count"] == 1
    retried = json.loads(job["payload"])
    assert retried["attempt_count"] == 1 and retried["payload"]["conversation_id"] == "c1"   # the ORIGINAL envelope comes back


async def test_a_failed_branch_check_is_scheduled_for_retry():
    repo = Repo()
    assert await branch_worker(repo).handle_with_retry("tp", kmsg(BRANCH_EVENT)) is True
    (job,) = repo.jobs
    assert job["status"] == "PENDING" and job["job_type"] == BRANCH_JOB == "branch_summary"
    assert json.loads(job["payload"])["payload"]["parent_conversation_id"] == "p"


@pytest.mark.parametrize("make", [lambda r: delete_worker(r), lambda r: branch_worker(r)])
async def test_retries_stop_at_the_limit_and_leave_a_failed_dead_letter_record(make):
    repo = Repo()
    event = envelope(parent_conversation_id="p", branch_message_id="m")
    event["attempt_count"] = MAX_ATTEMPTS - 1
    assert await make(repo).handle_with_retry("tp", kmsg(event)) is True
    (job,) = repo.jobs
    assert job["status"] == "FAILED" and "Dead-lettered" in job["last_error"]


@pytest.mark.parametrize("make", [lambda r: delete_worker(r), lambda r: branch_worker(r)])
async def test_undecodable_events_are_dead_lettered_not_retried_forever(make):
    repo = Repo()
    assert await make(repo).handle_with_retry("tp", kmsg(b"\xff not json")) is True
    (job,) = repo.jobs
    assert job["status"] == "FAILED" and job["payload"]


@pytest.mark.parametrize("make", [lambda r: delete_worker(r), lambda r: branch_worker(r)])
async def test_if_the_failure_cannot_be_recorded_the_offset_is_not_committed_and_the_consumer_rewinds(make, monkeypatch):
    monkeypatch.setattr("asyncio.sleep", AsyncMock())
    worker = make(Repo(fail_insert=True))
    assert await worker.handle_with_retry("tp", kmsg(envelope(parent_conversation_id="p", branch_message_id="m"), offset=42)) is False
    worker._consumer.seek.assert_called_once_with("tp", 42)


async def test_a_successful_event_records_nothing():
    class Ok(Repo):
        def delete_conversation_data(self, cid):
            pass

        def get_facts(self, user_id, category, include_inactive=False):
            return []

    repo = Ok()
    assert await delete_worker(repo).handle_with_retry("tp", kmsg(envelope())) is True
    assert repo.jobs == []


def test_retry_scheduler_republishes_these_job_types_to_dedicated_retry_topics():
    """Not the shared conversation.* topics: other services consume those and must not see a replay."""
    source = open(RetryScheduler.__module__.replace(".", "/") + ".py", encoding="utf-8").read()
    assert '"branch_summary": settings.KAFKA_BRANCH_SUMMARY_RETRY_TOPIC' in source
    assert '"conversation_delete": settings.KAFKA_CONVERSATION_DELETE_RETRY_TOPIC' in source
    assert settings.KAFKA_BRANCH_SUMMARY_RETRY_TOPIC not in ("conversation.created", "conversation.deleted")
    assert settings.KAFKA_CONVERSATION_DELETE_RETRY_TOPIC not in ("conversation.created", "conversation.deleted")


def test_helper_treats_missing_required_key_as_permanent():
    repo = Repo()
    assert schedule_retry_or_park(repo, "x", {"payload": {"no": "id"}}, b"{}", RuntimeError("boom")) == "dead_lettered"
