"""
tests/integration/test_cassandra.py

Integration tests for Phase 7: Cassandra Repository Layer.
Uses real Cassandra connections to verify all DML functions and LWT operations.
"""

import asyncio
import uuid
from datetime import datetime, timezone, timedelta
import pytest

from app.db.session import initialize_db_sessions, close_db_sessions
from app.db.cassandra import get_session
from app.repositories.cassandra_repository import CassandraRepository


@pytest.fixture(scope="module", autouse=True)
def setup_integration_db():
    """
    Initializes real database connection sessions on startup and closes them on teardown.
    """
    loop = asyncio.get_event_loop_policy().new_event_loop()
    loop.run_until_complete(initialize_db_sessions())
    yield
    loop.run_until_complete(close_db_sessions())
    loop.close()


@pytest.fixture
def clean_tables():
    """
    Truncates all tables used in tests before running them.
    """
    session = get_session()
    tables = [
        "conversation_snapshots",
        "conversation_summaries",
        "conversation_recent_messages",
        "user_facts",
        "outbox_jobs",
        "outbox_processing_index",
        "retry_jobs"
    ]
    for t in tables:
        session.execute(f"TRUNCATE {t}")
    yield


def test_snapshot_lifecycle(clean_tables):
    """Verifies upsert and retrieval of conversation snapshots."""
    session = get_session()
    repo = CassandraRepository(session)

    conv_id = "conv-111"
    snap = {
        "conversation_id": conv_id,
        "user_id": "user-222",
        "message_count": 5,
        "state": "ACTIVE",
        "summary_version": 1,
        "fact_version": 2,
        "snapshot_version": 3,
        "last_summary_msg_id": "msg-9",
        "updated_at": datetime.now(timezone.utc)
    }

    # 1. Verify None on missing snapshot
    assert repo.get_snapshot(conv_id) is None

    # 2. Save and retrieve
    repo.upsert_snapshot(snap)
    fetched = repo.get_snapshot(conv_id)
    assert fetched is not None
    assert fetched["conversation_id"] == conv_id
    assert fetched["user_id"] == "user-222"
    assert fetched["message_count"] == 5
    assert fetched["state"] == "ACTIVE"


def test_summary_lifecycle(clean_tables):
    """Verifies upsert and retrieval of summaries."""
    session = get_session()
    repo = CassandraRepository(session)

    conv_id = "conv-222"
    summary = {
        "conversation_id": conv_id,
        "summary_text": "Conversation about coding",
        "summary_version": 4,
        "model_name": "gpt-4",
        "model_version": "v1",
        "generated_at": datetime.now(timezone.utc)
    }

    assert repo.get_summary(conv_id) is None

    repo.upsert_summary(summary)
    fetched = repo.get_summary(conv_id)
    assert fetched is not None
    assert fetched["summary_text"] == "Conversation about coding"
    assert fetched["summary_version"] == 4


def test_recent_messages_sliding_window(clean_tables):
    """Verifies that appended messages are correctly retrieved in DESC time-order."""
    session = get_session()
    repo = CassandraRepository(session)

    conv_id = "conv-333"
    base_time = datetime.now(timezone.utc)

    msg1 = {
        "message_id": "msg-1",
        "role": "user",
        "content": "First message",
        "created_at": base_time - timedelta(minutes=2)
    }
    msg2 = {
        "message_id": "msg-2",
        "role": "assistant",
        "content": "Second message",
        "created_at": base_time - timedelta(minutes=1)
    }
    msg3 = {
        "message_id": "msg-3",
        "role": "user",
        "content": "Third message",
        "created_at": base_time
    }

    # Append out of order in test to ensure database query handles DESC clustering sort
    repo.append_recent_message(conv_id, msg2)
    repo.append_recent_message(conv_id, msg1)
    repo.append_recent_message(conv_id, msg3)

    # Fetch last 2 (should return msg3 then msg2 because msg3 is the most recent)
    fetched = repo.get_recent_messages(conv_id, limit=2)
    assert len(fetched) == 2
    assert fetched[0]["message_id"] == "msg-3"
    assert fetched[1]["message_id"] == "msg-2"

    # Delete message row
    repo.delete_recent_message_row(conv_id, msg1["created_at"], msg1["message_id"])
    all_msgs = repo.get_recent_messages(conv_id, limit=10)
    assert len(all_msgs) == 2
    assert not any(m["message_id"] == "msg-1" for m in all_msgs)


def test_user_facts_lifecycle(clean_tables):
    """Verifies upsert, category fetch, and delete of user facts."""
    session = get_session()
    repo = CassandraRepository(session)

    user_id = "user-777"
    category = "preferences"
    fact_id = uuid.uuid4()

    fact = {
        "user_id": user_id,
        "category": category,
        "fact_id": fact_id,
        "conversation_id": "conv-1",
        "statement": "User likes blue color.",
        "importance": 0.8,
        "fact_version": 1,
        "embedding_version": "v1.0",
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc)
    }

    assert len(repo.get_facts(user_id, category)) == 0

    repo.upsert_fact(fact)
    facts = repo.get_facts(user_id, category)
    assert len(facts) == 1
    assert facts[0]["statement"] == "User likes blue color."
    assert pytest.approx(facts[0]["importance"], abs=1e-5) == 0.8

    repo.delete_fact(user_id, category, fact_id)
    assert len(repo.get_facts(user_id, category)) == 0


def test_outbox_claiming_atomic_lwt(clean_tables):
    """
    Verifies outbox job insertions, pending polling, and LWT-driven claim behavior
    (only one claim succeeds, concurrent claims fail).
    """
    session = get_session()
    repo = CassandraRepository(session)

    job_id = uuid.uuid4()
    created_at = datetime.now(timezone.utc)
    conv_id = "conv-out"

    job = {
        "job_id": job_id,
        "topic": "test.topic",
        "conversation_id": conv_id,
        "payload": '{"foo": "bar"}',
        "created_at": created_at
    }

    # 1. Insert and verify it is pending
    repo.insert_outbox_job(job)
    pending = repo.get_pending_outbox_jobs(limit=10)
    assert len(pending) == 1
    assert pending[0]["job_id"] == job_id

    # 2. Claim job (First attempt should succeed)
    success = repo.claim_outbox_job(pending[0])
    assert success is True

    # 3. Second claim attempt on same PENDING descriptor should return False
    # (since the PENDING row has been deleted and replaced with a PROCESSING row)
    success_retry = repo.claim_outbox_job(pending[0])
    assert success_retry is False

    # Check status changed to PROCESSING
    job_processing = repo.get_outbox_job("PROCESSING", created_at, job_id)
    assert job_processing is not None
    assert job_processing["status"] == "PROCESSING"

    # 4. Fail outbox job (moves attempt and keeps PROCESSING)
    repo.fail_outbox_job(job_processing, "network issue")
    failed_job = repo.get_outbox_job("PROCESSING", created_at, job_id)
    assert failed_job["attempt_count"] == 1
    assert failed_job["last_error"] == "network issue"

    # 5. Delete job
    repo.delete_outbox_job("PROCESSING", created_at, job_id)
    assert repo.get_outbox_job("PROCESSING", created_at, job_id) is None


def test_retry_claiming_atomic_lwt(clean_tables):
    """Verifies retry job insertions, scheduling queries, and LWT claiming."""
    session = get_session()
    repo = CassandraRepository(session)

    job_id = uuid.uuid4()
    next_retry = datetime.now(timezone.utc) - timedelta(seconds=1)

    job = {
        "job_id": job_id,
        "next_retry": next_retry,
        "job_type": "summary",
        "payload": "payload",
        "retry_count": 0,
        "max_retry": 5,
        "last_error": None
    }

    repo.insert_retry_job(job)

    # 1. Get due retries
    due_jobs = repo.get_pending_retry_jobs(datetime.now(timezone.utc) + timedelta(seconds=10))
    assert len(due_jobs) == 1
    assert due_jobs[0]["job_id"] == job_id

    # 2. Claim retry (should succeed)
    assert repo.claim_retry_job(due_jobs[0]) is True

    # 3. Duplicate claim should fail (PENDING row has been deleted)
    assert repo.claim_retry_job(due_jobs[0]) is False

    # 4. Cleanup
    repo.delete_retry_job("PROCESSING", next_retry, job_id)
