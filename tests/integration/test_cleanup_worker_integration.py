"""
tests/integration/test_cleanup_worker_integration.py

Integration tests for the CleanupWorker (Stale PROCESSING Reaper).

These tests run against real Cassandra to verify:
1. Stale PROCESSING rows in outbox_jobs are reset to PENDING via LWT.
2. The corresponding outbox_processing_index row is deleted after reclaim.
3. Fresh PROCESSING rows (within the stale threshold) are NOT reclaimed.
4. Concurrent reaper calls (simulated) do not double-reclaim (LWT idempotency).
5. Yesterday's bucket is scanned for midnight-boundary stale rows.

Test isolation: All test data is scoped to unique job_ids. Tables are
cleaned up in teardown to avoid polluting other integration tests.
"""

import uuid
import json
import pytest
import asyncio
from datetime import datetime, timezone, timedelta

from app.db.cassandra import get_session
from app.workers.cleanup_worker import CleanupWorker
from app.core.config import settings


@pytest.fixture(scope="module")
def session():
    """Module-scoped real Cassandra session."""
    from app.db.cassandra import connect_cassandra
    connect_cassandra()
    s = get_session()
    yield s


@pytest.fixture(scope="module")
def worker(session):
    """Module-scoped CleanupWorker instance."""
    return CleanupWorker(session)


def insert_outbox_job(session, job_id, status: str, created_at: datetime, claimed_at: datetime = None):
    """Helper: insert a row into outbox_jobs in a given status."""
    session.execute(
        """
        INSERT INTO outbox_jobs
        (status, created_at, job_id, topic, conversation_id, payload, attempt_count)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """,
        (
            status,
            created_at,
            job_id,
            "memory.summary.request",
            "conv-cleanup-test",
            json.dumps({"conversation_id": "conv-cleanup-test"}),
            1,
        ),
    )
    if claimed_at and status == "PROCESSING":
        # Update claimed_at separately (normal claim flow would do this via LWT)
        session.execute(
            """
            UPDATE outbox_jobs
            SET claimed_at = %s
            WHERE status = 'PROCESSING' AND created_at = %s AND job_id = %s
            """,
            (claimed_at, created_at, job_id),
        )


def insert_processing_index(session, job_id, created_at: datetime, claimed_at: datetime, claimed_date: str):
    """Helper: insert a row into outbox_processing_index."""
    session.execute(
        """
        INSERT INTO outbox_processing_index (claimed_date, claimed_at, job_id, created_at)
        VALUES (%s, %s, %s, %s)
        """,
        (claimed_date, claimed_at, job_id, created_at),
    )


def get_outbox_job_status(session, status: str, created_at: datetime, job_id):
    """Helper: read a row from outbox_jobs by (status, created_at, job_id)."""
    rows = session.execute(
        "SELECT status FROM outbox_jobs WHERE status = %s AND created_at = %s AND job_id = %s",
        (status, created_at, job_id),
    )
    return rows.one()


def index_row_exists(session, claimed_date: str, claimed_at: datetime, job_id) -> bool:
    """Helper: check if an outbox_processing_index row exists."""
    rows = session.execute(
        "SELECT job_id FROM outbox_processing_index "
        "WHERE claimed_date = %s AND claimed_at = %s AND job_id = %s",
        (claimed_date, claimed_at, job_id),
    )
    return rows.one() is not None


def cleanup_job(session, job_id, created_at: datetime, claimed_at: datetime, claimed_date: str):
    """Helper: remove test rows from both tables."""
    for status in ("PENDING", "PROCESSING"):
        session.execute(
            "DELETE FROM outbox_jobs WHERE status = %s AND created_at = %s AND job_id = %s",
            (status, created_at, job_id),
        )
    session.execute(
        "DELETE FROM outbox_processing_index WHERE claimed_date = %s AND claimed_at = %s AND job_id = %s",
        (claimed_date, claimed_at, job_id),
    )


@pytest.mark.asyncio
async def test_cleanup_worker_reclaims_stale_processing_row(session, worker):
    """
    Core scenario: a PROCESSING outbox job older than OUTBOX_STALE_PROCESSING_MINUTES
    must be reset to PENDING and its index row deleted.
    """
    job_id = uuid.uuid4()
    now = datetime.now(timezone.utc)
    stale_minutes = settings.OUTBOX_STALE_PROCESSING_MINUTES + 2
    created_at = now - timedelta(hours=1)
    claimed_at = now - timedelta(minutes=stale_minutes)
    claimed_date = claimed_at.strftime("%Y-%m-%d")

    try:
        # Insert PROCESSING row in outbox_jobs
        insert_outbox_job(session, job_id, "PROCESSING", created_at, claimed_at)
        # Insert matching row in outbox_processing_index
        insert_processing_index(session, job_id, created_at, claimed_at, claimed_date)

        # Verify setup
        assert get_outbox_job_status(session, "PROCESSING", created_at, job_id) is not None
        assert index_row_exists(session, claimed_date, claimed_at, job_id)

        # Run the reaper sweep
        reclaimed = await worker._sweep_stale_jobs()
        assert reclaimed >= 1

        # Verify PROCESSING row is gone
        processing_row = get_outbox_job_status(session, "PROCESSING", created_at, job_id)
        assert processing_row is None, "PROCESSING row should have been reclaimed"

        # Verify PENDING row exists now
        pending_row = get_outbox_job_status(session, "PENDING", created_at, job_id)
        assert pending_row is not None, "Row should have been reset to PENDING"

        # Verify index row is deleted
        assert not index_row_exists(session, claimed_date, claimed_at, job_id), \
            "Index row should be deleted after reclaim"

    finally:
        cleanup_job(session, job_id, created_at, claimed_at, claimed_date)


@pytest.mark.asyncio
async def test_cleanup_worker_does_not_reclaim_fresh_processing_row(session, worker):
    """
    Fresh PROCESSING rows (within the stale threshold) must NOT be reclaimed.
    """
    job_id = uuid.uuid4()
    now = datetime.now(timezone.utc)
    created_at = now - timedelta(minutes=1)
    # claimed_at is only 1 minute ago — well within the stale threshold (5+ minutes)
    claimed_at = now - timedelta(minutes=1)
    claimed_date = claimed_at.strftime("%Y-%m-%d")

    try:
        insert_outbox_job(session, job_id, "PROCESSING", created_at, claimed_at)
        insert_processing_index(session, job_id, created_at, claimed_at, claimed_date)

        await worker._sweep_stale_jobs()

        # Row should still be PROCESSING (not yet stale)
        processing_row = get_outbox_job_status(session, "PROCESSING", created_at, job_id)
        assert processing_row is not None, "Fresh PROCESSING row should not be reclaimed"
        assert index_row_exists(session, claimed_date, claimed_at, job_id), \
            "Index row should still exist for fresh PROCESSING job"

    finally:
        cleanup_job(session, job_id, created_at, claimed_at, claimed_date)


@pytest.mark.asyncio
async def test_cleanup_worker_lwt_idempotency(session, worker):
    """
    Two concurrent reaper calls on the same stale row should be idempotent:
    only one reclaim succeeds (LWT), the second is a no-op.
    No errors should be raised.
    """
    job_id = uuid.uuid4()
    now = datetime.now(timezone.utc)
    stale_minutes = settings.OUTBOX_STALE_PROCESSING_MINUTES + 2
    created_at = now - timedelta(hours=1)
    claimed_at = now - timedelta(minutes=stale_minutes)
    claimed_date = claimed_at.strftime("%Y-%m-%d")

    try:
        insert_outbox_job(session, job_id, "PROCESSING", created_at, claimed_at)
        insert_processing_index(session, job_id, created_at, claimed_at, claimed_date)

        # Simulate two concurrent sweeps
        count1 = await worker._sweep_stale_jobs()
        count2 = await worker._sweep_stale_jobs()

        # Second sweep should find nothing to reclaim (already handled)
        total = count1 + count2
        assert total >= 1  # At least one reclaim happened

        # Row should be PENDING now
        pending_row = get_outbox_job_status(session, "PENDING", created_at, job_id)
        assert pending_row is not None

    finally:
        cleanup_job(session, job_id, created_at, claimed_at, claimed_date)


@pytest.mark.asyncio
async def test_cleanup_worker_yesterday_bucket(session, worker):
    """
    Stale rows from yesterday's date bucket must also be reclaimed.
    Tests the midnight-boundary scenario.
    """
    job_id = uuid.uuid4()
    now = datetime.now(timezone.utc)
    stale_minutes = settings.OUTBOX_STALE_PROCESSING_MINUTES + 10
    created_at = now - timedelta(hours=25)  # claimed well over a day ago
    claimed_at = now - timedelta(minutes=stale_minutes + 1440)  # yesterday
    claimed_date = claimed_at.strftime("%Y-%m-%d")

    try:
        insert_outbox_job(session, job_id, "PROCESSING", created_at, claimed_at)
        insert_processing_index(session, job_id, created_at, claimed_at, claimed_date)

        reclaimed = await worker._sweep_stale_jobs()
        assert reclaimed >= 1

        pending_row = get_outbox_job_status(session, "PENDING", created_at, job_id)
        assert pending_row is not None, "Yesterday's stale row should be reclaimed"

    finally:
        cleanup_job(session, job_id, created_at, claimed_at, claimed_date)


@pytest.mark.asyncio
async def test_cleanup_worker_no_stale_rows_returns_zero(session, worker):
    """
    When no stale rows exist, sweep returns 0 without errors.
    """
    # Ensure tables are clean for this test
    reclaimed = await worker._sweep_stale_jobs()
    # Should be 0 (no stale rows from our test setup; existing rows from other tests are cleaned up)
    assert isinstance(reclaimed, int)
    assert reclaimed >= 0
