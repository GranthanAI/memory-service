"""
tests/integration/test_outbox_worker_integration.py

Integration tests for Phase 17 Outbox Daemon Worker.
Spawns multiple concurrent worker loops processing a seeded outbox table,
verifying that Cassandra LWT prevents duplicate Kafka publishing.
"""

import asyncio
import json
import uuid
import pytest
import random
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock

from app.db.cassandra import get_session
from app.db.session import initialize_db_sessions, close_db_sessions
from app.repositories.cassandra_repository import CassandraRepository
from app.workers.outbox_worker import OutboxDaemonWorker


def run_async(coro):
    """Helper to run async coroutines in a consistent event loop."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


@pytest.fixture(scope="module", autouse=True)
def setup_integration_db():
    """Initializes and tears down database sessions."""
    run_async(initialize_db_sessions())
    yield
    run_async(close_db_sessions())


@pytest.fixture
def clean_outbox_table():
    """Clears Cassandra outbox_jobs before and after tests."""
    session = get_session()
    # Cassandra delete requires filtering by status, created_at, job_id,
    # but we can truncate or delete rows by querying them first.
    rows = list(session.execute("SELECT created_at, job_id FROM outbox_jobs"))
    for row in rows:
        session.execute(
            "DELETE FROM outbox_jobs WHERE status = 'PENDING' AND created_at = %s AND job_id = %s",
            (row.created_at, row.job_id)
        )
        session.execute(
            "DELETE FROM outbox_jobs WHERE status = 'PROCESSING' AND created_at = %s AND job_id = %s",
            (row.created_at, row.job_id)
        )
    yield
    
    # Post clean
    rows = list(session.execute("SELECT created_at, job_id FROM outbox_jobs"))
    for row in rows:
        session.execute(
            "DELETE FROM outbox_jobs WHERE status = 'PENDING' AND created_at = %s AND job_id = %s",
            (row.created_at, row.job_id)
        )
        session.execute(
            "DELETE FROM outbox_jobs WHERE status = 'PROCESSING' AND created_at = %s AND job_id = %s",
            (row.created_at, row.job_id)
        )


class MockConcurrentProducer:
    """Mock Kafka Producer capturing calls and introducing random latency to simulate racing."""
    def __init__(self):
        self.published_events = []
        self.lock = asyncio.Lock()

    async def publish_task(self, topic: str, conversation_id: str, payload: dict) -> None:
        # Simulate realistic network delay to force concurrent workers to overlap execution
        await asyncio.sleep(random.uniform(0.01, 0.05))
        async with self.lock:
            self.published_events.append({
                "topic": topic,
                "conversation_id": conversation_id,
                "payload": payload
            })


def test_outbox_worker_concurrency_lwt_duplicate_prevention(clean_outbox_table):
    """
    Spawns three concurrent outbox worker daemons to process 20 seeded outbox jobs.
    Asserts LWT prevents duplicate publishing and that the outbox is completely cleared.
    """
    session = get_session()
    producer = MockConcurrentProducer()

    # 1. Seed 20 PENDING outbox jobs in Cassandra
    # To keep them ordered, we use a single conversation_id or multiple distinct ones
    conversation_id = "test-concurrent-conv"
    job_ids = []
    
    # We must insert them with slightly offset created_at values to prevent primary key collision
    base_time = datetime.now(timezone.utc)
    for i in range(20):
        job_id = uuid.uuid4()
        job_ids.append(job_id)
        
        created_at = base_time + timedelta(milliseconds=i)
        payload = json.dumps({"job_index": i, "data": "dummy"})
        
        # Insert statement directly into outbox
        session.execute(
            "INSERT INTO outbox_jobs (status, created_at, job_id, topic, conversation_id, payload, attempt_count) "
            "VALUES ('PENDING', %s, %s, 'test-events', %s, %s, 0)",
            (created_at, job_id, conversation_id, payload)
        )

    # Verify 20 jobs exist
    rows_initial = list(session.execute("SELECT job_id FROM outbox_jobs WHERE status = 'PENDING'"))
    assert len(rows_initial) == 20

    # 2. Instantiate 3 outbox worker daemons
    worker_1 = OutboxDaemonWorker(session, producer)
    worker_2 = OutboxDaemonWorker(session, producer)
    worker_3 = OutboxDaemonWorker(session, producer)

    # Set very small poll interval (10ms) to trigger frequent execution loops
    worker_1._poll_interval = 0.01
    worker_2._poll_interval = 0.01
    worker_3._poll_interval = 0.01

    # 3. Spawn workers in the async event loop concurrently
    async def run_workers():
        # Start all workers
        tasks = [
            asyncio.create_task(worker_1.start()),
            asyncio.create_task(worker_2.start()),
            asyncio.create_task(worker_3.start())
        ]
        
        # Let them run concurrently for 2 seconds to process the 20 items
        await asyncio.sleep(2.0)
        
        # Stop all workers
        await worker_1.stop()
        await worker_2.stop()
        await worker_3.stop()
        
        # Cancel tasks
        for t in tasks:
            t.cancel()
            
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            pass

    run_async(run_workers())

    # 4. Assertions
    # Verify that exactly 20 events were published to the Kafka mock
    assert len(producer.published_events) == 20

    # Verify that there are no duplicate job indexes published
    job_indices = [event["payload"]["job_index"] for event in producer.published_events]
    assert len(set(job_indices)) == 20
    assert sorted(job_indices) == list(range(20))

    # Verify that the outbox_jobs table is now completely empty
    rows_remaining = list(session.execute("SELECT job_id FROM outbox_jobs"))
    assert len(rows_remaining) == 0
