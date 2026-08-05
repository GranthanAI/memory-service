"""
tests/integration/test_long_memory_service_integration.py

Integration tests for Phase 14 User Fact Merging Logic (Fact Merge Policy).
Executes merging against live Cassandra and Milvus database containers.
"""

import asyncio
import uuid
import pytest
from datetime import datetime, timezone, timedelta

from app.db.cassandra import get_session
from app.db.session import initialize_db_sessions, close_db_sessions
from app.repositories.cassandra_repository import CassandraRepository
from app.repositories.milvus_repository import MilvusRepository
from app.services.long_memory_service import LongMemoryService
from app.core.config import settings


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
    """Initializes and tears down real database sessions for this test module."""
    run_async(initialize_db_sessions())
    yield
    run_async(close_db_sessions())


@pytest.fixture
def clean_databases():
    """Cleans up Cassandra and Milvus tables for user facts."""
    session = get_session()
    user_id = "test-integration-user-facts"
    category = "preferences"
    
    # 1. Clean Cassandra
    session.execute("DELETE FROM user_facts WHERE user_id = %s AND category = %s", (user_id, category))
    
    # 2. Clean Milvus
    milvus_repo = MilvusRepository()
    milvus_repo.delete_user_facts(user_id)
    
    yield user_id, category
    
    # Post-clean
    session.execute("DELETE FROM user_facts WHERE user_id = %s AND category = %s", (user_id, category))
    milvus_repo.delete_user_facts(user_id)


def test_long_memory_service_fact_merge_policy_integration(clean_databases):
    """
    Asserts exact matching, low-similarity insertions, lower importance ignores,
    and higher importance supersedes against live Cassandra and Milvus containers.
    """
    user_id, category = clean_databases
    cassandra_session = get_session()

    cassandra_repo = CassandraRepository(cassandra_session)
    milvus_repo = MilvusRepository()
    service = LongMemoryService(cassandra_repo, milvus_repo)

    # 1. Seed an initial fact in Cassandra and Milvus
    initial_fact_id = uuid.uuid4()
    
    cassandra_record = {
        "user_id": user_id,
        "category": category,
        "fact_id": initial_fact_id,
        "conversation_id": "conv-1",
        "statement": "User likes drinking green tea.",
        "importance": 0.6,
        "fact_version": 1,
        "embedding_version": "v1.0.0",
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc)
    }
    cassandra_repo.upsert_fact(cassandra_record)

    # Vector representation of "User likes drinking green tea."
    green_tea_vector = [1.0] + [0.0] * (settings.VECTOR_DIMENSION - 1)
    milvus_record = {
        "fact_id": str(initial_fact_id),
        "user_id": user_id,
        "conversation_id": "conv-1",
        "category": category,
        "statement": "User likes drinking green tea.",
        "importance": 0.6,
        "fact_version": 1,
        "embedding_ver": "v1.0.0",
        "created_at": datetime.now(timezone.utc).timestamp(),
        "vector": green_tea_vector
    }
    milvus_repo.insert_facts([milvus_record])

    # 2. Test Rule 1: Exact statement match -> Skip
    skipped_fact = {
        "statement": "user likes drinking green tea.",  # case differences
        "category": category,
        "importance": 0.9,
        "vector": green_tea_vector
    }
    stats_skipped = run_async(service.merge_user_facts(user_id, "conv-1", [skipped_fact]))
    assert stats_skipped["skipped"] == 1
    assert stats_skipped["inserted"] == 0
    assert stats_skipped["superseded"] == 0

    # 3. Test Rule 2: Low similarity -> Insert as new fact
    # Vector representing something completely different, e.g. "User plays the piano."
    piano_vector = [0.0, 1.0] + [0.0] * (settings.VECTOR_DIMENSION - 2)
    new_inserted_fact = {
        "statement": "User plays the piano.",
        "category": category,
        "importance": 0.7,
        "vector": piano_vector
    }
    stats_inserted = run_async(service.merge_user_facts(user_id, "conv-1", [new_inserted_fact]))
    assert stats_inserted["inserted"] == 1
    assert stats_inserted["skipped"] == 0

    # Verify both facts exist in Cassandra
    facts_cassandra = cassandra_repo.get_facts(user_id, category)
    assert len(facts_cassandra) == 2
    statements = [f["statement"] for f in facts_cassandra]
    assert "User likes drinking green tea." in statements
    assert "User plays the piano." in statements

    # 4. Test Rule 3: High similarity, lower importance -> Ignore
    # Vector close to green tea vector
    close_tea_vector = [0.99, 0.1] + [0.0] * (settings.VECTOR_DIMENSION - 2)
    ignored_fact = {
        "statement": "User enjoys green tea.",
        "category": category,
        "importance": 0.4,  # Lower than initial 0.6
        "vector": close_tea_vector
    }
    stats_ignored = run_async(service.merge_user_facts(user_id, "conv-1", [ignored_fact]))
    assert stats_ignored["ignored"] == 1
    assert stats_ignored["superseded"] == 0

    # 5. Test Rule 4: High similarity, higher importance -> Supersede
    superseding_fact = {
        "statement": "User loves green tea over black tea.",
        "category": category,
        "importance": 0.9,  # Higher than initial 0.6
        "vector": close_tea_vector
    }
    stats_superseded = run_async(service.merge_user_facts(user_id, "conv-1", [superseding_fact]))
    assert stats_superseded["superseded"] == 1
    assert stats_superseded["inserted"] == 0

    # Verify that the initial fact is deleted and replaced with the new one at version 2
    facts_cassandra_final = cassandra_repo.get_facts(user_id, category)
    assert len(facts_cassandra_final) == 2
    
    statements_final = [f["statement"] for f in facts_cassandra_final]
    assert "User plays the piano." in statements_final
    assert "User loves green tea over black tea." in statements_final
    assert "User likes drinking green tea." not in statements_final

    # Find the green tea fact and check version is incremented (1 + 1 = 2)
    green_tea_fact = next(f for f in facts_cassandra_final if "green tea" in f["statement"])
    assert green_tea_fact["fact_version"] == 2
    assert green_tea_fact["fact_id"] != initial_fact_id
