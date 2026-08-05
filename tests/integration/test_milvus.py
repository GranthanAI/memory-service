"""
tests/integration/test_milvus.py

Integration tests for Phase 10: Milvus Repository Layer.
Runs queries on the live Milvus container to assert schema creation, dynamic user partitions,
Cosine similarity searches, and deletions.
"""

import asyncio
import time
import pytest
from pymilvus import utility

from app.db.session import initialize_db_sessions, close_db_sessions
from app.repositories.milvus_repository import MilvusRepository
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
def clean_collection():
    """Drops the test Milvus collection before and after running tests to guarantee isolation."""
    coll_name = "test_user_memory_vectors"
    if utility.has_collection(coll_name):
        utility.drop_collection(coll_name)
    yield
    if utility.has_collection(coll_name):
        utility.drop_collection(coll_name)


def test_milvus_lifecycle_integration(clean_collection):
    """
    Verifies Milvus collection instantiation, batch insertion, dynamic partitioning search,
    Cosine metric range, and deletion operations on a live server.
    """
    coll_name = "test_user_memory_vectors"
    repo = MilvusRepository(collection_name=coll_name)

    # 1. Prepare records for two users (user-1 and user-2)
    # Vectors are designed such that vector1 and vector1_query are identical (cosine similarity ~1.0)
    # vector2 is orthogonal (cosine similarity ~0.0)
    vector1 = [1.0] + [0.0] * (settings.VECTOR_DIMENSION - 1)
    vector2 = [0.0, 1.0] + [0.0] * (settings.VECTOR_DIMENSION - 2)
    
    rec_user1_pref = {
        "fact_id": "fact-1",
        "user_id": "user-1",
        "conversation_id": "conv-a",
        "category": "preferences",
        "statement": "User likes warm tea.",
        "importance": 0.85,
        "fact_version": 1,
        "embedding_ver": "v1.0",
        "created_at": time.time(),
        "vector": vector1
    }

    rec_user1_work = {
        "fact_id": "fact-2",
        "user_id": "user-1",
        "conversation_id": "conv-a",
        "category": "work",
        "statement": "User is a backend engineer.",
        "importance": 0.90,
        "fact_version": 1,
        "embedding_ver": "v1.0",
        "created_at": time.time(),
        "vector": vector1
    }

    rec_user2 = {
        "fact_id": "fact-3",
        "user_id": "user-2",
        "conversation_id": "conv-b",
        "category": "preferences",
        "statement": "User likes cold coffee.",
        "importance": 0.75,
        "fact_version": 1,
        "embedding_ver": "v1.0",
        "created_at": time.time(),
        "vector": vector2
    }

    # 2. Insert records
    ids = repo.insert_facts([rec_user1_pref, rec_user1_work, rec_user2])
    assert len(ids) == 3
    assert "fact-1" in ids
    assert "fact-2" in ids
    assert "fact-3" in ids

    # 3. Perform semantic search for user-1 (routing partition key filter)
    # Search vector matches user-1's tea preference vector
    search_vector = [1.0] + [0.0] * (settings.VECTOR_DIMENSION - 1)
    hits_user1 = repo.search_facts(
        user_id="user-1",
        query_vector=search_vector,
        limit=5,
        consistency_level="Strong"
    )

    # Search should only return user-1's facts (fact-1, fact-2), excluding user-2's fact-3
    assert len(hits_user1) == 2
    assert any(h["fact_id"] == "fact-1" for h in hits_user1)
    assert any(h["fact_id"] == "fact-2" for h in hits_user1)
    assert not any(h["fact_id"] == "fact-3" for h in hits_user1)

    # Verify that distance scores return valid Cosine values (~1.0 for vector1)
    tea_hit = [h for h in hits_user1 if h["fact_id"] == "fact-1"][0]
    assert pytest.approx(tea_hit["distance"], abs=1e-3) == 1.0

    # 4. Perform search with category filter
    hits_category = repo.search_facts(
        user_id="user-1",
        query_vector=search_vector,
        limit=5,
        category="work",
        consistency_level="Strong"
    )
    assert len(hits_category) == 1
    assert hits_category[0]["fact_id"] == "fact-2"

    # 5. Delete a specific fact and assert it is gone
    repo.delete_fact(user_id="user-1", fact_id="fact-1")
    hits_after_del = repo.search_facts(
        user_id="user-1",
        query_vector=search_vector,
        limit=5,
        consistency_level="Strong"
    )
    assert len(hits_after_del) == 1
    assert hits_after_del[0]["fact_id"] == "fact-2"

    # 6. Delete all remaining user-1 facts
    repo.delete_user_facts(user_id="user-1")
    hits_after_user_del = repo.search_facts(
        user_id="user-1",
        query_vector=search_vector,
        limit=5,
        consistency_level="Strong"
    )
    assert len(hits_after_user_del) == 0

    # Ensure user-2's facts are unaffected
    hits_user2 = repo.search_facts(
        user_id="user-2",
        query_vector=[0.0, 1.0] + [0.0] * (settings.VECTOR_DIMENSION - 2),
        limit=5,
        consistency_level="Strong"
    )
    assert len(hits_user2) == 1
    assert hits_user2[0]["fact_id"] == "fact-3"
