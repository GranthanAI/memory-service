"""
tests/unit/test_ranking_service.py

Unit tests for Phase 15 Decoupled Scoring & Ranking Engine.
Validates exponential time-decay mathematics, clock drift capping,
importance normalization, and descending sorted ranking lists.
"""

import pytest
from datetime import datetime, timezone, timedelta

from app.core.config import settings
from app.services.ranking_service import RankingService


def test_calculate_score_recency_decay_exact_values():
    """Asserts that scoring calculations apply correct exponential decay based on days."""
    now = datetime.now(timezone.utc)
    
    # 1. Created right now: t=0 -> e^0 = 1.0
    score_now = RankingService.calculate_score(
        similarity=0.8,
        importance=0.6,
        created_at=now,
        now=now
    )
    # Expected: 0.5 * 0.8 + 0.2 * 1.0 + 0.3 * 0.6 = 0.4 + 0.2 + 0.18 = 0.78
    assert pytest.approx(score_now, 1e-5) == 0.78

    # 2. Future creation time (drift): capped at t=0 -> e^0 = 1.0
    future_time = now + timedelta(minutes=5)
    score_future = RankingService.calculate_score(
        similarity=0.8,
        importance=0.6,
        created_at=future_time,
        now=now
    )
    assert pytest.approx(score_future, 1e-5) == 0.78

    # 3. Created 20 days ago: t=20 -> e^(-0.05 * 20) = e^-1 approx 0.36787944
    twenty_days_ago = now - timedelta(days=20)
    score_past = RankingService.calculate_score(
        similarity=0.8,
        importance=0.6,
        created_at=twenty_days_ago,
        now=now
    )
    # Expected: 0.5 * 0.8 + 0.2 * e^-1 + 0.3 * 0.6 = 0.4 + 0.2 * 0.36787944 + 0.18 = 0.58 + 0.0735758 = 0.6535758
    expected_score = 0.5 * 0.8 + 0.2 * 0.36787944 + 0.3 * 0.6
    assert pytest.approx(score_past, 1e-5) == expected_score


def test_calculate_score_accepts_timestamp_float():
    """Asserts that created_at accepts epoch float timestamps."""
    now = datetime.now(timezone.utc)
    timestamp = now.timestamp()

    score = RankingService.calculate_score(
        similarity=0.8,
        importance=0.6,
        created_at=timestamp,
        now=now
    )
    assert pytest.approx(score, 1e-5) == 0.78


def test_rank_facts_sorting_and_limits():
    """Asserts that rank_facts returns facts ordered descending by score and respects limits."""
    now = datetime.now(timezone.utc)

    # 3 mock facts:
    # - fact_1: created now, high similarity, high importance
    # - fact_2: created now, low similarity, low importance
    # - fact_3: created 100 days ago, high similarity, high importance (decayed)
    facts = [
        {
            "fact_id": "fact-2",
            "distance": 0.3,
            "importance": 0.3,
            "created_at": now
        },
        {
            "fact_id": "fact-1",
            "distance": 0.9,
            "importance": 0.8,
            "created_at": now
        },
        {
            "fact_id": "fact-3",
            "distance": 0.9,
            "importance": 0.8,
            "created_at": now - timedelta(days=100)
        }
    ]

    # Rank all
    ranked_all = RankingService.rank_facts(facts, now=now)
    assert len(ranked_all) == 3
    
    # Assert sorted descending: fact-1 (high score) -> fact-3 (decayed) -> fact-2 (low similarity)
    assert ranked_all[0]["fact_id"] == "fact-1"
    assert ranked_all[1]["fact_id"] == "fact-3"
    assert ranked_all[2]["fact_id"] == "fact-2"

    assert ranked_all[0]["score"] > ranked_all[1]["score"]
    assert ranked_all[1]["score"] > ranked_all[2]["score"]

    # Rank with limit = 2
    ranked_limited = RankingService.rank_facts(facts, limit=2, now=now)
    assert len(ranked_limited) == 2
    assert ranked_limited[0]["fact_id"] == "fact-1"
    assert ranked_limited[1]["fact_id"] == "fact-3"


def test_rank_facts_legacy_and_edge_normalization():
    """Asserts that importance values above 1.0 (legacy 1-10 scale) are normalized, and similarity aliases work."""
    now = datetime.now(timezone.utc)

    facts = [
        {
            "fact_id": "fact-legacy",
            "similarity": 0.8,      # Uses 'similarity' alias instead of 'distance'
            "importance": 8.0,      # Legacy 1-10 scale
            "created_at": now
        }
    ]

    ranked = RankingService.rank_facts(facts, now=now)
    assert len(ranked) == 1
    
    # Expected: similarity=0.8, importance normalized to 0.8, recency=1.0
    # Score: 0.5 * 0.8 + 0.2 * 1.0 + 0.3 * 0.8 = 0.4 + 0.2 + 0.24 = 0.84
    assert ranked[0]["score"] == 0.84
