"""
app/services/ranking_service.py

Decoupled Scoring & Ranking Engine calculates retrieval importance weights
using Cosine similarity, fact importance, and exponential recency decay.
"""

import logging
import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.core.config import settings

logger = logging.getLogger("memory_service.services.ranking_service")


class RankingService:
    """
    Calculates combined scoring for facts using:
    Score = w_sim * S_sim + w_rec * e^(-lambda * t) + w_imp * S_imp
    """

    @staticmethod
    def calculate_score(
        similarity: float,
        importance: float,
        created_at: Any,
        now: Optional[datetime] = None
    ) -> float:
        """
        Calculates the combined ranking score for a single fact.
        - similarity: Float between 0.0 and 1.0 (Cosine similarity score).
        - importance: Float between 0.0 and 1.0 (Fact importance score).
        - created_at: datetime object or numeric timestamp (UTC).
        - now: Optional datetime object representing reference 'current' time.
        """
        # 1. Parse created_at
        if isinstance(created_at, (int, float)):
            created_dt = datetime.fromtimestamp(created_at, timezone.utc)
        elif isinstance(created_at, datetime):
            created_dt = created_at
            if created_dt.tzinfo is None:
                created_dt = created_dt.replace(tzinfo=timezone.utc)
        else:
            raise ValueError("created_at must be a datetime or numeric timestamp.")

        # 2. Parse reference now
        ref_now = now or datetime.now(timezone.utc)
        if ref_now.tzinfo is None:
            ref_now = ref_now.replace(tzinfo=timezone.utc)

        # 3. Calculate time difference in days (cap at 0.0 to handle future clock drift)
        delta_seconds = (ref_now - created_dt).total_seconds()
        t_days = max(0.0, delta_seconds / 86400.0)

        # 4. Calculate exponential decay term for recency: e^(-lambda * t)
        decay_rate = settings.RETRIEVAL_DECAY_RATE
        recency_score = math.exp(-decay_rate * t_days)

        # 5. Extract weights from configuration
        w_sim = settings.RETRIEVAL_WEIGHT_SIMILARITY
        w_rec = settings.RETRIEVAL_WEIGHT_RECENCY
        w_imp = settings.RETRIEVAL_WEIGHT_IMPORTANCE

        # 6. Apply scoring function
        final_score = (w_sim * similarity) + (w_rec * recency_score) + (w_imp * importance)
        return float(final_score)

    @classmethod
    def rank_facts(
        cls,
        facts: List[Dict[str, Any]],
        limit: Optional[int] = None,
        now: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Calculates scores for a list of facts, appends the score to each fact,
        sorts them descending by score, and returns the top limit.
        Each fact must contain: "distance" or "similarity", "importance", and "created_at".
        """
        if not facts:
            return []

        scored_facts = []
        for fact in facts:
            # support either 'distance' (milvus) or 'similarity'
            similarity = fact.get("distance")
            if similarity is None:
                similarity = fact.get("similarity", 0.0)

            # support created_at as timestamp or datetime
            created_at = fact.get("created_at")
            importance = fact.get("importance", 0.0)

            # If importance is stored on 1-10 scale in legacy records, normalize it
            if importance > 1.0:
                importance = importance / 10.0

            score = cls.calculate_score(
                similarity=similarity,
                importance=importance,
                created_at=created_at,
                now=now
            )

            # Copy fact to avoid mutating original list and insert final score
            fact_copy = dict(fact)
            fact_copy["score"] = round(score, 4)
            scored_facts.append(fact_copy)

        # Sort descending by score
        scored_facts.sort(key=lambda x: x["score"], reverse=True)

        if limit is not None:
            return scored_facts[:limit]
        return scored_facts
