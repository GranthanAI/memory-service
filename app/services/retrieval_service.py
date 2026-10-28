"""
app/services/retrieval_service.py

Retrieval Service handles cache read-through and DB fallback lookups,
combining snapshots, recent messages, summaries, and Milvus semantic search.
"""

import logging
import uuid
from typing import Any, Dict, List, Optional

from app.core.config import settings
from app.repositories.memory_repository import MemoryRepository
from app.repositories.milvus_repository import MilvusRepository
from app.services.ranking_service import RankingService
from app.services.memory_controls import MemoryControls, statement_hash

logger = logging.getLogger("memory_service.services.retrieval_service")

STABLE_PREFERENCE_MIN_IMPORTANCE = 0.7
STABLE_PREFERENCE_LIMIT = 3
STABLE_PREFERENCE_BASELINE_SIMILARITY = 0.4


class RetrievalService:
    """
    Orchestrates high-speed read-through queries for conversation state and semantic facts.
    """

    def __init__(self, memory_repo: MemoryRepository, milvus_repo: MilvusRepository,
                 controls: Optional[MemoryControls] = None):
        self.memory_repo = memory_repo
        self.milvus_repo = milvus_repo
        # User memory controls (pause / forgotten statements). Required in production
        # wiring (container); optional only for older callers/tests.
        self.controls = controls

    async def get_or_hydrate_snapshot(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves snapshot from memory repository (which handles read-through cache automatically).
        """
        try:
            return await self.memory_repo.get_snapshot(conversation_id)
        except Exception as e:
            logger.error(f"Error fetching snapshot for conversation {conversation_id}: {e}")
            return None

    async def get_or_hydrate_recent_messages(
        self,
        conversation_id: str,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Retrieves recent messages list from memory repository (handles sliding list read-through).
        """
        try:
            return await self.memory_repo.get_recent_messages(conversation_id, limit=limit)
        except Exception as e:
            logger.error(f"Error fetching recent messages for conversation {conversation_id}: {e}")
            return []

    async def get_or_hydrate_summary_record(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Summary text, version and watermark as one object from one source (see MemoryRepository.get_summary_record)."""
        try:
            return await self.memory_repo.get_summary_record(conversation_id)
        except Exception as e:
            logger.error(f"Error fetching summary record for conversation {conversation_id}: {e}")
            return None

    async def get_or_hydrate_summary(self, conversation_id: str) -> Optional[str]:
        """
        Retrieves summary text from memory repository (handles read-through zstd cached summary).
        """
        try:
            return await self.memory_repo.get_summary(conversation_id)
        except Exception as e:
            logger.error(f"Error fetching summary for conversation {conversation_id}: {e}")
            return None

    def _stable_preferences(self, user_id: str, already: set) -> List[Dict[str, Any]]:
        """Top stored preferences that are important enough to apply to any request."""
        try:
            rows = self.milvus_repo.list_user_facts(user_id)
        except Exception:
            return []
        if not isinstance(rows, list):
            return []
        stable = [
            r for r in rows
            if r.get("category") == "preferences"
            and float(r.get("importance") or 0) >= STABLE_PREFERENCE_MIN_IMPORTANCE
            and r.get("fact_id") not in already
        ]
        stable.sort(key=lambda r: float(r.get("importance") or 0), reverse=True)
        # No topical similarity: a neutral baseline lets importance and recency decide the rank.
        return [dict(r, distance=STABLE_PREFERENCE_BASELINE_SIMILARITY) for r in stable[:STABLE_PREFERENCE_LIMIT]]

    def _core_profile(self, user_id: str, already: set) -> List[Dict[str, Any]]:
        """Who the user is (category "personal": identity, role, location...), always offered, within a small budget."""
        if not settings.CORE_PROFILE_ENABLED:
            return []
        try:
            rows = self.milvus_repo.list_user_facts(user_id)
        except Exception:
            return []
        if not isinstance(rows, list):
            return []
        core = [r for r in rows
                if r.get("category") == "personal"
                and float(r.get("importance") or 0) >= settings.CORE_PROFILE_MIN_IMPORTANCE
                and r.get("fact_id") not in already]
        core.sort(key=lambda r: float(r.get("importance") or 0), reverse=True)
        picked, used = [], 0
        for r in core[: settings.CORE_PROFILE_MAX_FACTS]:
            size = len(str(r.get("statement") or ""))
            if picked and used + size > settings.CORE_PROFILE_CHAR_BUDGET:
                break
            used += size
            picked.append(dict(r, distance=STABLE_PREFERENCE_BASELINE_SIMILARITY, core_profile=True))
        return picked

    def _enrich_and_filter_active(self, user_id: str, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Adds last_confirmed / last_used from Cassandra (the source of truth) and drops any hit whose fact is no longer
        ACTIVE there (a retired fact whose vector delete was lost must never be shown). Without Cassandra access the hits
        are returned unchanged."""
        cassandra = getattr(self.memory_repo, "cassandra_repo", None)
        if cassandra is None or not hasattr(cassandra, "get_fact"):
            return hits
        out = []
        for hit in hits:
            try:
                row = cassandra.get_fact(user_id, hit.get("category"), uuid.UUID(str(hit.get("fact_id"))))
            except Exception:
                out.append(hit)                                      # cannot tell: keep (Milvus is already per-user)
                continue
            if not isinstance(row, dict):
                out.append(hit)
                continue
            status = row.get("status")
            if isinstance(status, str) and status != "ACTIVE":
                continue
            out.append(dict(hit, last_confirmed=row.get("last_confirmed"), last_used=row.get("last_used")))
        return out

    @staticmethod
    def _within_budget(ranked: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Keeps the best-ranked facts that fit the character budget for injected memory."""
        kept: List[Dict[str, Any]] = []
        used = 0
        for fact in ranked:
            size = len(str(fact.get("statement") or ""))
            if kept and used + size > settings.RETRIEVAL_FACT_CHAR_BUDGET:
                continue
            used += size
            kept.append(fact)
        return kept

    async def retrieve_relevant_facts(
        self,
        user_id: str,
        query_vector: List[float],
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Retrieves closest facts for a user from Milvus, then scores and ranks them
        using the Decoupled Scoring & Ranking Engine.
        """
        if self.controls is not None and not self.controls.is_enabled(user_id):
            return []  # memory paused: saved memories are not used in answers
        try:
            # 1. Fetch from Milvus scoped to user
            hits = self.milvus_repo.search_facts(
                user_id=user_id,
                query_vector=query_vector,
                limit=limit
            )
            # Nearest-neighbour search always returns *something*: facts that are not actually about this
            # request are dropped so unrelated memories are not injected into the prompt.
            hits = [h for h in (hits or []) if float(h.get("distance") or 0.0) >= settings.RETRIEVAL_MIN_SIMILARITY]
            # Stable preferences (how the user likes answers) apply whatever the topic, so they are
            # offered even when they are not semantically close to this question.
            hits.extend(self._stable_preferences(user_id, {h.get("fact_id") for h in hits}))
            hits.extend(self._core_profile(user_id, {h.get("fact_id") for h in hits}))
            hits = self._enrich_and_filter_active(user_id, hits)
            if not hits:
                return []
            # A memory the user deleted must never be used, even if its vector delete
            # had not completed (belt and braces next to the delete ordering).
            if self.controls is not None:
                forgotten = self.controls.tombstones(user_id)
                if forgotten:
                    hits = [h for h in hits if statement_hash(h.get("statement") or "") not in forgotten]
                    if not hits:
                        return []

            # 2. Score and rank using RankingService
            ranked = RankingService.rank_facts(hits, limit=limit)
            return self._within_budget(ranked)
        except Exception as e:
            logger.error(f"Error retrieving relevant facts for user {user_id}: {e}")
            return []
