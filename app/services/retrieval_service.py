"""
app/services/retrieval_service.py

Retrieval Service handles cache read-through and DB fallback lookups,
combining snapshots, recent messages, summaries, and Milvus semantic search.
"""

import logging
from typing import Any, Dict, List, Optional

from app.core.config import settings
from app.repositories.memory_repository import MemoryRepository
from app.repositories.milvus_repository import MilvusRepository
from app.services.ranking_service import RankingService

logger = logging.getLogger("memory_service.services.retrieval_service")


class RetrievalService:
    """
    Orchestrates high-speed read-through queries for conversation state and semantic facts.
    """

    def __init__(self, memory_repo: MemoryRepository, milvus_repo: MilvusRepository):
        self.memory_repo = memory_repo
        self.milvus_repo = milvus_repo

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

    async def get_or_hydrate_summary(self, conversation_id: str) -> Optional[str]:
        """
        Retrieves summary text from memory repository (handles read-through zstd cached summary).
        """
        try:
            return await self.memory_repo.get_summary(conversation_id)
        except Exception as e:
            logger.error(f"Error fetching summary for conversation {conversation_id}: {e}")
            return None

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
        try:
            # 1. Fetch from Milvus scoped to user
            hits = self.milvus_repo.search_facts(
                user_id=user_id,
                query_vector=query_vector,
                limit=limit
            )
            if not hits:
                return []

            # 2. Score and rank using RankingService
            ranked = RankingService.rank_facts(hits, limit=limit)
            return ranked
        except Exception as e:
            logger.error(f"Error retrieving relevant facts for user {user_id}: {e}")
            return []
