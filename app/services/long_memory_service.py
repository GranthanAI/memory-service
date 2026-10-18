"""
app/services/long_memory_service.py

Long Memory Service: the single place long-term user facts are written.

Every candidate goes through the policy in memory_policy.py (is it durable, user-specific and
important enough?) and then the merge below (is it new, a duplicate, a corroboration or a
correction of something already stored?), against Cassandra (source of truth) and Milvus (vectors).
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.services import memory_policy
from app.services.memory_controls import MemoryControls, statement_hash
from app.core.config import settings
from app.repositories.cassandra_repository import CassandraRepository
from app.repositories.milvus_repository import MilvusRepository

logger = logging.getLogger("memory_service.services.long_memory_service")


class LongMemoryService:
    """
    Manages long-term user memories and applies the Fact Merge Policy to incoming facts:
    reject, skip (duplicate), insert, supersede (newer / corrected / corroborated) or ignore.
    """

    def __init__(self, cassandra_repo: CassandraRepository, milvus_repo: MilvusRepository):
        self.cassandra_repo = cassandra_repo
        self.milvus_repo = milvus_repo
        self.controls = MemoryControls(cassandra_repo)

    # ------------------------------------------------------------------ helpers

    def _existing_by_key(self, user_id: str) -> Dict[str, Dict[str, Any]]:
        """All of the user's stored facts keyed by normalized statement (any category), so a
        reworded or re-categorized duplicate is recognised."""
        existing: Dict[str, Dict[str, Any]] = {}
        for category in memory_policy.CATEGORIES:
            for row in self.cassandra_repo.get_facts(user_id, category):
                existing[memory_policy.normalize_statement(row["statement"])] = row
        return existing

    def _write(self, *, user_id: str, conversation_id: str, category: str, statement: str, importance: float,
               vector: List[float], version: int) -> uuid.UUID:
        fact_id = uuid.uuid4()
        now = datetime.now(timezone.utc)
        self.cassandra_repo.upsert_fact({
            "user_id": user_id, "category": category, "fact_id": fact_id, "conversation_id": conversation_id,
            "statement": statement, "importance": importance, "fact_version": version,
            "embedding_version": settings.EMBEDDING_MODEL_VERSION, "created_at": now, "updated_at": now,
        })
        self.milvus_repo.insert_facts([{
            "fact_id": str(fact_id), "user_id": user_id, "conversation_id": conversation_id,
            "category": category, "statement": statement, "importance": importance, "fact_version": version,
            "embedding_ver": settings.EMBEDDING_MODEL_VERSION, "created_at": now.timestamp(), "vector": vector,
        }])
        return fact_id

    def _remove(self, user_id: str, category: str, fact_id_str: str) -> None:
        self.cassandra_repo.delete_fact(user_id, category, uuid.UUID(fact_id_str))
        self.milvus_repo.delete_fact(user_id, fact_id_str)

    # ------------------------------------------------------------------ merge

    async def merge_user_facts(
        self,
        user_id: str,
        conversation_id: str,
        incoming_facts: List[Dict[str, Any]]
    ) -> Dict[str, int]:
        """
        Applies the policy and merge rules to freshly extracted facts. Each incoming fact has
        "statement", "category", "importance" and "vector". Returns counts of what happened:
        inserted / superseded / ignored / skipped / rejected.
        """
        stats = {"inserted": 0, "superseded": 0, "ignored": 0, "skipped": 0, "rejected": 0}
        if not incoming_facts:
            return stats

        # User controls (enforced here, the only place facts are written): a paused memory stores
        # nothing, and statements the user deleted/corrected are never re-learned.
        if not self.controls.is_enabled(user_id):
            logger.info(f"Memory is off for user {user_id}; not storing {len(incoming_facts)} fact(s).")
            stats["skipped"] += len(incoming_facts)
            return stats
        forgotten = self.controls.tombstones(user_id)

        existing = self._existing_by_key(user_id)

        for fact in incoming_facts:
            # 1. Policy: is it a durable, user-specific, important statement?
            cand = memory_policy.evaluate(fact["category"], fact["importance"], fact["statement"])
            if not cand.accepted:
                logger.info(f"Rejected candidate memory ({cand.reason}) for user {user_id}.")
                stats["rejected"] += 1
                continue
            statement, category, importance, vector = cand.statement, cand.category, cand.importance, fact["vector"]

            if forgotten and statement_hash(statement) in forgotten:
                stats["skipped"] += 1
                continue

            # 2. Same statement already stored (ignoring wording noise and category)?
            key = memory_policy.normalize_statement(statement)
            same = existing.get(key)
            if same is not None:
                if (memory_policy.other_conversation(same.get("conversation_id"), conversation_id)
                        and same["category"] == "interests"):
                    self._corroborate(user_id, conversation_id, same, importance, vector, existing, key)
                    stats["superseded"] += 1
                else:
                    stats["skipped"] += 1
                continue

            # 3. Semantically equivalent (or corrected) memory?
            hits = self.milvus_repo.search_facts(user_id=user_id, query_vector=vector, limit=1)
            closest = hits[0] if hits else None
            if closest is None or closest["distance"] < memory_policy.threshold_for(category, statement):
                fact_id = self._write(user_id=user_id, conversation_id=conversation_id, category=category,
                                      statement=statement, importance=importance, vector=vector, version=1)
                existing[key] = {"fact_id": fact_id, "category": category, "statement": statement,
                                 "conversation_id": conversation_id, "importance": importance}
                stats["inserted"] += 1
                continue

            old_importance = closest["importance"]
            is_correction = memory_policy.has_update_language(statement)
            if category == "interests" and memory_policy.other_conversation(closest.get("conversation_id"), conversation_id):
                importance = memory_policy.corroborated_importance(category, old_importance, importance)
            # Rewording alone never replaces a memory (no churn): it needs to be a correction, or more
            # important than what is stored (which includes corroboration from another conversation).
            if importance > old_importance or is_correction:
                self._remove(user_id, closest.get("category") or category, closest["fact_id"])
                existing.pop(memory_policy.normalize_statement(closest["statement"]), None)
                fact_id = self._write(user_id=user_id, conversation_id=conversation_id, category=category,
                                      statement=statement, importance=max(importance, 0.0), vector=vector,
                                      version=closest["fact_version"] + 1)
                existing[key] = {"fact_id": fact_id, "category": category, "statement": statement,
                                 "conversation_id": conversation_id, "importance": importance}
                stats["superseded"] += 1
            else:
                logger.info(f"Ignoring lower-importance duplicate of '{closest['statement']}'.")
                stats["ignored"] += 1

        return stats

    def _corroborate(self, user_id: str, conversation_id: str, same: Dict[str, Any], importance: float,
                     vector: List[float], existing: Dict[str, Dict[str, Any]], key: str) -> None:
        """The same interest seen in a different conversation: one memory, raised importance, new version."""
        boosted = memory_policy.corroborated_importance(same["category"], same["importance"], importance)
        self._remove(user_id, same["category"], str(same["fact_id"]))
        fact_id = self._write(user_id=user_id, conversation_id=conversation_id, category=same["category"],
                              statement=same["statement"], importance=boosted, vector=vector,
                              version=int(same.get("fact_version") or 1) + 1)
        existing[key] = {**same, "fact_id": fact_id, "importance": boosted, "conversation_id": conversation_id}
