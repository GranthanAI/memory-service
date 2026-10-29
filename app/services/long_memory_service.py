"""
app/services/long_memory_service.py

Long Memory Service: the single place long-term user facts are written.

Every candidate goes through the policy in memory_policy.py (is it durable, user-specific and
important enough?) and then the merge below (is it new, a duplicate, a corroboration or a
correction of something already stored?), against Cassandra (source of truth) and Milvus (vectors).

Nothing is destroyed by a merge. A replaced memory is RETIRED: its Cassandra row is kept as history (status SUPERSEDED,
superseded_by, valid_to) and only its vector is removed from Milvus so it can never be retrieved. The replacement is written
FIRST, then the old one retired, so a crash in between leaves two active versions (harmless, repaired by `reconcile_user_facts`)
rather than losing the memory. "Is this a correction?" is an explicit LLM judgement over the top candidates of the same
category (`judge`); without a judge (or when it fails) nothing is ever replaced on the strength of wording alone.
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, List, Optional

from app.services import memory_policy
from app.services.memory_controls import MemoryControls, statement_hash
from app.core.config import settings
from app.repositories.cassandra_repository import CassandraRepository
from app.repositories.milvus_repository import MilvusRepository

logger = logging.getLogger("memory_service.services.long_memory_service")

# similarity a stored memory needs to even be shown to the judge (below this it is a different topic)
CANDIDATE_MIN_SIMILARITY = memory_policy.UPDATE_SIMILARITY
CANDIDATE_LIMIT = 5

# async (new_statement, [existing statements]) -> {"relation": same|correction|different, "index": int|None}
FactJudge = Callable[[str, List[str]], Awaitable[Dict[str, Any]]]


class LongMemoryService:
    """
    Manages long-term user memories and applies the Fact Merge Policy to incoming facts:
    reject, skip (duplicate), insert, supersede (newer / corrected / corroborated) or ignore.
    """

    def __init__(self, cassandra_repo: CassandraRepository, milvus_repo: MilvusRepository,
                 judge: Optional[FactJudge] = None):
        self.cassandra_repo = cassandra_repo
        self.milvus_repo = milvus_repo
        self.controls = MemoryControls(cassandra_repo)
        self.judge = judge

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

    def _retire(self, user_id: str, category: str, fact_id_str: str, superseded_by: Optional[uuid.UUID]) -> None:
        """Retire a memory: keep the row as history, drop its vector. Cassandra first (the source of truth): if the vector
        delete then fails, the fact is already inactive and `reconcile_user_facts` removes the orphan vector."""
        self.cassandra_repo.retire_fact(user_id, category, uuid.UUID(str(fact_id_str)), superseded_by)
        try:
            self.milvus_repo.delete_fact(user_id, str(fact_id_str))
        except Exception as exc:
            logger.warning(f"Could not delete the vector of retired fact {fact_id_str} ({type(exc).__name__}); "
                           "reconcile_user_facts will remove it.")

    def _confirm(self, user_id: str, same: Dict[str, Any], conversation_id: Optional[str] = None) -> None:
        """The user said it again (in `conversation_id`): refresh its recency and make that conversation its source (best effort)."""
        try:
            self.cassandra_repo.touch_fact_confirmed(
                user_id, same["category"], uuid.UUID(str(same["fact_id"])), conversation_id)
        except Exception as exc:
            logger.debug(f"Could not refresh last_confirmed ({type(exc).__name__}).")

    async def _relation(self, statement: str, candidates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """The judge's verdict, or None when there is no judge / it failed (callers then stay non-destructive)."""
        if self.judge is None:
            return None
        try:
            verdict = await self.judge(statement, [c["statement"] for c in candidates])
        except Exception as exc:
            logger.warning(f"Fact judge unavailable ({type(exc).__name__}); not replacing anything on wording alone.")
            return None
        return verdict if isinstance(verdict, dict) else None

    async def reconcile_user_facts(self, user_id: str, embed: Optional[Callable[[str], Awaitable[List[float]]]] = None) -> Dict[str, int]:
        """Repairs Cassandra / Milvus divergence for one user (Cassandra is the truth):
          * a vector whose fact is not an ACTIVE Cassandra fact (retired, deleted, or half-finished merge) is deleted;
          * an ACTIVE fact with no vector is re-embedded and re-inserted when `embed` is given, else only counted."""
        stats = {"orphan_vectors_removed": 0, "missing_vectors_restored": 0, "missing_vectors": 0}
        active: Dict[str, Dict[str, Any]] = {}
        for category in memory_policy.CATEGORIES:
            for row in self.cassandra_repo.get_facts(user_id, category):
                active[str(row["fact_id"])] = row
        vectors = {str(r["fact_id"]) for r in self.milvus_repo.list_user_facts(user_id)}
        for fact_id in vectors - set(active):
            self.milvus_repo.delete_fact(user_id, fact_id)
            stats["orphan_vectors_removed"] += 1
        for fact_id in set(active) - vectors:
            row = active[fact_id]
            if embed is None:
                stats["missing_vectors"] += 1
                continue
            self.milvus_repo.insert_facts([{
                "fact_id": fact_id, "user_id": user_id, "conversation_id": row.get("conversation_id") or "",
                "category": row["category"], "statement": row["statement"], "importance": row["importance"],
                "fact_version": row.get("fact_version") or 1, "embedding_ver": settings.EMBEDDING_MODEL_VERSION,
                "created_at": (row.get("created_at") or datetime.now(timezone.utc)).timestamp(),
                "vector": await embed(row["statement"]),
            }])
            stats["missing_vectors_restored"] += 1
        return stats

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
                    self._confirm(user_id, same, conversation_id)
                    stats["skipped"] += 1
                continue

            # 3. Semantically equivalent (or corrected) memory? Top candidates of the SAME category; the decision whether
            #    one of them is replaced is an explicit judgement, never a keyword match.
            hits = self.milvus_repo.search_facts(user_id=user_id, query_vector=vector, limit=CANDIDATE_LIMIT) or []
            candidates = [h for h in hits
                          if (h.get("category") in (None, category)) and h["distance"] >= CANDIDATE_MIN_SIMILARITY]
            if not candidates:
                self._insert_new(user_id, conversation_id, category, statement, importance, vector, key, existing, stats)
                continue

            verdict = await self._relation(statement, candidates)
            relation = verdict["relation"] if verdict else None
            target = candidates[verdict["index"]] if verdict and verdict.get("index") is not None else candidates[0]
            if relation is None:
                # No judgement: only a near-identical match counts as the same memory; anything else is kept as new.
                same_memory = candidates[0]["distance"] >= memory_policy.SIMILARITY_THRESHOLDS.get(category, 0.85)
                relation, target = ("same" if same_memory else "different"), candidates[0]

            if relation == "different":
                self._insert_new(user_id, conversation_id, category, statement, importance, vector, key, existing, stats)
                continue

            old_importance = target["importance"]
            if category == "interests" and memory_policy.other_conversation(target.get("conversation_id"), conversation_id):
                importance = memory_policy.corroborated_importance(category, old_importance, importance)
            # Rewording alone never replaces a memory (no churn): it needs to be a correction (judged), or more
            # important than what is stored (which includes corroboration from another conversation).
            if relation == "correction" or importance > old_importance:
                fact_id = self._write(user_id=user_id, conversation_id=conversation_id, category=category,
                                      statement=statement, importance=max(importance, 0.0), vector=vector,
                                      version=int(target.get("fact_version") or 1) + 1)
                self._retire(user_id, target.get("category") or category, target["fact_id"], fact_id)
                existing.pop(memory_policy.normalize_statement(target["statement"]), None)
                existing[key] = {"fact_id": fact_id, "category": category, "statement": statement,
                                 "conversation_id": conversation_id, "importance": importance}
                stats["superseded"] += 1
            else:
                logger.info(f"Ignoring lower-importance duplicate of '{target['statement']}'.")
                self._confirm(user_id, {"category": target.get("category") or category, "fact_id": target["fact_id"]},
                              conversation_id)
                stats["ignored"] += 1

        return stats

    def _insert_new(self, user_id, conversation_id, category, statement, importance, vector, key, existing, stats) -> None:
        fact_id = self._write(user_id=user_id, conversation_id=conversation_id, category=category,
                              statement=statement, importance=importance, vector=vector, version=1)
        existing[key] = {"fact_id": fact_id, "category": category, "statement": statement,
                         "conversation_id": conversation_id, "importance": importance}
        stats["inserted"] += 1

    def _corroborate(self, user_id: str, conversation_id: str, same: Dict[str, Any], importance: float,
                     vector: List[float], existing: Dict[str, Dict[str, Any]], key: str) -> None:
        """The same interest seen in a different conversation: one memory, raised importance, new version."""
        boosted = memory_policy.corroborated_importance(same["category"], same["importance"], importance)
        fact_id = self._write(user_id=user_id, conversation_id=conversation_id, category=same["category"],
                              statement=same["statement"], importance=boosted, vector=vector,
                              version=int(same.get("fact_version") or 1) + 1)
        self._retire(user_id, same["category"], str(same["fact_id"]), fact_id)
        existing[key] = {**same, "fact_id": fact_id, "importance": boosted, "conversation_id": conversation_id}
