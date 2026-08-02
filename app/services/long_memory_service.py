"""
app/services/long_memory_service.py

Long Memory Service implements structured long-term user fact extraction
and the Fact Merge Policy using Cassandra and Milvus indices.
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.core.config import settings
from app.repositories.cassandra_repository import CassandraRepository
from app.repositories.milvus_repository import MilvusRepository

logger = logging.getLogger("memory_service.services.long_memory_service")


class LongMemoryService:
    """
    Manages long-term user memories.
    Applies the Fact Merge Policy on incoming facts to skip, ignore, insert,
    or supersede existing memories.
    """

    def __init__(self, cassandra_repo: CassandraRepository, milvus_repo: MilvusRepository):
        self.cassandra_repo = cassandra_repo
        self.milvus_repo = milvus_repo

    async def merge_user_facts(
        self,
        user_id: str,
        conversation_id: str,
        incoming_facts: List[Dict[str, Any]]
    ) -> Dict[str, int]:
        """
        Processes a list of newly extracted facts for a user and applies the Fact Merge Policy.
        Returns a dictionary summarizing the actions taken (inserted, superseded, ignored, skipped).
        Each incoming fact must contain: "statement", "category", "importance", and "vector".
        """
        stats = {
            "inserted": 0,
            "superseded": 0,
            "ignored": 0,
            "skipped": 0
        }

        if not incoming_facts:
            return stats

        # Group incoming facts by category to minimize database queries
        categories = set(f["category"] for f in incoming_facts)
        existing_facts_by_cat = {}
        for cat in categories:
            # Query Cassandra for existing user facts under this category
            existing_facts_by_cat[cat] = self.cassandra_repo.get_facts(user_id, cat)

        threshold = settings.FACT_MERGE_SIMILARITY_THRESHOLD

        for fact in incoming_facts:
            category = fact["category"]
            statement = fact["statement"].strip()
            importance = fact["importance"]
            vector = fact["vector"]

            existing_list = existing_facts_by_cat.get(category, [])

            # Rule 1: Exact statement match (hash equality) -> Skip
            is_exact_match = False
            for old_fact in existing_list:
                if old_fact["statement"].strip().lower() == statement.lower():
                    is_exact_match = True
                    break

            if is_exact_match:
                logger.info(f"Skipping exact match fact: '{statement}' for user {user_id}.")
                stats["skipped"] += 1
                continue

            # Rule 2: Vector search in Milvus within the user's category partition
            hits = self.milvus_repo.search_facts(
                user_id=user_id,
                query_vector=vector,
                limit=1,
                category=category
            )

            closest_hit = hits[0] if hits else None

            # Rule 3: Apply similarity comparison
            if closest_hit is None or closest_hit["distance"] < threshold:
                # Similarity is low -> Insert as new fact
                fact_id = uuid.uuid4()
                logger.info(
                    f"No matching fact found for '{statement}' (similarity below threshold). "
                    f"Inserting as new fact {fact_id}."
                )

                # Persist to Cassandra
                cassandra_record = {
                    "user_id": user_id,
                    "category": category,
                    "fact_id": fact_id,
                    "conversation_id": conversation_id,
                    "statement": statement,
                    "importance": importance,
                    "fact_version": 1,
                    "embedding_version": settings.EMBEDDING_MODEL_VERSION,
                    "created_at": datetime.now(timezone.utc),
                    "updated_at": datetime.now(timezone.utc)
                }
                self.cassandra_repo.upsert_fact(cassandra_record)

                # Persist to Milvus
                milvus_record = {
                    "fact_id": str(fact_id),
                    "user_id": user_id,
                    "conversation_id": conversation_id,
                    "category": category,
                    "statement": statement,
                    "importance": importance,
                    "fact_version": 1,
                    "embedding_ver": settings.EMBEDDING_MODEL_VERSION,
                    "created_at": datetime.now(timezone.utc).timestamp(),
                    "vector": vector
                }
                self.milvus_repo.insert_facts([milvus_record])
                
                # Append to existing list to handle consecutive duplicates in the same batch
                existing_list.append(cassandra_record)
                stats["inserted"] += 1

            else:
                # Similarity is high (>= threshold)
                old_importance = closest_hit["importance"]
                old_fact_id_str = closest_hit["fact_id"]
                old_fact_id = uuid.UUID(old_fact_id_str)
                old_version = closest_hit["fact_version"]

                if importance >= old_importance:
                    # New fact has higher/equal importance -> Supersede old fact
                    new_fact_id = uuid.uuid4()
                    logger.info(
                        f"Superseding existing fact '{closest_hit['statement']}' (importance: {old_importance}) "
                        f"with new fact '{statement}' (importance: {importance})."
                    )

                    # Delete old fact from Cassandra & Milvus
                    self.cassandra_repo.delete_fact(user_id, category, old_fact_id)
                    self.milvus_repo.delete_fact(user_id, old_fact_id_str)

                    # Insert new fact with incremented version
                    new_version = old_version + 1
                    
                    cassandra_record = {
                        "user_id": user_id,
                        "category": category,
                        "fact_id": new_fact_id,
                        "conversation_id": conversation_id,
                        "statement": statement,
                        "importance": importance,
                        "fact_version": new_version,
                        "embedding_version": settings.EMBEDDING_MODEL_VERSION,
                        "created_at": datetime.now(timezone.utc),
                        "updated_at": datetime.now(timezone.utc)
                    }
                    self.cassandra_repo.upsert_fact(cassandra_record)

                    milvus_record = {
                        "fact_id": str(new_fact_id),
                        "user_id": user_id,
                        "conversation_id": conversation_id,
                        "category": category,
                        "statement": statement,
                        "importance": importance,
                        "fact_version": new_version,
                        "embedding_ver": settings.EMBEDDING_MODEL_VERSION,
                        "created_at": datetime.now(timezone.utc).timestamp(),
                        "vector": vector
                    }
                    self.milvus_repo.insert_facts([milvus_record])

                    # Remove old fact and add new fact to existing list
                    existing_list = [f for f in existing_list if f["fact_id"] != old_fact_id]
                    existing_list.append(cassandra_record)
                    existing_facts_by_cat[category] = existing_list

                    stats["superseded"] += 1

                else:
                    # New fact has lower importance -> Ignore
                    logger.info(
                        f"Ignoring fact '{statement}' (importance: {importance}) because it has lower "
                        f"importance than existing fact '{closest_hit['statement']}' (importance: {old_importance})."
                    )
                    stats["ignored"] += 1

        return stats
