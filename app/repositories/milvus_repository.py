"""
app/repositories/milvus_repository.py

Milvus Repository Layer handles semantic memory vector storage, HNSW indexing,
user_id dynamic partitioning, similarity search, and entity deletions.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, utility

from app.core.config import settings

logger = logging.getLogger("memory_service.repositories.milvus_repository")


class MilvusRepository:
    """
    Manages vector index definitions, bulk ingestions, dynamic partition key lookups,
    and deletion of conceptual memories in Milvus.
    """

    def __init__(self, collection_name: str = "user_memory_vectors"):
        self.collection_name = collection_name
        self.collection = None
        self.init_collection()

    def init_collection(self) -> None:
        """
        Idempotently initializes the Milvus collection schema with HNSW index,
        dynamic user_id routing, and loads the collection to RAM.
        """
        try:
            if utility.has_collection(self.collection_name):
                logger.info(f"Milvus collection '{self.collection_name}' already exists. Loading to memory.")
                self.collection = Collection(self.collection_name)
                self.collection.load()
                return

            logger.info(f"Creating new Milvus collection '{self.collection_name}'...")
            
            # 1. Define fields based on HLD/LLD
            fields = [
                FieldSchema(name="fact_id", dtype=DataType.VARCHAR, max_length=64, is_primary=True),
                # Dynamic Partition Key: routes user facts into internal virtual partitions by hash.
                # Avoids physical partition limits while ensuring partition-scoped queries.
                FieldSchema(name="user_id", dtype=DataType.VARCHAR, max_length=64, is_partition_key=True),
                FieldSchema(name="conversation_id", dtype=DataType.VARCHAR, max_length=64),
                FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=32),
                FieldSchema(name="statement", dtype=DataType.VARCHAR, max_length=1024),
                FieldSchema(name="importance", dtype=DataType.FLOAT),
                FieldSchema(name="fact_version", dtype=DataType.INT32),
                FieldSchema(name="embedding_ver", dtype=DataType.VARCHAR, max_length=32),
                FieldSchema(name="created_at", dtype=DataType.DOUBLE),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=settings.VECTOR_DIMENSION),
            ]

            schema = CollectionSchema(fields, description="User memory vectors with user_id partition key")
            self.collection = Collection(self.collection_name, schema)

            # 2. Build HNSW index on the vector field using Cosine similarity
            logger.info("Creating HNSW vector index on 'vector' field...")
            index_params = {
                "index_type": "HNSW",
                "metric_type": "COSINE",
                "params": {"M": 16, "efConstruction": 256}
            }
            self.collection.create_index(field_name="vector", index_params=index_params)

            # 3. Load the collection to prepare for ANN queries
            logger.info(f"Loading collection '{self.collection_name}' into RAM...")
            self.collection.load()
            logger.info(f"Milvus collection '{self.collection_name}' loaded successfully.")

        except Exception as e:
            logger.critical(f"Failed to initialize Milvus collection '{self.collection_name}': {e}")
            raise e

    def insert_facts(self, records: List[Dict[str, Any]]) -> List[str]:
        """
        Inserts semantic memory records into the Milvus collection.
        Slices operations into bulk batches using settings.MILVUS_BULK_INSERT_BATCH_SIZE.
        """
        if not records:
            return []

        # Enforce exact column ordering to match Milvus schema requirements
        field_names = [
            "fact_id",
            "user_id",
            "conversation_id",
            "category",
            "statement",
            "importance",
            "fact_version",
            "embedding_ver",
            "created_at",
            "vector"
        ]

        batch_size = settings.MILVUS_BULK_INSERT_BATCH_SIZE
        inserted_ids = []

        try:
            for start in range(0, len(records), batch_size):
                batch = records[start : start + batch_size]
                columns = {field: [] for field in field_names}

                for rec in batch:
                    for field in field_names:
                        val = rec.get(field)
                        if val is None:
                            # Apply sensible schema fallbacks
                            if field == "fact_version":
                                val = 1
                            elif field == "importance":
                                val = 0.0
                            elif field == "created_at":
                                val = datetime.now(timezone.utc).timestamp()
                            elif field == "embedding_ver":
                                val = settings.EMBEDDING_MODEL_VERSION
                            else:
                                val = ""
                        columns[field].append(val)

                # Format as list of lists and insert
                data_list = [columns[field] for field in field_names]
                res = self.collection.insert(data_list)
                inserted_ids.extend(res.primary_keys)

            # Flush to index new segments
            self.collection.flush()
            logger.info(f"Inserted {len(records)} records into Milvus.")
            return inserted_ids

        except Exception as e:
            logger.error(f"Error bulk-inserting vectors into Milvus: {e}")
            raise e

    def search_facts(
        self,
        user_id: str,
        query_vector: List[float],
        limit: int = 5,
        category: Optional[str] = None,
        consistency_level: str = "Bounded"
    ) -> List[Dict[str, Any]]:
        """
        Executes semantic nearest-neighbor search inside a user's partition.
        Uses COSINE metric type to rank similarity.
        """
        if self.collection is None:
            logger.error("Milvus collection is not initialized.")
            return []

        try:
            search_params = {
                "metric_type": "COSINE",
                "params": {"ef": 64}
            }

            # Partition key routing: must query by user_id scalar expression
            expr = f"user_id == '{user_id}'"
            if category:
                expr += f" && category == '{category}'"

            output_fields = [
                "fact_id",
                "user_id",
                "conversation_id",
                "category",
                "statement",
                "importance",
                "fact_version",
                "embedding_ver",
                "created_at"
            ]

            results = self.collection.search(
                data=[query_vector],
                anns_field="vector",
                param=search_params,
                limit=limit,
                expr=expr,
                output_fields=output_fields,
                consistency_level=consistency_level
            )

            hits = []
            if results and len(results) > 0:
                for hit in results[0]:
                    entity = {field: hit.entity.get(field) for field in output_fields}
                    entity["distance"] = hit.score  # Cosine similarity score
                    hits.append(entity)

            return hits

        except Exception as e:
            logger.error(f"Error performing Milvus vector search for user {user_id}: {e}")
            return []

    def delete_fact(self, user_id: str, fact_id: str) -> None:
        """
        Deletes a specific fact record.
        Uses primary key comparison as required by Milvus delete constraints.
        """
        try:
            # Milvus delete only supports expressions referencing the primary key (fact_id in ["val"])
            expr = f"fact_id in ['{fact_id}']"
            self.collection.delete(expr)
            logger.info(f"Deleted fact {fact_id} for user {user_id} in Milvus.")
        except Exception as e:
            logger.error(f"Error deleting fact {fact_id} in Milvus: {e}")
            raise e

    def delete_user_facts(self, user_id: str) -> None:
        """
        Deletes all facts for a given user from the collection.
        Queries the user's fact primary keys first, then deletes them bulk-style by primary key.
        """
        try:
            # Query all primary keys for user_id first using Strong consistency to get recent inserts
            results = self.collection.query(
                expr=f"user_id == '{user_id}'",
                output_fields=["fact_id"],
                consistency_level="Strong"
            )
            if not results:
                logger.info(f"No facts found to delete for user {user_id} in Milvus.")
                return

            fact_ids = [r["fact_id"] for r in results]
            import json
            expr = f"fact_id in {json.dumps(fact_ids)}"
            self.collection.delete(expr)
            logger.info(f"Deleted {len(fact_ids)} facts for user {user_id} in Milvus.")
        except Exception as e:
            logger.error(f"Error deleting user facts for {user_id} in Milvus: {e}")
            raise e
