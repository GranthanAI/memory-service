# Task Update 4: Summarization, Fact Merging, Ranking, and Context Assembly (Phases 13-16)

This update details the design choices, architectural implementations, code structures, and verification logs completed for:
* **Phase 13: Incremental Summarization Service**
* **Phase 14: User Fact Merging Logic**
* **Phase 15: Decoupled Scoring & Ranking Engine**
* **Phase 16: Structured Context Builder & Retrieval Service**

---

## 1. Phase 13: Incremental Summarization Service

### 1.1 Objective & Algorithm
The Incremental Summarization Service processes conversation summaries incrementally to prevent unbounded LLM context growth. Rather than feeding the entire history, it retrieves the previous summary and the latest 20 message sliding window, reverses the message order to ensure chronological ordering (oldest first), and sends this combined context to the LLM.

### 1.2 gRPC Protocol Contract
We defined the gRPC service contracts in `app/proto/llm.proto` and compiled the python stubs.
```protobuf
syntax = "proto3";

package llm;

service LLMService {
    rpc GenerateSummary (SummaryRequest) returns (SummaryResponse);
    rpc ExtractFacts (FactExtractionRequest) returns (FactExtractionResponse);
    rpc GenerateEmbedding (EmbeddingRequest) returns (EmbeddingResponse);
}

message SummaryRequest {
    string previous_summary = 1;
    repeated Message new_messages = 2;
}

message Message {
    string role = 1;
    string content = 2;
}

message SummaryResponse {
    string summary_text = 1;
    string model_name = 2;
    string model_version = 3;
}

message FactExtractionRequest {
    string summary_text = 1;
    repeated Message messages = 2;
}

message Fact {
    string statement = 1;
    string category = 2;
    float importance = 3;
}

message FactExtractionResponse {
    repeated Fact facts = 1;
}

message EmbeddingRequest {
    string text = 1;
}

message EmbeddingResponse {
    repeated float vector = 1;
}
```

### 1.3 Service Implementation
Implemented in `app/services/summary_service.py`. It fetches data, chronologically orders the message list, invokes the stub under circuit-breaker protection, persists the results to Cassandra, and evicts stale Redis caches.

---

## 2. Phase 14: User Fact Merging Logic

### 2.1 Fact Merge Policy Matrix
The `LongMemoryService` applies the following merge policy to incoming facts relative to the user's category-scoped existing facts:

| Scenario | Policy | Action |
| :--- | :--- | :--- |
| Exact statement match (case-insensitive string match) | **Skip** | No database write (idempotent no-op) |
| Similarity < threshold (0.85) | **Insert** | Create new fact with `fact_id` and `fact_version = 1` |
| Similarity ≥ threshold, new importance ≥ existing | **Supersede** | Delete old fact (Cassandra + Milvus) and insert new fact with `fact_version = old_version + 1` |
| Similarity ≥ threshold, new importance < existing | **Ignore** | Discard the incoming fact (no-op) |

### 2.2 Cassandra Partition Key Constraint
Since `user_facts` uses a composite partition key `((user_id, category), fact_id)`, all queries and deletes must include both `user_id` and `category`. The `LongMemoryService` utilizes this structure to ensure that all lookups and deletion tombstone writes are correctly routed.

### 2.3 Milvus Vector Search Scopes
By declaring `user_id` as the dynamic partition key (`is_partition_key=True`), Milvus automatically partitions data by hashing `user_id`. Queries execute inside `user_id == '...'` expressions to limit ANN search spaces and guarantee high performance.

---

## 3. Phase 15: Decoupled Scoring & Ranking Engine

### 3.1 Scoring & Decay Equation
The `RankingService` calculates memory retrieval priority weights using:
$$\text{Score} = w_{\text{sim}} S_{\text{sim}} + w_{\text{rec}} e^{-\lambda t} + w_{\text{imp}} S_{\text{imp}}$$

Where:
* $S_{\text{sim}}$: Similarity score (Cosine metric, range `0.0` to `1.0`).
* $S_{\text{imp}}$: Fact importance (normalized to `0.0` - `1.0`).
* $t$: Time elapsed in days since fact creation. Capped at `0.0` to handle system clock drifts.
* $\lambda$: Exponential decay rate configured via `settings.RETRIEVAL_DECAY_RATE` (default `0.05`).
* Weights $w_{\text{sim}}, w_{\text{rec}}, w_{\text{imp}}$ load from settings (default `0.5`, `0.2`, `0.3`).

### 3.2 Legacy Metric Normalization
If facts are imported with legacy scoring systems (e.g. importance on a scale of `1` to `10`), the ranking service dynamically scales these values back into the standard `0.0` to `1.0` range.

---

## 4. Phase 16: Structured Context Builder & Retrieval Service

### 4.1 Concurrent Aggregations
To compile prompt contexts under sub-100ms response targets, the `ContextBuilder` aggregates memory components concurrently using `asyncio.gather`:
* Conversation snapshot (read-through)
* Current summary text (read-through)
* Recent message sliding window (read-through)
* User partition facts (Milvus search + Ranking service)
* Ancestor summaries (Graph Service HTTP query)

### 4.2 Graph Client Timeout & Degraded Metadata
To prevent slow network hops from blocking context builders:
* We wrap the HTTP query to `GraphClient` in a strict 200ms timeout (`settings.GRAPH_SERVICE_TIMEOUT_MS`).
* If the Graph Service times out or throws an error, the context builder catches the exception, logs a warning, returns an empty parent list, and returns a metadata flag `parent_summaries_available: false`.

---

## 5. Implementation Code

### 5.1 Long Memory Service (`app/services/long_memory_service.py`)
```python
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from app.core.config import settings
from app.repositories.cassandra_repository import CassandraRepository
from app.repositories.milvus_repository import MilvusRepository

logger = logging.getLogger("memory_service.services.long_memory_service")

class LongMemoryService:
    def __init__(self, cassandra_repo: CassandraRepository, milvus_repo: MilvusRepository):
        self.cassandra_repo = cassandra_repo
        self.milvus_repo = milvus_repo

    async def merge_user_facts(
        self, user_id: str, conversation_id: str, incoming_facts: List[Dict[str, Any]]
    ) -> Dict[str, int]:
        stats = {"inserted": 0, "superseded": 0, "ignored": 0, "skipped": 0}
        if not incoming_facts:
            return stats

        categories = set(f["category"] for f in incoming_facts)
        existing_facts_by_cat = {c: self.cassandra_repo.get_facts(user_id, c) for c in categories}
        threshold = settings.FACT_MERGE_SIMILARITY_THRESHOLD

        for fact in incoming_facts:
            category = fact["category"]
            statement = fact["statement"].strip()
            importance = fact["importance"]
            vector = fact["vector"]

            existing_list = existing_facts_by_cat.get(category, [])
            is_exact_match = any(o["statement"].strip().lower() == statement.lower() for o in existing_list)

            if is_exact_match:
                stats["skipped"] += 1
                continue

            hits = self.milvus_repo.search_facts(user_id=user_id, query_vector=vector, limit=1, category=category)
            closest_hit = hits[0] if hits else None

            if closest_hit is None or closest_hit["distance"] < threshold:
                fact_id = uuid.uuid4()
                cassandra_record = {
                    "user_id": user_id, "category": category, "fact_id": fact_id,
                    "conversation_id": conversation_id, "statement": statement, "importance": importance,
                    "fact_version": 1, "embedding_version": settings.EMBEDDING_MODEL_VERSION,
                    "created_at": datetime.now(timezone.utc), "updated_at": datetime.now(timezone.utc)
                }
                self.cassandra_repo.upsert_fact(cassandra_record)
                self.milvus_repo.insert_facts([{
                    "fact_id": str(fact_id), "user_id": user_id, "conversation_id": conversation_id,
                    "category": category, "statement": statement, "importance": importance,
                    "fact_version": 1, "embedding_ver": settings.EMBEDDING_MODEL_VERSION,
                    "created_at": datetime.now(timezone.utc).timestamp(), "vector": vector
                }])
                existing_list.append(cassandra_record)
                stats["inserted"] += 1
            else:
                old_importance = closest_hit["importance"]
                old_fact_id_str = closest_hit["fact_id"]
                old_fact_id = uuid.UUID(old_fact_id_str)
                old_version = closest_hit["fact_version"]

                if importance >= old_importance:
                    new_fact_id = uuid.uuid4()
                    self.cassandra_repo.delete_fact(user_id, category, old_fact_id)
                    self.milvus_repo.delete_fact(user_id, old_fact_id_str)

                    new_version = old_version + 1
                    cassandra_record = {
                        "user_id": user_id, "category": category, "fact_id": new_fact_id,
                        "conversation_id": conversation_id, "statement": statement, "importance": importance,
                        "fact_version": new_version, "embedding_version": settings.EMBEDDING_MODEL_VERSION,
                        "created_at": datetime.now(timezone.utc), "updated_at": datetime.now(timezone.utc)
                    }
                    self.cassandra_repo.upsert_fact(cassandra_record)
                    self.milvus_repo.insert_facts([{
                        "fact_id": str(new_fact_id), "user_id": user_id, "conversation_id": conversation_id,
                        "category": category, "statement": statement, "importance": importance,
                        "fact_version": new_version, "embedding_ver": settings.EMBEDDING_MODEL_VERSION,
                        "created_at": datetime.now(timezone.utc).timestamp(), "vector": vector
                    }])
                    existing_list = [f for f in existing_list if f["fact_id"] != old_fact_id]
                    existing_list.append(cassandra_record)
                    existing_facts_by_cat[category] = existing_list
                    stats["superseded"] += 1
                else:
                    stats["ignored"] += 1

        return stats
```

### 5.2 Ranking Service (`app/services/ranking_service.py`)
```python
import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from app.core.config import settings

class RankingService:
    @staticmethod
    def calculate_score(
        similarity: float, importance: float, created_at: Any, now: Optional[datetime] = None
    ) -> float:
        if isinstance(created_at, (int, float)):
            created_dt = datetime.fromtimestamp(created_at, timezone.utc)
        elif isinstance(created_at, datetime):
            created_dt = created_at
            if created_dt.tzinfo is None:
                created_dt = created_dt.replace(tzinfo=timezone.utc)
        else:
            raise ValueError("created_at must be a datetime or numeric timestamp.")

        ref_now = now or datetime.now(timezone.utc)
        if ref_now.tzinfo is None:
            ref_now = ref_now.replace(tzinfo=timezone.utc)

        delta_seconds = (ref_now - created_dt).total_seconds()
        t_days = max(0.0, delta_seconds / 86400.0)

        decay_rate = settings.RETRIEVAL_DECAY_RATE
        recency_score = math.exp(-decay_rate * t_days)

        w_sim = settings.RETRIEVAL_WEIGHT_SIMILARITY
        w_rec = settings.RETRIEVAL_WEIGHT_RECENCY
        w_imp = settings.RETRIEVAL_WEIGHT_IMPORTANCE

        final_score = (w_sim * similarity) + (w_rec * recency_score) + (w_imp * importance)
        return float(final_score)

    @classmethod
    def rank_facts(
        cls, facts: List[Dict[str, Any]], limit: Optional[int] = None, now: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        if not facts:
            return []

        scored_facts = []
        for fact in facts:
            similarity = fact.get("distance")
            if similarity is None:
                similarity = fact.get("similarity", 0.0)

            created_at = fact.get("created_at")
            importance = fact.get("importance", 0.0)
            if importance > 1.0:
                importance = importance / 10.0

            score = cls.calculate_score(similarity, importance, created_at, now)
            fact_copy = dict(fact)
            fact_copy["score"] = round(score, 4)
            scored_facts.append(fact_copy)

        scored_facts.sort(key=lambda x: x["score"], reverse=True)
        return scored_facts[:limit] if limit is not None else scored_facts
```

### 5.3 Context Builder Service (`app/services/context_builder.py`)
```python
import asyncio
import logging
from typing import Any, Dict, List, Optional
from app.core.config import settings
from app.clients.graph_client import GraphClient
from app.services.retrieval_service import RetrievalService

logger = logging.getLogger("memory_service.services.context_builder")

class ContextBuilder:
    def __init__(self, retrieval_service: RetrievalService, graph_client: GraphClient):
        self.retrieval_service = retrieval_service
        self.graph_client = graph_client

    async def get_parent_summaries(self, conversation_id: str) -> tuple[List[Dict[str, Any]], bool]:
        timeout_seconds = settings.GRAPH_SERVICE_TIMEOUT_MS / 1000.0
        try:
            async with asyncio.timeout(timeout_seconds):
                ancestors = await self.graph_client.get_ancestors(conversation_id)
                return ancestors, True
        except (asyncio.TimeoutError, Exception) as e:
            logger.warning(
                f"Graph Service unavailable for {conversation_id}: {e}. "
                f"Falling back to current summary only."
            )
            return [], False

    async def build_context(
        self, user_id: str, conversation_id: str, query_vector: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        snapshot_task = self.retrieval_service.get_or_hydrate_snapshot(conversation_id)
        summary_task = self.retrieval_service.get_or_hydrate_summary(conversation_id)
        messages_task = self.retrieval_service.get_or_hydrate_recent_messages(
            conversation_id, limit=settings.SHORT_TERM_MESSAGE_LIMIT
        )
        parents_task = self.get_parent_summaries(conversation_id)

        if query_vector:
            facts_task = self.retrieval_service.retrieve_relevant_facts(
                user_id=user_id, query_vector=query_vector, limit=settings.RETRIEVAL_TOP_K_FACTS
            )
        else:
            facts_task = asyncio.sleep(0, result=[])

        snapshot, summary, messages, parents_result, facts = await asyncio.gather(
            snapshot_task, summary_task, messages_task, parents_task, facts_task
        )
        parent_summaries, parents_available = parents_result

        resolved_user_id = user_id
        if snapshot:
            if not resolved_user_id:
                resolved_user_id = snapshot.get("user_id")
            elif snapshot.get("user_id") != resolved_user_id:
                logger.warning(
                    f"User ID mismatch for conversation {conversation_id}: "
                    f"snapshot user_id={snapshot.get('user_id')}, requested user_id={resolved_user_id}"
                )

        return {
            "conversation_id": conversation_id,
            "user_id": resolved_user_id,
            "current_summary": summary or "",
            "short_term_messages": messages,
            "parent_summaries": parent_summaries,
            "relevant_facts": facts,
            "metadata": {
                "parent_summaries_available": parents_available,
                "facts_retrieved_count": len(facts)
            }
        }
```

---

## 6. Verification Results & Regression Metrics

All test suites were executed sequentially. This includes the new unit tests and integration tests written for context builders, long term memory services, and decay parameters.

* **Summary Service tests** (4/4 passed)
* **Fact Merging tests** (5/5 passed)
* **Decay and Scoring tests** (4/4 passed)
* **Context Assembly tests** (5/5 passed)
* **Regression suites** (63/63 passed)

Total active database tests: **81/81 passed successfully** with zero errors.

```text
tests/unit/test_context_builder.py::test_context_builder_success PASSED
tests/unit/test_context_builder.py::test_context_builder_graph_timeout_fallback PASSED
tests/unit/test_context_builder.py::test_context_builder_graph_exception_fallback PASSED
tests/unit/test_context_builder.py::test_retrieval_service_read_through_cache_miss PASSED
tests/integration/test_context_builder_integration.py::test_context_builder_read_through_hydration_integration PASSED
tests/unit/test_long_memory_service.py::test_merge_user_facts_skips_on_exact_match PASSED
tests/unit/test_long_memory_service.py::test_merge_user_facts_inserts_on_low_similarity PASSED
tests/unit/test_long_memory_service.py::test_merge_user_facts_ignores_on_lower_importance PASSED
tests/unit/test_long_memory_service.py::test_merge_user_facts_supersedes_on_higher_importance PASSED
tests/integration/test_long_memory_service_integration.py::test_long_memory_service_fact_merge_policy_integration PASSED
tests/unit/test_ranking_service.py::test_calculate_score_recency_decay_exact_values PASSED
tests/unit/test_ranking_service.py::test_calculate_score_accepts_timestamp_float PASSED
tests/unit/test_ranking_service.py::test_rank_facts_sorting_and_limits PASSED
tests/unit/test_ranking_service.py::test_rank_facts_legacy_and_edge_normalization PASSED

======================== 81 passed in 125.43s ========================
```
