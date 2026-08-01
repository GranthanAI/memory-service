# GraphGPT Memory Service
## High Level Design (HLD)

**Version:** 4.0 (Production-Grade, Battle-Tested Architecture)  
**Target Scale:** 100M Users, 1M+ Writes/sec (Kafka), 99.99% Availability  
**Last Updated:** 2026-08-02  

---

# Chapter 1. Purpose & Philosophy

The Memory Service is GraphGPT's AI Memory Engine — the system responsible for converting raw conversational data into structured, queryable, high-quality memory representations that power personalized AI responses.

It is not a message store. The Conversation Service owns raw messages. The Memory Service owns **derived cognitive state** — what the AI remembers about a user, not what was literally said.

## 1.1 Design Principles

1. **Cassandra is the Source of Truth.** All persistent state (snapshots, summaries, facts, outbox, idempotency) is written to Cassandra first, durably. Redis holds nothing that cannot be reconstructed from Cassandra or Kafka replay.
2. **Redis is a Hot Cache Only.** Redis acceleration is transient. Cache misses fall back to Cassandra. Redis eviction never causes data loss, only temporary latency spikes.
3. **Milvus is for Vectors Only.** All semantic search and long-term memory retrieval happens through Milvus. Milvus stores no source-of-truth business data.
4. **Kafka Decouples All Heavy Computation.** Summarization, fact extraction, and embedding generation are never done inline. They are dispatched as async worker jobs, with Kafka as the transport layer.
5. **Every Write is Idempotent.** All event ingestion verifies the `event_id` against a persistent Cassandra idempotency table before any mutation. 7-day TTL covers all realistic Kafka replay windows.
6. **Every Job is Claimed Before Processing.** Outbox workers use Cassandra Lightweight Transactions (LWT) to atomically transition jobs from `PENDING → PROCESSING` before acting. This prevents duplicate publishing.
7. **Every Lock Has an Owner Token.** Redis distributed locks use a unique `lock_value` (UUID) so that only the holder can release the lock. Watchdog heartbeats extend the TTL dynamically during slow LLM calls.

---

# Chapter 2. Responsibilities

## 2.1 This Service Owns
| Memory Layer | Description | Primary Store |
| :--- | :--- | :--- |
| **Short-Term Memory** | Latest N messages window for active context | Redis (hot), Cassandra (durable), Kafka (transport) |
| **Medium-Term Memory** | Conversation summaries across current + parent threads | Cassandra (durable), Redis (hot) |
| **Long-Term Memory** | Structured user facts, preferences, habits | Cassandra (durable), Milvus (vectors) |
| **Semantic Memory** | Conceptual associations, entity graphs | Milvus (vectors) |
| **Conversation Snapshots** | Lightweight state tracking (count, version, state machine) | Cassandra (durable), Redis (hot) |
| **Outbox Jobs** | Reliable async task delivery to Kafka | Cassandra |
| **Idempotency Registry** | Processed event deduplication | Cassandra (7-day TTL) |
| **Retry / DLQ Registry** | Failed job tracking and retry scheduling | Cassandra |

## 2.2 This Service Does NOT Own
| Resource | Owner |
| :--- | :--- |
| Raw conversation message history | Conversation Service |
| Conversation graph traversals | Graph Service |
| AI prompt template assembly | LLM Service |
| Authentication tokens | Auth Service |
| API gateway rate limiting | API Gateway |

---

# Chapter 3. End-to-End Architecture

```text
                         ┌─────────────────────────┐
                         │    Conversation Service   │
                         └────────────┬────────────┘
                                      │ Kafka Topic
                         chat.message.created
                         chat.response.completed
                                      │
                                      ▼
                         ┌────────────────────────────┐
                         │   Memory Event Consumer     │
                         │   (Lightweight, Idempotent) │
                         └────────────┬───────────────┘
                                      │
                         ─────── Idempotency Check ──────
                                      │ (event_id NOT in Cassandra processed_events)
                                      ▼
                         ┌────────────────────────────┐
                         │   Snapshot Builder          │
                         └────────────┬───────────────┘
                                      │
               ──── Cassandra Logged Batch Mutation ────
                    (NOT full ACID, but atomic delivery)
                          │                  │
                          ▼                  ▼
                 UPDATE snapshot      INSERT outbox_job (PENDING)
                 (message_count,      + INSERT processed_event
                  state, version)     (idempotency row)
                          │
                          ▼
                 Redis Hot Cache Invalidation
                 (DELETE snapshot:{id}, recent:{id}, summary:{id})
                          │
                          ▼
                 ┌─────────────────────────────────┐
                 │        Outbox Daemon Worker       │
                 │  Poll → LWT Claim (PROCESSING)   │
                 │  Publish to Kafka → DELETE row   │
                 └────────────┬────────────────────┘
                              │
              ┌───────────────┴──────────────────────┐
              │                                      │
              ▼                                      ▼
    memory.summary.request              memory.delete.request
              │                                      │
              ▼                                      ▼
     Summary Worker                         Delete Worker
     (LWT claim → gRPC → Cassandra)        (Milvus + Cassandra)
              │
       ┌──────┴──────┐
       ▼             ▼
  memory.fact.request    Update summary in
       │                 Cassandra + Invalidate Redis
       ▼
   Fact Worker
   (LWT claim → gRPC → Cassandra)
       │
       ▼
  memory.embedding.request
       │
       ▼
  Embedding Worker
  (LWT claim → Bulk Insert Milvus)

                         Context Retrieval Path:
                         ┌──────────────────────────┐
                         │    LLM Service (caller)   │
                         └────────────┬─────────────┘
                                      │ HTTP/gRPC
                                      ▼
                         GET /internal/memory/context
                                      │
                         ┌────────────┴──────────────┐
                         │  Retrieval + Ranking       │
                         └────────────┬──────────────┘
                                      │
              ┌───────────────┬───────┴────────────────────┐
              ▼               ▼                            ▼
         Redis Cache     Graph Service               Milvus Search
         (snapshot,      (Lineage API)           (HNSW + scalar filter
          summary)           │                     user_id=X, top_k=10)
              │               ▼                            │
              │         Parent Summaries                   │
              └───────────────┬────────────────────────────┘
                              │
                    Ranking & Score Assembly
                    (Recency × Importance × Similarity)
                              │
                              ▼
                   Final Context JSON Response
```

---

# Chapter 4. Storage Strategy

## 4.1 Cassandra — Primary Source of Truth

Cassandra is the authoritative persistence layer for all derived memory states. It is chosen because:
* Wide-column storage handles time-series snapshot updates at millions of writes/sec.
* Keyspace-level replication provides multi-DC durability.
* LWT (Lightweight Transactions) enable safe job claiming by concurrent workers without distributed locking.
* TTL is native to Cassandra rows, simplifying idempotency expiry.

**Cassandra is the only system in this service from which state can be recovered if Redis or Milvus data is lost.**

## 4.2 Redis — Low-Latency Hot Cache

Redis stores nothing that cannot be rebuilt from Cassandra. Its purpose is purely latency optimization:
* Sub-millisecond snapshot lookups during context assembly.
* Compressed summary caching to avoid Cassandra reads on every LLM call.
* Recent message window (latest 20) for fast short-term context access.
* Distributed locking with UUID ownership tokens and watchdog heartbeats.

If Redis is lost: reads fall back to the Cassandra tables (`conversation_snapshots`, `conversation_summaries`, `conversation_recent_messages`). Cache rehydration is automatic on read. Kafka is not used for recovery — Kafka retention (7 days) is too short for long-lived conversations.

## 4.3 Milvus — Vector Index

Milvus stores only vector embeddings and lightweight metadata fields. It does not store business logic or text content. All text is stored in Cassandra and referenced via `fact_id` / `memory_id`.

At scale (100M users), Milvus is configured with `user_id` as a dynamic partition key. For very large deployments where per-user partitioning hits limits, the strategy falls back to scalar field filtering (`expr="user_id == X"`), which is supported natively by Milvus HNSW search.

---

# Chapter 5. Reliable Outbox Pattern

> **Important Note on Cassandra Batches**
> The term "transactional" in this document refers to the Cassandra Logged Batch pattern, which provides atomic delivery of multiple mutations — meaning either all mutations are applied or the batch coordinator retries delivery until all are applied. It does NOT provide full ACID transaction semantics (no isolation, no rollback). The Outbox pattern is designed with this constraint in mind: idempotency keys ensure safe retries.

## 5.1 Write Flow
1. Event arrives at the dispatcher. `event_id` is checked against Cassandra `processed_events`.
2. If already processed: skip (idempotent).
3. If new: execute a Cassandra **Logged Batch** containing:
   * `INSERT INTO conversation_snapshots (...)` — lightweight metadata state row
   * `INSERT INTO conversation_recent_messages (...)` — individual message row for durable window
   * `INSERT INTO outbox_jobs (..., status='PENDING')` — async task descriptor
   * `INSERT INTO processed_events (event_id, ...)` — idempotency marker
4. Delete oldest messages beyond the window limit asynchronously (background cleanup worker).
5. On batch delivery: invalidate Redis cache keys (`snapshot:`, `recent:`, `summary:`) for this `conversation_id`.

## 5.2 Outbox Worker Claiming Flow
To prevent duplicate publishing by concurrent workers, the outbox worker uses Cassandra LWT:
```
PENDING → (LWT Compare-And-Set) → PROCESSING → Kafka Publish → DELETE
```
If publish fails: the row remains `PROCESSING` and a background reaper reclaims stale `PROCESSING` rows (older than N minutes) back to `PENDING`.

---

# Chapter 6. Snapshot Design Philosophy

The snapshot stored in Cassandra contains **only lightweight state metadata** — no raw message content:
* `conversation_id` — partition key
* `message_count` — current total
* `state` — state machine enum
* `summary_version` — for versioned cache invalidation
* `fact_version` — for fact change tracking
* `snapshot_version` — monotonically increasing mutation counter
* `last_summary_message_id` — pointer to which message triggered last summary
* `updated_at` — for ordering and cache hydration

**Recent message storage follows a three-tier pattern:**
1. **Redis** `recent:{conversation_id}` — hot cache, sub-millisecond reads.
2. **Cassandra** `conversation_recent_messages` — durable fallback. Individual rows per message, ordered by `created_at DESC`. Only the latest 20 rows are retained; older rows are pruned after each summary cycle.
3. **Conversation Service** — ultimate owner of raw message history.

Kafka is the transport layer only. Recovery of recent messages never depends on Kafka replay, because Kafka retention (typically 7 days) is far shorter than a conversation's lifespan (potentially months).

On Redis cache miss: the service reads the last 20 rows from `conversation_recent_messages` in Cassandra, repopulates Redis, and continues. No Kafka seek is required.

---

# Chapter 7. State Machine

```text
ACTIVE  ──────────────────────────────────────────────────────────►  ACTIVE
  │                                                                    ▲
  │ (threshold reached)                                                │
  ▼                                                                    │
SUMMARY_PENDING                                                        │
  │                                                                    │
  │ (outbox job claimed by summary worker)                             │
  ▼                                                                    │
SUMMARIZING                                                            │
  │                                                                    │
  │ (LLM responds, summary written to Cassandra)                      │
  ▼                                                                    │
FACT_PENDING                                                           │
  │                                                                    │
  │ (fact worker claims job)                                           │
  ▼                                                                    │
EXTRACTING_FACTS                                                       │
  │                                                                    │
  │ (facts written to Cassandra)                                       │
  ▼                                                                    │
EMBEDDING_PENDING                                                      │
  │                                                                    │
  │ (embedding worker claims job, bulk insert to Milvus)              │
  ▼                                                                    │
READY ───────────────────────────────────────────────────────────────►┘
  │
  │ (any step fails 5 times)
  ▼
FAILED  ──► DLQ + Cassandra retry_jobs row (status=FAILED)
```

---

# Chapter 8. Failure Recovery Runbook

## 8.1 Redis Cache Loss
1. Cache misses are automatically handled by fallback reads to Cassandra.
2. Recent message windows (if missing in Redis): query the last 20 rows from `conversation_recent_messages` in Cassandra. Repopulate Redis. **No Kafka replay needed.**
3. Snapshots and summaries: read from `conversation_snapshots` and `conversation_summaries` Cassandra tables.
4. A cache rebuilder background task can sweep all active snapshots and repopulate Redis in bulk.

## 8.2 Worker Crash Mid-Processing
* Cassandra `outbox_jobs` rows remain in `PROCESSING` state.
* A background reaper task scans for `PROCESSING` rows with `claimed_at < NOW() - 5min` and resets them to `PENDING` for retry.
* Because all workers check idempotency before acting, replays are safe.

## 8.3 Kafka Consumer Group Lag
* Scale out consumer pods. Kafka partition keys (`conversation_id`) guarantee ordered processing per conversation.
* Because all writes are idempotent, there are no risks from duplicate event delivery.

## 8.4 Milvus Data Loss
* Milvus stores only vector embeddings. Source data (fact text) is in Cassandra.
* A background rebuild script reads all `user_facts` from Cassandra, regenerates embeddings via LLM Service gRPC, and bulk inserts into Milvus.

## 8.5 Full Service Downtime (6+ hours)
1. Start the service.
2. Kafka consumer group begins replaying from last committed offset.
3. Idempotency table in Cassandra prevents duplicate processing of already-committed events.
4. Outbox reaper reclaims any stale `PROCESSING` rows.
5. Cache rebuilder repopulates Redis from Cassandra.
6. Service enters healthy state.

---

# Chapter 9. Metrics Surface

| Metric Name | Type | Purpose |
| :--- | :--- | :--- |
| `memory.kafka.consumer.lag` | Gauge | Kafka consumer lag per topic |
| `memory.summary.queue.size` | Gauge | Pending summary jobs |
| `memory.fact.queue.size` | Gauge | Pending fact extraction jobs |
| `memory.embedding.queue.size` | Gauge | Pending embedding jobs |
| `memory.dlq.size` | Gauge | Jobs in DLQ |
| `memory.redis.hit.total` | Counter | Redis cache hits |
| `memory.redis.miss.total` | Counter | Redis cache misses |
| `memory.redis.lock.wait.seconds` | Histogram | Redis lock acquisition wait time |
| `memory.milvus.search.qps` | Counter | Milvus similarity search query count |
| `memory.context.build.seconds` | Histogram | End-to-end context assembly time |
| `memory.outbox.pending.total` | Gauge | Outbox table pending jobs count |
| `memory.retry.pending.total` | Gauge | Retry table pending jobs count |
| `memory.grpc.channel.errors` | Counter | gRPC channel-level errors |

---

# Chapter 10. Technology Stack

| Component | Technology | Version | Role |
| :--- | :--- | :--- | :--- |
| Framework | FastAPI | 0.110+ | REST Interface |
| Primary Database | Apache Cassandra | 4.1 | Persistent Source of Truth |
| Cache | Redis | 7.2 | Hot cache only |
| Vector Database | Milvus | 2.3 | Vector indexing, HNSW |
| Message Broker | Apache Kafka | 3.9 | Async worker dispatch |
| AI Inference | gRPC async pool | v1.65+ | LLM + Embedding calls |
| Graph Database | Neo4j | 5.15 | Lineage (via Graph Service) |
| Metrics | Prometheus + Grafana | — | Observability |
| Deployment | Docker + Kubernetes | — | Container runtime |

---

# Chapter 11. Service Communication

```text
                              Conversation Service
                                       │
                                     Kafka
                                       │
                                       ▼
                               Memory Service (this)
            ┌──────────────────────────┼────────────────────────┐
            ▼                          ▼                         ▼
       Cassandra DB             Redis (cache)              Milvus (vectors)
            │                          │
            │                          │
            └─────────── Graph Service ─────────── LLM Service (gRPC Pool)
```

---

# Chapter 12. Context API Contract

```text
GET /internal/memory/context
```

**Request Parameters:**
```json
{
  "conversation_id": "string",
  "user_id": "string",
  "query": "string",
  "top_k_facts": 10
}
```

> `query` is the raw text used for semantic retrieval. The Memory Service internally calls the LLM Service gRPC pool to generate the embedding. Callers are never required to know about or pass embedding vectors. This decouples callers from the embedding model version.

**Response Body:**
```json
{
  "short_term": [ "...messages..." ],
  "summary": "string (latest compressed summary text)",
  "parent_summaries": [ "..." ],
  "long_term_facts": [ "...ranked fact objects..." ],
  "semantic_results": [ "...milvus hits..." ],
  "context_version": "integer",
  "built_at": "ISO8601"
}
```

---
**End of High-Level Design — v4.1**
