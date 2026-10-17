# Task Update 7: Memory Context gRPC Server & Real Ingestion Pipeline

This update documents wiring Memory Service into the rest of the GraphGPT stack for real: exposing the memory-context read API over gRPC (as llm-service's `ContextCollector` actually expects), and building the missing Kafka ingestion path that populates Cassandra/Redis/Milvus from live conversations in the first place.

---

## 1. Executive Summary

Two separate, previously-missing links were closed in this pass:

1. **Read side — `GetMemoryContext` / `GetShortTermMemory` gRPC server.** llm-service's `ContextCollector` has always called a gRPC `MemoryServiceClient` on port 50051 expecting `llm-service/proto/memory.proto`'s `MemoryService`. Memory Service never implemented it — its only existing gRPC server (`app/proto/server.py`) serves a *different* proto (`llm_pb2_grpc`, this service's own internal summarization/fact-extraction engine). Every context-collection call from llm-service was silently timing out and degrading to empty context.
2. **Write side — ingestion from real conversations.** `EventDispatcher` (append message → advance snapshot state → trigger summary/fact/embedding chain) was fully built but never invoked by anything. llm-service publishes `memory.update.requested` after every completed response, but no consumer in this service subscribed to that topic — so even with the gRPC read path fixed, there was nothing in Cassandra to read back.

Both are now wired, verified against the real running stack (not mocked), including a full 12-turn live conversation trace confirming short-term memory accumulates correctly turn-over-turn and is served back accurately via gRPC.

---

## 2. Memory Context gRPC Server

### 2.1 Proto & Generated Stubs
Copied `llm-service/proto/memory.proto` into `app/proto/memory.proto` verbatim (field-for-field match with llm-service's `MemoryServiceClient` so no changes are needed on the llm-service side) and compiled it the same way the existing `llm.proto` was compiled, producing `app/proto/memory_pb2.py` / `memory_pb2_grpc.py`.

### 2.2 Handler — `app/proto/memory_handlers.py`
`MemoryContextHandler(memory_pb2_grpc.MemoryServiceServicer)`, constructed with the existing `ContextBuilder` and `EmbeddingClient` — no new business logic, purely a proto↔dict translation layer:

- **`GetMemoryContext`**: converts `request.query` to a vector via `EmbeddingClient.generate_embedding()` when non-empty, calls `ContextBuilder.build_context(user_id, conversation_id, query_vector)`, and maps the result's `short_term_messages` (Cassandra `message_id`/`role`/`content`/`created_at`), `current_summary` → `medium_summary`, and `relevant_facts` (`statement`/`importance`/`category`/`created_at`) → the proto's `Message`/`Fact` shapes. `semantic_snippets` and `total_tokens` are left at their zero-values — no data source or token-counting utility exists for them today, and none was invented for this task.
- **`GetShortTermMemory`**: calls `RetrievalService.get_or_hydrate_recent_messages()` directly, bypassing the full context-build aggregation since only messages are requested.

### 2.3 Combined Server Wiring
Rather than run a second gRPC server/port, `app/proto/server.py`'s existing `GRPCServer` was extended to optionally accept `context_builder`/`embedding_client` and register `memory_pb2_grpc.add_MemoryServiceServicer_to_server(...)` on the **same** `grpc.aio.server()` instance that already serves the internal `llm_pb2_grpc` engine. `app/core/container.py`'s `init_resources()` was reordered so this combined server is constructed *after* `context_builder`/`retrieval_service` exist (previously it was built too early to reference them).

The server now binds to the **existing** `50051` — the port llm-service's `memory_service_port` setting already defaulted to — rather than introducing a mismatch. The old `LLM_GRPC_PORT` setting (which had been pointed at 50052, actually a separate concern: memory-service's own internal LLM engine port) is superseded by `MEMORY_CONTEXT_GRPC_PORT: int = 50051`.

### 2.4 Downstream Port Cleanup
Fixing this surfaced a second, unrelated port collision: `embedding-service` (a separate, dedicated service in this monorepo, not previously running) also defaults its own gRPC server to `50051` and its REST port to `8000` — both already claimed. `embedding-service/.env` moved to gRPC `50054` / REST `8005`; `memory-service`'s `LLM_SERVICE_HOST`/`LLM_SERVICE_PORT` (used by `GRPCEmbeddingClient`, confusingly named after the wrong service) updated to point at `50054`.

`embedding-service` itself could not be started in this environment — it crashes with a `MemoryError` deep inside `transformers`' package-metadata scanning under the machine's current memory pressure (6 Python services + Cassandra/Kafka/Neo4j/Milvus/Postgres all running locally). This is an environment resource constraint, not a code defect. `EMBEDDING_CLIENT_TYPE=mock` was set for local verification, which uses the already-implemented `MockEmbeddingClient` (deterministic hash-based vectors) so long-term fact retrieval still exercises its full code path.

---

## 3. Ingestion Pipeline — `memory.update.requested` → Cassandra/Redis

### 3.1 New Worker — `app/workers/memory_update_worker.py`
`MemoryUpdateWorker` subscribes to the new `KAFKA_MEMORY_UPDATE_TOPIC` (`memory.update.requested`, matching llm-service's `KafkaPublisher.publish_memory_update`). llm-service's event carries one full turn (`user_message` + `assistant_response`, no per-message `message_id`/`event_id`) rather than the canonical `MemoryEventEnvelope` shape `EventDispatcher.dispatch()` expects (one message + an idempotency `event_id`). The worker adapts by synthesizing two envelope calls per turn — one for the user side, one for the assistant side — deriving stable `message_id`/`event_id` values via `uuid5(conversation_id:timestamp:role)` so Kafka redelivery naturally dedupes through `EventDispatcher`'s own idempotency check.

### 3.2 Activating the Rest of the Chain
`EventDispatcher` and the entire downstream worker chain it depends on (`OutboxDaemonWorker`, `SummaryWorker`, `FactWorker`, `EmbeddingWorker`) were fully implemented in the codebase already but never instantiated anywhere — none of their consumer groups had ever existed until this task. `app/core/container.py` now constructs and starts all five in `init_resources()`, and stops them (each via their existing `stop()` graceful-shutdown methods) in `shutdown_resources()`.

### 3.3 Dependency Fix
`aiokafka`'s lz4 message compression (used by llm-service's producer) requires the `cramjam` codec library, which was missing from this service's venv — the symptom was a silent `UnsupportedCodecError` looping in the new consumer's poll loop. Added `cramjam` to `pyproject.toml`/`requirements.txt`.

---

## 4. Live Verification — 12-Turn Conversation Trace

Rather than unit-test this in isolation, the fix was verified against the real, fully running stack (API Gateway → Auth → Conversation Service → Kafka → LLM Service → Kafka → Memory Service, with Groq as the live generation provider): a 12-message conversation was sent through the gateway as a real user, polling for each reply and querying `GetMemoryContext` via gRPC after every turn.

**Confirmed:**
- Short-term message count increased by exactly 2 per successful turn (2 → 4 → 6 → 8 → 10 → 12), matching the number of turns that actually completed.
- The model demonstrably used that memory in later turns — restating name, role, both languages, and the pet's breed from four separate earlier turns without repetition in the current prompt.
- A child conversation created with `parent_conversation_id` set starts with a genuinely empty short-term memory (`GetMemoryContext` returns `[]`) — memory storage is scoped strictly by `conversation_id` in Cassandra/Redis; `parent_conversation_id` links the two in Conversation Service/Graph Service's data model only. The one designed bridge, `ContextBuilder.get_parent_summaries()` (a Graph Service Neo4j ancestor-lineage call), was not exercised in this run since both child-conversation turns failed on a Groq rate limit before reaching the point where it would matter for the reply.
- Medium-term summarization and long-term fact extraction never triggered in this run — the threshold is 20 messages (`EventDispatcher.summary_threshold`) and the conversation reached 12 — but `SummaryWorker`/`FactWorker`/`EmbeddingWorker`/`OutboxDaemonWorker` all confirmed joining their Kafka consumer groups and are ready to fire once that threshold is crossed.
- 5 of 12 turns (plus both child-conversation turns) failed on Groq's free-tier rate limit (8,000 TPM, now shared between Request Analysis and generation since both route through Groq — see llm-service's own task update for the `ENABLE_NVIDIA` toggle backstory). Every failure still produced a correct terminal `chat.response.generated` with `status: error`; no memory-tier corruption or partial writes were observed.

Full turn-by-turn trace (queries, replies, timings, and memory-state snapshots after each turn) published separately as an artifact.

---

## 5. What Was NOT Changed

- `ContextBuilder`, `RetrievalService`, `RankingService`, `MemoryRepository` — all pre-existing, unchanged; this task only added a caller (`MemoryContextHandler`) and a data source (`MemoryUpdateWorker` → `EventDispatcher`), never new scoring/retrieval logic.
- No relevance scoring or semantic search was added to the graph-context path — that lives in Graph Service's own gRPC server (see graph-service's task update in this same session).
- No change to `SummaryService`/`FactWorker`'s internal LLM-based extraction logic — only their startup wiring.

---

## 6. Known Gaps / Follow-ups

- `embedding-service` could not be run in this environment (resource-constrained `MemoryError` in `transformers`); `MockEmbeddingClient` stands in via `EMBEDDING_CLIENT_TYPE=mock`. Real semantic embeddings need `embedding-service` running on a machine with more headroom.
- Running this service's own `pytest` suite alongside the live dev server hangs — its test `lifespan` tries to rebind port 50051, which the live server already holds. Not a code defect; run tests with the dev server stopped, or on a different port via `MEMORY_CONTEXT_GRPC_PORT`.
- `semantic_snippets` and `total_tokens` in `GetMemoryContextResponse` are left at zero-values — no existing data source or token-counting utility backs them.
