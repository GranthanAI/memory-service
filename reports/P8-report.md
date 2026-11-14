# P8: memory-service, report

- **Commits:** P8a `51180ed` (usage.metered for every LLM call) · P8b `5cef1de` (memory_facts plan limit; this commit also carries the user.deleted service, worker and tests) · P8c (docs and this report). **Pushed:** no (owner pushes).
- **Contracts touched:** plan 4.3 (topics), 4.4.5 (`usage.metered`), 4.4.6 (`user.deleted`), 4.4.7 (`entitlements.changed`), 4.5 (GetEntitlements deadline, breaker), 4.7 (entitlements cache), 4.8 (`memory_facts`), 4.9 (nothing unmetered). Deviations: below.
- **Tests:** see the list below (unit tests, one file at a time). Exit check: see below.
- **Exit check:** see the "Exit check" section.
- **Decisions needed:** see below.
- **Not done:** see below.

## Every paid LLM call site (and its feature)

All paid calls reach the provider through ONE place, `GroqClient.chat_completion` (via `GroqProvider.generate` and `LLMManager.generate_with_retry`), which reports the call to `usage_metering`. The callers are the five `LLMService` methods:

| `LLMService` method | Reached from | Job (worker) that meters it | Feature |
| --- | --- | --- | --- |
| `summarize` | `SummaryService.process_incremental_summary` (two attempts: first draft and a reviewed redraft) | `SummaryWorker` | `memory.summary` (count trigger), `memory.compaction` (reason pressure / idle / stale_read), `memory.branch_summary` (request bounded by `upto_message_id`) |
| `consolidate_summary` | `SummaryService._consolidate` (a summary over its token cap is rewritten shorter) | `SummaryWorker` | same job as above |
| `extract_anchors` | `SummaryService` (judgement-based anchor kinds) | `SummaryWorker` | same job as above |
| `extract_facts` | `FactWorker._extract_facts` | `FactWorker` | `memory.facts` |
| `judge_fact_relation` | `LongMemoryService.merge_user_facts` (does a new fact correct an old one?) | `EmbeddingWorker` (the merge) | `memory.facts` |
| `summarize`, `extract_facts` | `app/api/internal/llm.py` and `app/proto/handlers.py` (internal HTTP / gRPC callers) | none: a paid call outside a job is counted in `memory_unmetered_calls_total` and logged | (no caller identity: a bug to fix if it is ever used) |

`memory.update` is the request-level ingestion (`MemoryUpdateWorker`); it makes no LLM call itself, so it writes no event. It stamps the `request_message_id` of the `memory.update.requested` event it ingests on the summary and fact requests it triggers (a ContextVar for the dispatcher path, a payload field for the explicit-memory fast path), so billing can link cost to the message.
The mock provider (`MockLLMProvider`) never calls `record_call`, so it is never metered. The idle-compaction worker and the branch-summary worker make no model call themselves: they enqueue a summary request that `SummaryWorker` meters.

## Delivery guarantee per worker

A job writes its event in a `finally` (so a job that fails after calling the model still reports its cost), through the Cassandra outbox (`OutboxDaemonWorker`: at-least-once to Kafka). `event_id = uuid5(job key)` with job key = `<feature>:<topic>:<partition>:<offset>` of the Kafka message being processed, so billing (which dedupes usage on `event_id`) counts a redelivered job once.

| Worker | Guarantee |
| --- | --- |
| `SummaryWorker`, `FactWorker`, `EmbeddingWorker` | at-least-once publication, exactly-once effect in billing (deterministic id). A re-run of the SAME Kafka message re-calls the model but is billed once (the first run's calls); a retry through `retry_jobs` is a new message and a new id. A crash after the model call and before the `finally` loses that attempt's cost (no event), never double counts. |
| `BranchSummaryWorker`, `IdleCompactionWorker`, `MemoryUpdateWorker` | no model call of their own: nothing to meter; they pass the request id on. |
| usage outbox row | the row id is the event id; the outbox publishes with the user id as the Kafka key. |

## Settings (each also in `.env.example` and the local `.env`)

`BILLING_GRPC_TARGET` (127.0.0.1:50061: billing's gRPC, not this service's own 50051), `BILLING_SERVICE_URL`, `BILLING_SERVICE_TOKEN`, `BILLING_LIMITS_ENABLED` (false by default; true in the shared dev `.env`), `ENTITLEMENTS_CACHE_SECONDS`.

## memory_facts: how the count is kept

No separate counter. The merge already reads the user's ACTIVE facts from Cassandra (`_existing_by_key`); its size is the count the cap is checked against, so there is nothing to drift and nothing to reconcile for counts. Cassandra/Milvus drift is repaired by the existing `LongMemoryService.reconcile_user_facts`. A fact that replaces another keeps the count and stays allowed; at the cap, new facts are skipped (`memory_fact_cap_reached`, `memory_fact_cap_total`); existing facts stay readable and editable; deleting frees a slot; no replacing of low-importance facts. `GET /memory/facts` adds `facts_used` and `facts_limit` (`null` while limits are off, `-1` unlimited).

## Deviations and notes

1. **P8b and P8c share a commit** (`5cef1de`): the user.deleted service and worker were written together with the entitlements consumer; P8c adds docs and this report.
2. **Kafka topics created in the shared Kafka:** memory-service's startup validation refused to start because `chat.response.completed`, `memory.delete.request` and `memory.dlq` did not exist (nothing had created them). I created them (1 partition, replication 1) with the existing container's `kafka-topics.sh`.
3. **Local `.env` changes for the shared stack** (git-ignored): `HTTP_PORT=8004` (what the gateway expects), `KAFKA_BOOTSTRAP_SERVERS`, `MILVUS_HOST`, `REDIS_URL` on `127.0.0.1`, `GRAPH_SERVICE_URL=http://127.0.0.1:8000` (it pointed at auth's port), and **`EMBEDDING_CLIENT_TYPE=mock`**: embedding-service is not part of the running shared stack, so fact vectors in the exit check are the deterministic mock vectors, not real embeddings.
4. **Per-conversation data on user.deleted:** summaries and the durable window of the conversations memory-service can attribute to the user (a fact's source conversation, the idle-compaction index) are purged at once through the existing `ConversationDeleteWorker.purge`; the rest are removed by the `conversation.deleted` events conversation-service publishes when it purges the user's conversations after its retention period (30 days). There is no user-to-conversations index in this service.
5. PyYAML is not installed in this environment, so the "bundled Free limits equal plans.yaml" test reads `base_limits` by hand.

## Tests run (`.venv\Scripts\python -m pytest tests/unit/<file>.py -q`, one file at a time)

| File | Result | What it covers |
| --- | --- | --- |
| test_usage_metering.py (new) | 16 passed | one event per job with exact provider tokens, several calls in one event, timed-out retries counted, estimated usage flagged, deterministic event_id, mock never metered, failed job still reports, no-user job not billed, outbox failure never breaks a job, health probe is a system call, codecs, catalog warning |
| test_entitlements_cap.py (new) | 16 passed | billing client (300 ms deadline, token, breaker), Redis cache (hit, miss, stale, Free fallback, shared), bundled Free limits = plans.yaml, the cap (cap-1 stores, cap refuses, delete frees, shrink keeps data, -1, flag off, per-user), `facts_used`/`facts_limit`, entitlements.changed consumer deduped |
| test_user_deletion.py (new) | 4 passed | facts (active and retired), vectors, settings, idle index and attributable conversations removed, others untouched, replay no-op, failed delete retried, consumer deduped |
| test_long_memory_service.py, test_long_term_facts.py, test_memory_controls.py, test_api_endpoints.py | 4, 41, 19, 16 passed | touched code paths (merge, controls, facts API) |
| test_background_workers.py, test_events_pipeline.py, test_compaction_trigger.py, test_llm_business_layer.py, test_llm_foundation.py, test_llm_production_hardening.py, test_pipeline_integrity.py, test_outbox_worker.py, test_memory_service.py, test_consumer_dead_letter.py, test_startup_validation.py | 8, 6, 9, 2, 8, 3, 20, 3, 5, 11, 8 passed | workers, dispatcher, manager and startup code I changed |
| test_migrations_and_container.py | 9 passed | container wiring with the new worker |
| test_api_grpc.py | 1 passed, **3 errors** | the errors are `Failed to bind to address [::]:50051`: the running memory-service (shared stack) holds that port. Not a code failure; it needs port 50051 free. |

Not run: the integration tests (they start containers) and the full suite.

## Exit check (shared stack, 15 Nov 2026)

Setup: memory-service run from its own `.env` (git-ignored; `BILLING_SERVICE_TOKEN` = the memory value, `BILLING_LIMITS_ENABLED=true`, `HTTP_PORT=8004`) on port 8004, startup passed (codecs OK). A throwaway Free user registered through the gateway sent 22 messages in ONE conversation (paced 25 s apart for Groq's free-tier token limit); llm-service answered and published `memory.update.requested` with the `request_message_id`; memory-service ingested the turns, summarised and extracted facts.

```
billing.usage_events for this user, kind = 'aux':  20 rows | 535,593 micro-INR in total | min 4,168 (every row non-zero) | 21,791 in / 13,345 out tokens
usage.metered events on Kafka from memory-service for this user:  20  ->  memory.summary 10, memory.facts 10
   each: producer memory-service, schema 1.0, fixed_cost_micro_inr 0, calls from the provider's own usage (0 estimated of 32 calls),
         event_id = uuid5(job): all 20 distinct, source_id in billing = that event_id
   linked to the conversation's requests: 20 of 20 carry a request_message_id that is one of the 22 sent (10 distinct requests)
GET /api/v1/memory/facts (gateway): {'facts_used': 5, 'facts_limit': 50, 'memory_enabled': True}
   e.g. "User's favorite colour is teal.", "User lives in Pune.", "User works as a data engineer."
```

(billing keeps `source_id`, tokens, cost and provider cost per row, not the feature name or the request id; those are on the event itself, shown above.)

Delete the user: `DELETE /api/v1/auth/me` -> 204; `user.deleted` reached memory-service:

```
                                    BEFORE                      AFTER (within seconds)
Cassandra user_facts (5 categories)  2 0 0 3 0  (5 facts)         0 0 0 0 0
Milvus user_memory_vectors           5 vectors                    0
user_memory_settings / tombstones    -                            0 / 0
Redis ent:* keys, compact:idle       -                            0, 0
conversation snapshot (memory)       present                      0 (purged through the existing conversation-delete path)
login afterwards                                                  401
```

One finding during the run, fixed before the final restart: a startup/health `ping` is a paid call made by the service itself and was counted as an unmetered call. It is now an explicit "system call" (`memory_system_llm_calls_total`, not billed: no user), so `memory_unmetered_calls_total` stays reserved for real bugs. Note the readiness probe still makes that few-token call on every probe (existing behaviour).

## Decisions needed

1. **Health probes are paid calls** (existing behaviour): every `/health` readiness probe asks the model for 5 tokens. Cache the result or probe less often?
2. **Embedding client in the shared stack:** the exit check ran with `EMBEDDING_CLIENT_TYPE=mock` because embedding-service is not running. Switch back to `grpc` once it is up (the `.env` line is the only change).
3. **Internal HTTP / gRPC LLM endpoints** (`summarize`, `extract_facts`) have no user: if anything calls them they are logged as unmetered. Do they have a caller?

## Not done

- The remaining per-conversation memory data of a deleted user whose conversations this service cannot attribute (a conversation with no stored fact) is removed only when conversation-service's purge publishes `conversation.deleted` (30 days).
- Unit tests only; no integration test file was added (they start containers).

## Process state

Started by me and left running (shared stack): memory-service on 8004 (HTTP) and 50051 (its own gRPC), PID in `%TEMP%\memdev\pid`. Restart: `.venv\Scripts\python -m uvicorn app.main:app --host 0.0.0.0 --port 8004` from the repo (everything else comes from `.env`).
