
## Billing integration (cost metering, memory_facts cap, user deletion)

All of it is **off or passive by default**: `BILLING_LIMITS_ENABLED=false` keeps today's behaviour for facts, and usage events cost nothing when no paid call is made.

| Setting | Default | Meaning |
| --- | --- | --- |
| `BILLING_GRPC_TARGET` | `127.0.0.1:50061` | billing's gRPC (**not** this service's own gRPC, `MEMORY_CONTEXT_GRPC_PORT`=50051) |
| `BILLING_SERVICE_URL` | `http://127.0.0.1:8090` | billing REST: the startup check that `LLM_MODEL` is a model billing can price (warning only) |
| `BILLING_SERVICE_TOKEN` | empty | the memory token from billing's `dev_up.sh` (value only) |
| `BILLING_LIMITS_ENABLED` | `false` | enforce the plan's `memory_facts` cap (Free 50, Pro 500, Max 2000, -1 unlimited) |
| `ENTITLEMENTS_CACHE_SECONDS` | `60` | per-user entitlements in Redis (`ent:{user_id}`), a stale copy for 1 h, then bundled Free limits |
| `KAFKA_BOOTSTRAP_SERVERS` | `localhost:9092` | **on Windows set `127.0.0.1:9092`** |

**Cost metering.** Every paid call goes through `GroqClient.chat_completion` (via `GroqProvider` and `LLMManager`), inside a
*metered job* opened by the worker that owns it: `summary_worker` (`memory.summary`, `memory.branch_summary` for a bounded branch
summary, `memory.compaction` for pressure / idle / stale-read requests), `fact_worker` (`memory.facts`) and `embedding_worker`
(`memory.facts`, the model that judges whether a new fact corrects an old one). One job writes ONE `usage.metered` 1.0 event through
the existing outbox (key = user id): the provider's token counts as reported (estimated and flagged `usage_estimated` when it reports none,
`memory_usage_estimated_total`), one `calls` entry per call that reached the provider (a timed-out retry included), `event_id = uuid5(job)`
so a redelivered job cannot double count. The mock provider is never metered. A paid call outside any job is counted in
`memory_unmetered_calls_total` and logged (a bug to fix).

**memory_facts cap.** The count is the user's ACTIVE facts, read from Cassandra by the merge itself (`_existing_by_key`), so there is no
separate counter to drift. At the cap no NEW fact is stored (`memory_fact_cap_reached` log, `memory_fact_cap_total`); a fact that replaces
another keeps the count and is still allowed; existing facts stay readable and editable; deleting one frees a slot. `GET /memory/facts`
returns `facts_used` and `facts_limit` (`null` while limits are off, `-1` unlimited).

**user.deleted.** Deletes the user's facts (active and retired), vectors, memory settings, tombstones and idle-compaction entries, and
purges the conversations it can attribute to them through the existing conversation-delete path. The rest of the per-conversation data
goes when conversation-service publishes `conversation.deleted` for the user's conversations after its retention period. Idempotent and
deduped; a failure is retried (the offset is not committed).

Stubs: `proto/billing/` (vendored), `make proto` or `.venv\Scripts\python scripts\gen_billing_proto.py`.
