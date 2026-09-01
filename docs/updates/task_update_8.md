# Task Update 8: Summarization Ordering Bug, Health Check Typo, Timestamp Crash Fix, Real Embedding Model, and Groq Key/Model Fixes

This update documents four real bugs found and fixed during a full-stack live memory-tier test, plus switching from the mock embedding client to the real `embedding-service`, and adding a `SUMMARY_THRESHOLD` setting.

---

## 1. Executive Summary

A live end-to-end test (documented in full at `../../../docs/full-system-verification-report.md` at the repo root) drove real conversations through the entire stack and, for the first time, actually exercised the summarization and fact-extraction pipelines under real conditions rather than unit-test mocks. This surfaced four genuine defects — none of them previously caught, since nothing had exercised the full ingestion → summarize → extract-facts → embed chain live before:

1. `SummaryService` was constructed with a `None` `llm_service` reference due to a construction-order bug in `container.py` — **every summarization job had been silently failing since this code was written**.
2. The internal `/internal/health/ready` probe's `llm_engine` check called a nonexistent attribute, always reporting `error` regardless of actual health.
3. `GetMemoryContext`'s gRPC handler crashed whenever long-term facts were included, because it didn't handle Milvus's numeric (`DOUBLE`) timestamp format.
4. Both this service's and LLM Service's Groq API keys had expired.

All four are fixed. Additionally, Memory Service now uses the **real** embedding model (previously the deterministic mock, since `embedding-service` couldn't run earlier due to host memory pressure — it runs fine now), and a `SUMMARY_THRESHOLD` setting was added so the summarization cadence can be tuned without editing code.

---

## 2. Bug: `SummaryService` Constructed with `llm_service=None`

`app/core/container.py`'s `init_resources()` had:
```python
self.summary_service = SummaryService(self.memory_repo, self.cassandra_repo, self.llm_service)
...
self.llm_service = LLMService(self.llm_manager)   # assigned AFTER the line above
```
`self.llm_service` starts as `None` (per `Container.__init__`) and `SummaryService.__init__` only stores whatever reference it's given at construction time — it doesn't look it up later. Every `SummaryWorker` job therefore crashed with `'NoneType' object has no attribute 'summarize'`, and — depending on exactly when the crash occurred relative to schedule/retry logic — could leave a conversation's memory snapshot stuck in an inconsistent `state` (observed directly during testing: a conversation got stuck reporting `"cannot move from MemoryState.FACT_PENDING to MemoryState.SUMMARIZING"` on a stale retry attempt after a mid-flight fix; it self-healed on the next threshold crossing, since `EventDispatcher` unconditionally overwrites `state` to `SUMMARY_PENDING` regardless of current state — see the root-level report for the full narrative).

**Fix**: reordered construction so `self.llm_service = LLMService(self.llm_manager)` happens before `SummaryService(...)` is built.

**Verified live**: after the fix and a restart, a real conversation generated three genuine, correctly incremental summaries end-to-end via Groq, persisted to `conversation_summaries` in Cassandra.

---

## 3. Bug: `llm_engine` Health Check Called the Wrong Attribute

`app/api/internal/health.py`'s deep readiness probe had:
```python
llm_ok = await llm_service.manager.check_health()
```
`LLMService` stores its manager as `self.llm_manager`, never `self.manager` — this always raised `AttributeError`, which the surrounding `try/except` correctly caught and reported as `{"status": "error", "detail": "'LLMService' object has no attribute 'manager'"}`, masking whatever the real Groq health state actually was.

**Fix**: corrected to `llm_service.llm_manager.check_health()`. Readiness now correctly reports the real underlying key/model health (which, before the fixes in §5, was itself failing for a genuine reason — an expired key).

---

## 4. Bug: `GetMemoryContext` Crashed on Milvus's Numeric Timestamps

`app/proto/memory_handlers.py`'s `_to_epoch_millis()` helper (added in Task Update 7, when the gRPC context server was first built) only handled `None`, `str`, and datetime-like objects. Milvus stores `created_at` as a raw `DOUBLE` (epoch seconds) per its schema (`app/repositories/milvus_repository.py`), not a datetime — so any `GetMemoryContext` call that included long-term facts crashed with `'float' object has no attribute 'tzinfo'`.

**Fix**: added a numeric branch (`isinstance(value, (int, float))`) that converts directly to epoch milliseconds, before falling through to the existing string/datetime handling.

**Verified live**: a semantic fact-retrieval call (`GetMemoryContext` with a real query string) that previously crashed now correctly returns ranked long-term facts.

---

## 5. Groq API Key and Model Fixes

`GROQ_API_KEY` in `.env` had expired (`401 Invalid API Key` confirmed directly against Groq's API). Replaced with a new working key. `LLM_MODEL` was also switched from `llama-3.3-70b-versatile` (no longer available on the new key's account — confirmed via `/v1/models`) to `openai/gpt-oss-20b`, matching the same model switch made in LLM Service (see its own Task Update 14) for the same reason: better-suited to reliable structured/deterministic output than an agentic router model, without the unnecessary tool-use overhead for a bounded summarization/fact-extraction task.

---

## 6. Switched to the Real Embedding Model

`EMBEDDING_CLIENT_TYPE` changed from `mock` to `grpc` in `.env`, now that `embedding-service` runs successfully in this environment (`SentenceTransformer` `all-MiniLM-L6-v2`, 384 dimensions, gRPC on port 50054 — `LLM_SERVICE_HOST`/`LLM_SERVICE_PORT` already pointed at the correct port from earlier work).

**Verified live**: a direct gRPC call to `embedding-service` returned a real 384-dimensional vector; every long-term fact recorded during this session's test carries `embedding_version: v1.0.0`; semantic fact retrieval via a real query string returned correctly ranked results — confirming genuine embeddings are in the loop, not the deterministic mock.

---

## 7. New Setting: `SUMMARY_THRESHOLD`

`EventDispatcher`'s summary-trigger threshold (messages per conversation before a summary is staged) was hardcoded as a constructor default (`20`) in `app/events/dispatcher.py`, never exposed as configuration. Added `SUMMARY_THRESHOLD: int = 20` to `app/core/config.py` and wired it through `container.py`'s `EventDispatcher(...)` construction (`summary_threshold=settings.SUMMARY_THRESHOLD`), so it can be tuned per-environment without a code change. Temporarily set to `2` in `.env` for this session's live testing (fast, reliable triggering); should be restored to a production-appropriate value (`20`, or unset to use the default) before real deployment.

---

## 8. What Was NOT Changed

- `EventDispatcher`'s core logic, `MemoryService`'s state-machine transition matrix, `ContextBuilder`/`RetrievalService` — all pre-existing and correct; only the broken wiring around them was fixed.
- No change to the long-term-facts-are-scoped-by-`user_id`-not-`conversation_id` behavior — confirmed via live testing to be the deliberate, correct design (see the root-level verification report for the full parent/child isolation findings), not something to fix.

## 9. Follow-ups

- Restore `SUMMARY_THRESHOLD` to a production value before deployment.
- The snapshot's `fact_version` field showed `0` even after facts were successfully extracted and individually versioned in `user_facts` — worth a closer look to confirm whether this tracks something else intentionally or is a minor separate gap.
