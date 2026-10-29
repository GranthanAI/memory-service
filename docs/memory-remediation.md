# Memory system remediation

Scope: `memory-service`, `llm-service`, `conversation-service`. This file is the source of truth for later sessions;
update the status table at the end of every phase.

## Working rules

- One phase per run; stop and report after each. Wait for "continue".
- Test first: reproduce each finding with a failing test, then fix. Never weaken existing tests.
- **Git: work on `main` only, push to `main` only** (user instruction of 2026-10-28; supersedes the original
  branch-per-phase rule). No feature branches, no PRs. Small focused commits.
- Event / gRPC contract changes are additive only (new fields optional; consumer accepts old and new shapes; change
  the consumer before the producer).
- Cassandra changes only through new numbered migrations in `app/db/migrations.py` (Migration6, 7, ...), additive and
  re-runnable. Never drop or rewrite tables.
- Stored rows (uuid5 ids, equal user/assistant timestamps, summaries without version history) must keep working; when
  correctness cannot be proven, fail closed.
- Behaviour that changes what the model sees goes behind a config flag with a documented default.
- Reproduction tests that expose a live bug are committed as `xfail(strict=True)` in
  `tests/unit/test_memory_remediation_repro.py`; the fixing commit removes the marker.

## Findings

| ID | Finding | Where |
|----|---------|-------|
| P0-1 | User and assistant message of a turn get the same `created_at` and uuid5 ids; `message_key()` sorts by `(created_at, message_id)`, so ~50% of turns are stored assistant-first (window, summarizer input, history search). | `app/workers/memory_update_worker.py`, `app/services/summary_service.py` |
| P0-2 | `summary_valid_at_branch()` compares memory's uuid5 `through_message_id` with conversation-service's UUIDv7 `upto_message_id` numerically. ~0.6% pass: ancestor summaries are almost never shown. **Confirmed in Phase 0: the same random comparison also falsely accepts ~0.6% of summaries that DO cover messages after the branch point (a small leak), so it is a safety bug as well as a recall bug.** | `app/services/context_builder.py` |
| P0-3 | `process_incremental_summary` writes back the whole snapshot row read before the LLM call (`save_snapshot`), resetting `message_count`/`snapshot_version` bumped by the dispatcher meanwhile. | `app/services/summary_service.py`, `memory_repository.save_snapshot` |
| P0-4 | `MemoryUpdateWorker._loop` logs errors and commits the offset; a failure on the second dispatch leaves half a turn. | `app/workers/memory_update_worker.py` |
| P0-5 | `retrieved_history` is parsed in llm-service but never rendered in the prompt. | `llm-service/app/grpc/clients/memory_client.py`, `app/prompts/builder.py` |
| P0-6 | `PriorityTrimmer` head-keeps `conversation_history`: oldest turns kept, newest dropped. | llm-service |
| - | Trigger accounting (`message_count % SUMMARY_THRESHOLD`; `CompactionScheduler.reset()` zeroes counters), in-flight TTL 120 s, `model_version = LLM_PROVIDER`, Redis/Cassandra summary cache incoherence, history flattened into one user block, 180-char compaction note, no read-your-writes merge, `GetMemoryContext` ignores `max_tokens`, "Confidence" label on importance. | see Phase 2 / 3 |
| - | Branch-unsafe single-row summaries; short chats never yield facts; destructive fact supersede; no delete wiring; no anchor ledger; no eval harness. | see Phase 4-7 |

Not verified yet (confirm before fixing in its phase): P0-4 and everything in Phases 2-7.

## Phase plan

- **Phase 0** Baseline + failing repro tests for P0-1/2/3. No fixes.
- **Phase 1** Message identity and ordering (llm-service producer adds optional `user_message_id`,
  `assistant_message_id`, `user_created_at`, `assistant_created_at`; memory-service consumer uses them, legacy events
  keep uuid5 but user sorts strictly before assistant; `summary_valid_at_branch` compares UUIDv7 with UUIDv7, legacy
  uuid5 watermarks via `through_created_at` vs the UUIDv7-encoded time, else fail closed). Consumer first, then producer.
- **Phase 2** Pipeline integrity: LWT/column-scoped snapshot writes (P0-3), ingestion retry/DLQ + idempotency (P0-4),
  trigger accounting from watermark, in-flight TTL, summary metadata migration (`model_version`, `prompt_version`),
  single-object summary cache.
- **Phase 3** Request-time context: history trimming from the oldest end, render `retrieved_history`
  (`HISTORY_SEARCH_ENABLED`), role-separated history, head+tail compaction note, read-your-writes merge (message ids in
  conversation-service history), `GetMemoryContext` budget, fact label.
- **Phase 4** Branch-safe summaries: immutable `conversation_summary_versions`, bridge gap with raw messages,
  summary on branch creation.
- **Phase 5** Long-term facts: explicit-memory fast path, idle trigger, exact message ids for extraction, soft
  supersede with LLM judgement, core profile tier, recency from `last_confirmed`, delete wiring (decision needed: are
  facts from a deleted conversation removed too?), regenerated responses.
- **Phase 6** Summary fidelity: anchor ledger, pre-commit validation, per-section budgets, `anchor_lost` outcome.
- **Phase 7** Evaluation harness (no paid LLMs in CI), Make target.

## Status

| Phase | Status | Notes |
|-------|--------|-------|
| 0 | done | baseline + repro tests |
| 1 | done | message identity/ordering; branch validity by UUIDv7 time (see Phase 1 notes) |
| 2 | done | P0-3 column-scoped/CAS snapshot writes; P0-4 ingestion retry/DLQ; watermark-relative triggers; derived + renewed in-flight TTL; Migration6 summary metadata; one-object summary cache |
| 3 | done | P0-6 oldest-first history trimming; P0-5 retrieved history rendered; role-separated history (flag, default OFF); head+tail compaction note; read-your-writes merge; GetMemoryContext budget; fact label |
| 4 | done | Migration7 immutable summary versions; version choice + raw-message bridge; bounded summary jobs; branch-creation worker |
| 5 | done | Migration8 fact lifecycle; LLM-judged soft supersede; reconcile; explicit fast path; idle trigger; consumed-message extraction; conversation.deleted consumer; regenerated answers; core profile; previous-turn query |
| 6 | done | Migration9 anchor ledger; validation + one retry; per-section trimming; `anchor_lost` / `unsupported_entity` outcomes; anchors over gRPC |
| 7 | done | `make eval` (offline, deterministic) |

## Phase 0 baseline (2026-10-28, all on `main`)

| Repo | Command | Result |
|------|---------|--------|
| llm-service | `uv run pytest tests` | 711 passed, 2 skipped |
| conversation-service | `uv run python -m pytest tests/` (plain `pytest` fails to import `app`; the Makefile uses `python -m`) | 421 passed, 1 error (`tests/integration/test_citation_migration.py`, needs live infra) |
| memory-service | `uv run pytest tests/` | see report: integration tests need live Cassandra/Milvus/Redis; `test_background_workers_integration::test_end_to_end_worker_pipeline_integration` failed in the baseline |

Repro tests (`tests/unit/test_memory_remediation_repro.py`, all fail without `xfail`):

- `test_legacy_event_turns_are_stored_user_then_assistant` (P0-1)
- `test_valid_legacy_ancestor_summary_is_accepted` (P0-2, recall)
- `test_legacy_ancestor_summary_past_the_branch_point_is_rejected` (P0-2, leak)
- `test_summary_commit_does_not_reset_counters_bumped_during_the_llm_call` (P0-3)

## Phase 1 notes (message identity and ordering)

- **Event shape (additive):** `memory.update.requested` gains optional `user_message_id`, `assistant_message_id`,
  `user_created_at`, `assistant_created_at`. llm-service fills them from `ctx.message_id`, `ctx.response_message_id`,
  the user message's UUIDv7 time (else `ctx.started_at`) and the completion time. Old events (no fields) and new events
  are both accepted.
- **Ordering rule:** `message_key` is unchanged, `(created_at, message_id)`. The ingestion worker now guarantees the
  assistant message is stamped strictly after the user message (>= 1 ms, Cassandra's precision): the given times if
  they are ordered, else `user + 1 ms`. Legacy events keep their uuid5 ids and event ids (redelivery stays idempotent);
  only the assistant `created_at` moves by 1 ms. Watermarks, `select_new_messages`, `_watermark`, history search and the
  gRPC window all use `message_key`, so they need no change.
- **Not fixed (existing rows):** turns already stored with equal timestamps keep their old, possibly scrambled order.
  Rewriting them is out of scope (never rewrite stored rows); a redelivered legacy event after the upgrade would add a
  second row for the assistant at the shifted time (same message id, different `created_at`).
- **Branch validity (`summary_valid_at_branch`):** v7 vs v7 compares the millisecond timestamp (equal ms only if the ids
  are identical, since order inside one ms is random); a legacy uuid5 watermark compares `through_created_at` with the
  time in the branch id, strictly earlier only; anything else fails closed. Consequence: a legacy summary that covers the
  very latest turn is not shown to a branch made at that turn (publish time is after the message time, so it is not
  provable). New-format summaries do not have this limit.
- **Rollout order:** deploy memory-service first, then llm-service.

## Phase 2-7 notes

**Deploy order: memory-service -> llm-service -> conversation-service.** memory-service accepts old and new event shapes, and an OLD
llm-service simply ignores the new response fields (`retrieved_history`, `anchors`). The reverse pairing also works: a NEW llm-service
against an OLD memory-service receives those fields empty and builds the prompt without the new sections (tested in
llm-service `tests/unit/test_old_memory_service_compat.py`).
Cassandra migrations 6-9 run at memory-service startup (additive, re-runnable). The proto change (`Anchor`, `anchors = 11`) is additive.

**New configuration (all in `app/core/config.py`; defaults in brackets):**
`LLM_MODEL_VERSION` [""], `ANCESTOR_VERSION_LOOKBACK` [50], `ANCESTOR_BRIDGE_MAX_TOKENS` [1500], `CORE_PROFILE_ENABLED` [true],
`RETRIEVAL_INCLUDE_PREVIOUS_TURN` [true], `EXPLICIT_MEMORY_FAST_PATH` [true], `IDLE_COMPACTION_ENABLED` [true],
`IDLE_COMPACTION_MINUTES` [30], `DELETE_FACTS_WITH_CONVERSATION` [true], `ANCHOR_LEDGER_ENABLED` [true],
`ANCHOR_LLM_EXTRACTION` [true], `ANCHOR_RENDER_MAX_CHARS` [3000], `SUMMARY_VALIDATION_RETRIES` [1], `HISTORY_SEARCH_ENABLED` [true].
llm-service: `PROMPT_HISTORY_AS_MESSAGES` [**false**: changes every prompt's shape; enable after a canary],
`PROMPT_RETRIEVED_HISTORY_ENABLED` [true].

**Existing tests that had to change because the design changed (assertions kept otherwise):** `test_branch_summaries` (UUIDv7 stand-ins
instead of fake "v7-looking" ids), `test_db` (new Cassandra settings), `test_memory_service` (state change is a compare-and-set, not a
whole-row upsert), `test_long_memory_service` / `test_memory_quality` (facts are retired, not deleted; corrections need a judge),
`test_event_history` in conversation-service (history entries carry `message_id`).

**Decisions taken without asking (change them if you disagree):**
1. Deleting a conversation purges its messages, summaries, summary versions, snapshot and caches, and RETIRES the facts whose source
   conversation it was (`DELETE_FACTS_WITH_CONVERSATION=true`; retired = kept as history, vector removed, never retrieved). A fact the user
   restated in another conversation moves its source to that conversation (`touch_fact_confirmed`), so it survives. A branch made from a
   deleted conversation loses the inherited *summary* of it (conversation-service already stops lineage at a deleted ancestor).
2. Role-separated history ships dark (flag off).
3. Cassandra cannot make one conditional batch across two tables, so a superseded summary is archived first (idempotent), then replaced.

**Known limits / not done:**
- Turns stored before Phase 1 with equal timestamps keep their old order; a legacy summary through the latest turn is not shown to a branch
  made at that turn (not provable); a regenerated answer only replaces the raw message (summaries are not edited retroactively) and only when
  the event carries real ids.
- `last_used` is stamped in a background task when facts are retrieved (at most once per `FACT_LAST_USED_MIN_INTERVAL_SECONDS`, best effort); ranking recency uses the later of `last_confirmed` / `last_used`.
- The anchor ledger is not added to *inherited* (ancestor) summaries; a child sees the ancestor summary version + bridge only.
- The entity check is deterministic (identifiers, numbers, versions, paths, URLs): it does not catch an invented *name* in prose.
- The branch-creation and conversation-delete consumers record failures in `retry_jobs` (retry with backoff, then a FAILED dead-letter row; an unrecordable failure is not committed past). Retries go to dedicated topics (`memory.branch_summary.retry`, `memory.conversation_delete.retry`), which must exist.
- `evaluation.run --real` (paid model) is a stub that refuses to run; the harness is offline-only.

## Phase 7: evaluation harness

`make eval` (or `uv run python -m evaluation.run`): synthetic conversations with planted identifiers, constraints, reversed decisions and open
items; N compaction cycles with a deliberately lossy deterministic summarizer; reports recall of planted facts with and without the
ledger, leakage across a branch point, and fact-supersede precision/recall with a perfect and a noisy judge. Reference run (seed 7, 6 cycles,
120 messages, 5 conversations): mean recall 0.36 without the ledger, 1.00 with it; 0 facts leaked across the branch; supersede precision 1.00.

## Proposal (out of scope): multilingual embedding model

Switching `all-MiniLM-L6-v2` to a multilingual model (e.g. `paraphrase-multilingual-MiniLM-L12-v2`, same 384 dims, or
`multilingual-e5-base`, 768 dims) needs: a new Milvus collection (dimension may change), a re-embed job over all
stored messages/facts from Cassandra (source of truth), dual-write during backfill, a read switch behind a flag, and
retirement of the old collection after validation. Not started; only on request.
