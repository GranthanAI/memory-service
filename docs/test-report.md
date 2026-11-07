# Test report

Generated from a fresh run on 2026-11-07 13:23 UTC, repository commit `a26c838` (`main`). Re-run with the commands below.

| Suite | Command | Runs against | Tests | Passed | Failed | Errors | Skipped | Time |
|---|---|---|---|---|---|---|---|---|
| All tests | `uv run pytest tests -q` | the shared dev Cassandra and Redis (not isolated; see notes) | 423 | 419 | 3 | 1 | 0 | 182 s |

## Failing tests

| Suite | Kind | Test | First line of the error |
|---|---|---|---|
| All tests | failure | `tests.integration.test_background_workers_integration::test_end_to_end_worker_pipeline_integration` | assert 1 == 2 |
| All tests | failure | `tests.integration.test_long_memory_service_integration::test_long_memory_service_fact_merge_policy_integration` | assert 0 == 1 |
| All tests | failure | `tests.integration.test_memory_service_integration::test_memory_state_machine_and_cache_hydration_integration` | TypeError: 'NoneType' object is not subscriptable |
| All tests | error | `tests.integration.test_summary_service_integration::test_summary_service_incremental_algorithm_integration` | failed on setup with "RuntimeError: Cassandra session is not initialized. Call connect_cassandra() during application startup." |

## Notes on the memory-service result

The suite was run twice on commit `a26c838`; the table above is the **second** run.

| Run | Conditions | Tests | Passed | Failed | Errors |
|---|---|---|---|---|---|
| 1 | the dev memory-service and its worker were running | 423 | 414 | 5 | 4 |
| 2 | the dev memory-service stopped (this report) | 423 | 419 | 3 | 1 |

* **Run 1 extra failures were interference, not code.** Four unit tests failed with `Failed to bind to address [::]:50051` because the running dev memory-service owns that
  port. A fifth, `test_outbox_worker_concurrency_lwt_duplicate_prevention` (`assert 17 == 20`), reads and writes the shared dev Cassandra while the live worker consumes the
  same rows. All five pass when the dev service is stopped.
* **The remaining 3 failures and 1 error are not caused by the anchor change (`a26c838`).** The same four tests were run on the commit before it (`4caa558`) in a
  separate checkout with the same environment, and they fail there in the same way: `test_end_to_end_worker_pipeline_integration`,
  `test_long_memory_service_fact_merge_policy_integration`, `test_memory_state_machine_and_cache_hydration_integration` (failures on state/snapshot assertions) and
  `test_summary_service_incremental_algorithm_integration` (`Cassandra session is not initialized`, a fixture problem). They are integration tests against the shared
  dev Cassandra and Redis. The cause is not diagnosed here.
* To reproduce a clean unit run, stop the dev memory-service first (`make down-memory` from the repository root) and start it again afterwards (`make up-memory`).
