# Testing Report - Phase 11

Date: 2026-11-14
Scope: full application validation for the local LLM benchmarking project (Phases 01-10) plus Phase 11 defect fixes and regression tests. No new benchmark features were added.

## Environment

| Item | Value |
|---|---|
| Operating system | Windows 11 (win32), PowerShell 5.1 |
| Python | 3.13.5 (`.venv`) |
| Ollama | 0.40.16 at `http://127.0.0.1:11434` |
| Installed configured models | `llama3.1:8b` (candidate), `qwen2.5:3b` (judge) |
| Other installed models | `llama3.2:3b`, `llama3.1:latest`, `llama3.2:latest`, `bge-m3:latest` |
| Missing configured candidates | `qwen2.5:7b-instruct`, `mistral:7b-instruct-v0.3-q4_K_M` |
| Dataset | `benchmark-v1.0.0`, 100 items, SHA-256 `ceafe917235213c7...`, status VALID |
| Primary E2E database | `storage/phase11-2026-11-13/phase11.db` (gitignored) |
| Offline fault database | `storage/phase11-2026-11-13/offline2.db` (gitignored) |
| Report exports | `storage/phase11-2026-11-13/exports/` (gitignored) |

No model was pulled. Model downloads still require explicit user approval.

## Summary

| Check | Command | Result |
|---|---|---|
| Byte-compile | `python -m compileall -q src tests` | PASS (exit 0) |
| Full test suite (default) | `python -m pytest -q` | **328 passed, 3 skipped**, 91.69 s (exit 0) |
| Live Ollama tests | `BENCHMARK_LIVE_OLLAMA=1 pytest tests/integration/test_ollama_live.py -q` | **3 passed**, 24.11 s (exit 0) |
| Evaluation suite | `pytest tests/evaluation -q` | **81 passed**, 1.76 s (exit 0) |
| Persistence suite | `pytest tests/integration/test_experiment_persistence.py -q` | **8 passed**, 3.91 s (exit 0) |
| Health | `python -m benchmark health` | PASS with 1 warning (two configured candidates absent) |
| Dataset validation | `python scripts/validate_dataset.py` | VALID (100 items, probes valid) |
| Model verification | `python -m benchmark models verify` | exit 1: 2 configured candidates not installed (expected), installed models verified |
| Dashboard server | `streamlit run src/benchmark/dashboard/app.py --server.headless=true` | `/_stcore/health` = 200 `ok`; root page = 200 |
| Database integrity | `PRAGMA integrity_check` / `PRAGMA foreign_key_check` | `ok` / no rows |

The 3 skipped tests are the opt-in live Ollama tests; they pass when enabled. No test was reported as passing without being executed.

## 1. Repository audit

Inspected packaging/configuration, all `src/` packages (core, dataset, database, execution, monitoring, evaluation, analytics, dashboard, reporting), `scripts/`, `tests/`, and the documentation set.

- Architecture layers remain separated as documented: configuration and paths, Ollama client and streaming parser, inference engine and orchestrator, SQLite repositories, deterministic evaluation plus blind judge, analytics/scoring, dashboard read layer, and reporting exporters.
- The import graph is acyclic at the package level. No orphaned source modules were found by entry-point and cross-module import inspection.
- Schema, database, inference, monitoring, evaluation, analytics, reporting, and dashboard interfaces agree with their consumers, with two recorded inconsistencies:
  - `benchmark infer` accepts the judge tag as a model; `scripts/run_experiment.py --models` accepts candidate tags only.
  - `benchmark analyze` and `benchmark compare` do not expose `--database`, while `run_experiment.py` and `benchmark report` do.
- No secrets, credentials, or tokens were found in the repository. `.env.example` is a template and is not loaded by the application.

## 2. Automated tests

The full default suite passed: 328 passed, 3 skipped. Coverage includes configuration validation and hashing, logging, dataset models/loader/hash/schema checks, PDF/HTML/CSV/JSON exporters, analytics dimensions and recommendation gates, SQLite repositories, streaming parser, inference timing and cancellation, hardware summaries, judge parsing, deterministic evaluators, sandbox behavior, dashboard pages and runner locking, and reporting.

New Phase 11 regression tests:

- `tests/unit/test_config.py`: digest prefix normalization, non-UTF-8 configuration errors, non-empty default category judge weights.
- `tests/unit/test_dataset_loader.py`: non-UTF-8 dataset errors.
- `tests/evaluation/test_numeric.py`: colon answer markers and multi-number extraction.
- `tests/evaluation/test_code_tests.py`: timeout with a stale sandbox result never counts as passed.
- `tests/evaluation/test_scoring.py`: category-level judge missing.
- `tests/unit/test_database.py`: terminal-state guards, cancelled-run response rejection, error owner mismatch, response inspection filtered by run profile.
- `tests/analytics/test_analytics_migration.py`: migration idempotency.
- `tests/dashboard/test_dashboard_runner.py`: fresh empty lock blocks a concurrent start; stale empty lock is replaced.
- `tests/reporting/test_reporting_exports.py`: CSV exports neutralize spreadsheet formulas.
- `tests/unit/test_inference_engine.py`: a stream exceeding the total deadline times out.
- `tests/unit/test_inference_runner.py`: a sticky interrupt is reset before a new run.
- `tests/integration/test_experiment_persistence.py`: warm-up sequences exceed question count without collisions; resume rejects a changed configuration hash.
- `tests/unit/test_monitoring_gpu.py`: `nvidia-smi` fallback selects the configured device index.

## 3. Integration tests

| Area | Evidence | Result |
|---|---|---|
| Ollama connection | Live `version` + `/api/tags` test | PASS |
| Model management | `models verify`, live inventory; CLI tests | PASS for installed models; 2 missing candidates correctly reported |
| Dataset loading | `validate_dataset.py`, loader tests, canonical JSON + JSONL | PASS |
| Inference | Live streaming smoke; 10-question experiment; stored records | PASS |
| Streaming | Live NDJSON stream test; invalid/final-chunk tests | PASS |
| Performance metrics | Stored TTFT/latency/token counts; analytics tests | PASS |
| Hardware monitoring | 10 stored summaries in the 10-question run | PASS |
| Evaluation | Deterministic checks, real judge in the 10-question run, failure paths | PASS |
| Database persistence | Reopen after process exit, integrity check, no FK violations | PASS |
| Analytics | `analyze --all-profiles` on the 10-question run | PASS |
| Dashboard | Full AppTest suite; headless server health and root page | PASS |
| Reporting | CSV/JSON/HTML/PDF + manifest for a completed experiment; incomplete refused | PASS |

## 4. End-to-end scenarios

All scenarios used real Ollama inference except where explicitly fault-injected.

| Scenario | Execution | Result |
|---|---|---|
| A. One-question benchmark across three models | `infer` with installed `llama3.1:8b` + `qwen2.5:3b` on `gk-002`; the requested third configured candidate was unavailable | PASS for 2 models: 4 requests (2 warm-up), 2 measured, 2 successful, completed. A three-configured-candidate comparison was not possible because two candidates are not installed. |
| B. Ten-question benchmark | `run_experiment.py --models llama3.1:8b --limit 10 --warmup 1` with judge `qwen2.5:3b` | PASS: 1 warm-up + 10 measured responses, 10 evaluations, status completed, 0 failed (`phase11-fresh-ten`). |
| C. Repeated benchmark, 3 repetitions | `--items gk-001 --repeats 3 --profile consistency --no-judge` | PASS: 1 warm-up + 3 measured responses, 3 evaluations, completed (`phase11-fresh-repeats`). |
| D. Cold-start experiment | `--items json-001 --cold-start --profile cold_start --no-judge` | PASS: 3 responses, all marked `cold_start=true`, 3 evaluations, completed (`phase11-fresh-cold`). |
| E. Warm-start experiment | `--items gk-002 --warmup 1 --no-judge` | PASS: 1 warm-up + 1 measured response, 1 evaluation, completed (`phase11-fresh-warm`). |
| F. Failed model request | Selected an installed + uninstalled model; also exercised timeout/connection/error paths through automated tests | PASS: the command exits 1 with `selected models are not installed in Ollama: qwen2.5:7b-instruct...` and creates **zero** experiment or run rows. |
| G. Ollama unavailable | Dead endpoint `http://127.0.0.1:59999` via a copied config directory | PARTIAL / finding: `health` prints `[WARN]` and exits 0. `run_experiment --no-manager` exits 1 with a typed `ERROR: Could not connect...` but records one failed experiment and one failed run with zero responses. The first attempt without `--no-manager` did not return within 180 s and was terminated. |
| H. Invalid JSON response | Real `json-001` cold-start path plus the evaluation suite's invalid/schema-wrong response tests | PASS: invalid JSON is recorded as failed; schema-valid but wrong values are partial. The cold-start JSON run produced 3 passing evaluations; `tests/evaluation` passed 81/81. |
| I. Failed coding test | Real sandbox through the evaluation suite with wrong and reference implementations | PASS: wrong code fails all authored tests; reference code passes. Covered by `tests/evaluation` (81 passed). |
| J. Restart and historical retrieval | New process opened the database; analytics, comparison, reporting, and dashboard reads | PASS with one CLI caveat: integrity `ok`, no FK violations, 4 experiments / 20 responses / 17 evaluations. Reports produced 9 files + manifest (PDF 7 pages); incomplete experiment refused; `analyze --all-profiles` and `compare` succeeded; dashboard health/root = 200. `analyze`/`compare` required a copied config pointing at the DB because they lack `--database`. |

## 5. Performance and reliability

| Check | Method | Result |
|---|---|---|
| Timeouts | Automated engine tests for timeout status; sandbox per-test/total timeouts | PASS: timeouts are recorded as `timeout`, distinct from errors; sandbox enforces 5 s/test and 20 s total. |
| Total stream deadline | Automated slow-stream test | PASS after fix: the engine compares wall-clock time against the absolute deadline instead of comparing elapsed time against it. |
| Interrupted experiments | Persistence suite: interrupt, resume, captured interrupt, config mismatch | PASS: open experiments remain resumable; stored requests are not repeated; changed configs are rejected. |
| Database locking | WAL + busy timeout tests in the suite | PASS in the executed suite. |
| Duplicate submissions | Dashboard lock-file tests; duplicate response/recommendation tests | PASS: a live child blocks a second start; stale PIDs are not treated as live; identical writes are idempotent, conflicting writes raise. |
| Hardware data | 10-question run | 10 stored summaries; CPU average 55.74%, RAM average 13,261 MiB, GPU utilization average 26.53%, VRAM average 2,474 MiB. |
| Memory/thread soak | Not executed in this phase | Not measured; no long-duration leak test was performed. |
| Dashboard responsiveness | Headless health/root request | PASS; page timing was not re-measured in this phase. |

## 6. Security review

- **Code execution isolation:** generated code runs only in a fresh `python -I` subprocess with a temporary working directory, no shell, a minimal environment, capped output, and per-test/total timeouts. This is best-effort isolation, not a VM boundary; the subprocess retains the current user's filesystem and network permissions.
- **Subprocess timeouts:** evaluated code and `nvidia-smi` invocations are timeout-bounded; the dashboard child is started without a shell and with `CREATE_NO_WINDOW` on Windows.
- **Input validation:** dataset items, configuration, model tags, report formats, repetition counts, and item identifiers are validated before use; YAML is loaded with `safe_load`.
- **Secrets:** none found in source, configs, tests, or docs. The application reads environment variables and does not load `.env`.
- **File paths:** runtime paths are resolved against the project root; report output is written only to the requested, explicit directory; dataset loading rejects unsupported extensions.
- **Unsafe model output:** responses are never executed in-process; HTML reports use Jinja2 autoescaping; the judge treats candidate text as untrusted JSON data with a prompt-injection warning in the system message.
- **CSV injection:** exported text starting with `=`, `+`, `-`, or `@` is prefixed with an apostrophe before CSV serialization.
- **Static review:** no `shell=True`, `eval`, `exec`, `os.system`, `os.popen`, `pickle`, `yaml.load`, or `verify=False` occurrences were found under `src/`.

## 7. Defects found and fixed

### Configuration, dataset, and health

- `_load_yaml` now catches `UnicodeError`; non-UTF-8 configuration raises a typed configuration error.
- Expected model digests normalize the `sha256:` prefix; metadata verification strips it from both sides.
- The default `category_judge_weights` is non-empty.
- Dataset loading catches `UnicodeError` and raises `DatasetError`.
- The health dataset check catches broad `Exception` and records a failed check instead of crashing.

### Evaluation

- Sandbox timeouts can no longer count as passed: `passed = not timed_out and bool(result and result.get("passed") is True)`.
- Numeric answers recognize the colon marker; the evaluator prefers the last number in the first sentence after the marker, then falls back to the first number.
- Category-level `judge_missing` is true if any result lacks a judge.
- `_select_sample` accepts fraction `0.0` and returns `[]`.

### Database and persistence

- Migrations are keyed by source version, check `PRAGMA table_info` idempotently, and apply ALTER + version update atomically.
- Terminal experiment/run states are guarded by `CLOSED_RUN_STATUSES`; closed runs and experiments reject new responses.
- Error records derive the experiment from `run_id` and reject mismatched ownership.
- `open_database` wraps construction/initialization, disposes on failure, and raises `DatabaseError`.
- Response inspection accepts `run_profile` and filters by the stored run profile.

### Execution

- Warm-up repetitions use `index + 1` / `warmup_index + 1`, avoiding collisions when warm-up count exceeds question count.
- A failed run raises `ExperimentStateError`; pending evaluations resolve from the full dataset.
- Resume validates the dataset hash, closed status, and now the configuration hash. Resuming with changed settings raises a clear error before any request is repeated.
- `reset_interruption()` runs before each new inference run, preventing a sticky interrupt from cancelling subsequent work.
- `manager.load` is inside the error-handling block.
- The total stream deadline compares `now_ns > deadline_ns`; the previous elapsed-time comparison could expire too late.

### Monitoring and dashboard

- Sampler shutdown uses `_closing`/`_gpu_closed`; the worker closes the GPU handle in `finally`, and `_close_gpu()` is idempotent.
- psutil calls catch broad `Exception`; `nvidia-smi` fallback selects the configured device index.
- Dashboard Refresh clears both data and resource caches.
- Inspection supports cold/warm and run-profile selection with unique widget keys and displays `Run profile` / `Start mode`.
- The dashboard runner treats a fresh empty lock as in-progress for 60 s, replaces stale empty locks, and cleans up lock/temp/child on failed record writes.

### CLI and reporting

- `benchmark report --database` is honored.
- CSV exports neutralize spreadsheet formulas.

## 8. Known failures

There are no unresolved failing tests or unhandled defects.

Expected non-zero or warning results, documented rather than treated as pass:

- `benchmark models verify` exits 1 because two configured candidates are not installed. The installed candidate and judge verify successfully.
- `benchmark health` exits 0 with a warning for those missing candidates, and also exits 0 when Ollama is unreachable.
- The default `pytest` run reports 3 skipped live Ollama tests; they pass with `BENCHMARK_LIVE_OLLAMA=1`.
- An unavailable Ollama endpoint produces a typed `ERROR` and exit 1, but leaves one failed experiment/run (zero responses) in the database.

## 9. Known limitations and tests not executed

- Only one configured candidate is installed. No three-model comparison, model ranking, or workload-winner conclusion was produced; only single-model quality/composite values exist for the 10-question run.
- A full 100-item quality run, performance probe (9 prompts x 5 repeats), consistency probe (10 items x 3 repeats), and repeated full runs were not executed because of time cost on this CPU-only laptop. Scenario C used one item x 3 repetitions.
- No multi-hour soak, memory-profiler run, or long-duration leak test was performed.
- The dashboard was validated through Streamlit's AppTest suite and a headless server health/root check, not by interactive browser inspection or new screenshots.
- No lint or type checker is configured (no ruff, mypy, black, flake8, or pre-commit). Only byte-compilation and the test suite were available.
- No code-coverage measurement was produced (no coverage tool configured).
- The per-request timeout is applied per HTTP read by `httpx`; a server that keeps sending keep-alive bytes without finishing can exceed the nominal total request duration. The engine now enforces a wall-clock deadline when chunks arrive, but a stream that sends no chunks still relies on the HTTP read timeout.
- `run_experiment.py --models` rejects the judge tag while `benchmark infer` accepts it.
- `analyze` and `compare` do not expose `--database`; this phase used a copied config with `storage_dir` pointing at the fresh database.
- `health` treats Ollama unavailability as a warning, so automation that relies on its exit code will not stop when the service is down.
- Hardware values are system-wide; GPU/VRAM attribution on Windows WDDM remains coarse, and model-specific VRAM is unavailable for some responses.
- The code sandbox is a restricted subprocess, not a security boundary against determined adversarial code.
- Remaining lower-priority audit items were not changed: NVML fallback remains downgraded for the process lifetime after a failure; model tag normalization is not applied; recommendation persistence edge cases remain; judge preflight is not exhaustive.

## 10. Manual test results

| Manual action | Result |
|---|---|
| `python -m benchmark --help`, `scripts/run_experiment.py --help` | Expected commands and options displayed |
| `python -m benchmark health` | Pass with the expected missing-candidate warning |
| `python scripts/validate_dataset.py` | `Status: VALID`, 100 items, both probes valid |
| `python -m benchmark models verify` | Installed models verified; two missing candidates reported |
| Reports for `phase11-fresh-ten` | 9 files plus `manifest.json`; CSV/JSON/HTML/PDF exist and are non-empty |
| Cold-start report with `--run-profile cold_start --include-cold-start` | JSON report generated successfully |
| Dashboard headless start | Server started on port 8765; health endpoint 200 `ok`; root page 200 |
| Dashboard AppTest (empty and populated) | Passed within the full suite |
| Database after restart | 4 experiments, 20 responses, 17 evaluations; integrity and foreign keys clean |

## 11. Remaining risks

1. Benchmark conclusions are provisional until the two missing candidates are installed and compared under identical conditions.
2. Results come from a single machine and short runs; variance across runs, thermal state, and memory pressure is not quantified.
3. The blind judge is a small 3B model; judge agreement with human review was not measured. Judge failures are recorded as missing, not as zero, but low judge coverage lowers confidence.
4. All measured latencies are single-sample or few-sample estimates, not statistically robust distributions.
5. The sandbox limits runtime and output but does not isolate the filesystem or network.
6. SQLite remains a single-writer store; concurrent experiments are prevented only by the dashboard lock file, not by a database-level lease. A stale lock can require manual dismissal.
7. Timeout semantics are per-read rather than wall-clock total for streams that stop sending chunks.
8. An unavailable Ollama endpoint leaves a closed failed experiment row; it cannot be resumed. This is useful for diagnosis but differs from the missing-model preflight behavior.
9. Dashboard caching is keyed on the database file version; unusual filesystems with coarse timestamps could delay fresh reads.

## 12. Readiness assessment

The application is functional and internally consistent for controlled local use: configuration, dataset validation, streaming inference, hardware monitoring, hybrid evaluation, persistence, resumption, analytics, reporting, and the dashboard all execute against real Ollama and real SQLite. The full suite, byte-compilation, dataset validation, health check, live tests, and a real Streamlit startup pass.

It is **not yet ready for three-model comparative conclusions** because two configured candidates are absent and no multi-model ranking was measured. The next step should be an explicit, user-approved model pull followed by a complete 100-item run across all three candidates with repeated measurements.

## Changed files in this phase

Source:

- `src/benchmark/__main__.py`
- `src/benchmark/core/config.py`
- `src/benchmark/core/health.py`
- `src/benchmark/dashboard/app.py`
- `src/benchmark/dashboard/runner.py`
- `src/benchmark/dashboard/views/inspection.py`
- `src/benchmark/database/db.py`
- `src/benchmark/database/migrations.py`
- `src/benchmark/database/repositories.py`
- `src/benchmark/dataset/loader.py`
- `src/benchmark/evaluation/deterministic/numeric.py`
- `src/benchmark/evaluation/engine.py`
- `src/benchmark/evaluation/sandbox.py`
- `src/benchmark/evaluation/scoring.py`
- `src/benchmark/execution/engine.py`
- `src/benchmark/execution/metadata.py`
- `src/benchmark/execution/orchestrator.py`
- `src/benchmark/monitoring/gpu.py`
- `src/benchmark/monitoring/sampler.py`
- `src/benchmark/monitoring/system.py`
- `src/benchmark/reporting/bundle.py`
- `src/benchmark/reporting/csv_export.py`

Tests:

- `tests/analytics/test_analytics_migration.py`
- `tests/dashboard/test_dashboard_runner.py`
- `tests/evaluation/test_code_tests.py`
- `tests/evaluation/test_numeric.py`
- `tests/evaluation/test_scoring.py`
- `tests/integration/test_experiment_persistence.py`
- `tests/reporting/test_reporting_exports.py`
- `tests/unit/test_config.py`
- `tests/unit/test_database.py`
- `tests/unit/test_dataset_loader.py`
- `tests/unit/test_inference_engine.py`
- `tests/unit/test_inference_runner.py`
- `tests/unit/test_monitoring_gpu.py`

Documentation:

- `docs/TESTING_REPORT.md`
