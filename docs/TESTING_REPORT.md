# Testing Report — Phase 11

Date: 2026-11-12
Scope: full application validation for the local LLM benchmarking project (Phases 01–10).
Changes in this phase are limited to defect fixes and tests; no new benchmark features were added.

## Environment

| Item | Value |
|---|---|
| Operating system | Windows 11 (win32), Git Bash shell |
| Python | 3.13.5 (`.venv`) |
| Ollama | 0.40.16 at `http://127.0.0.1:11434` |
| Installed configured models | `llama3.1:8b` (candidate), `qwen2.5:3b` (judge) |
| Missing configured candidates | `qwen2.5:7b-instruct`, `mistral:7b-instruct-v0.3-q4_K_M` |
| Dataset | `benchmark-v1.0.0`, 100 items, SHA-256 `ceafe917…`, status VALID |
| Database used for E2E evidence | `storage/phase11/phase11.db` (plus `storage/phase11/benchmark.db` for dashboard reads) |
| Test artifacts | `storage/phase11/` (exports, logs, configs, databases); gitignored |

No model was pulled. Model downloads still require explicit user approval.

## Summary

| Check | Command | Result |
|---|---|---|
| Byte-compile | `python -m compileall -q src` | PASS (exit 0) |
| Full test suite (default) | `python -m pytest -q` | **308 passed, 3 skipped**, 76.80 s (exit 0) |
| Live Ollama tests | `BENCHMARK_LIVE_OLLAMA=1 pytest tests/integration/test_ollama_live.py -q` | **3 passed**, 18.46 s (exit 0) |
| Health | `python -m benchmark health` | PASS with 1 warning (two configured candidates absent) |
| Dataset validation | `python scripts/validate_dataset.py` | VALID (100 items, probes valid) |
| Model verification | `python -m benchmark models verify` | exit 1: 2 configured candidates not installed (expected), installed models verified |
| Dashboard server | `streamlit run src/benchmark/dashboard/app.py --server.headless=true` | Started; `/_stcore/health` = 200 `ok`; root page = 200 |
| Dashboard pages | Streamlit `AppTest`, empty and populated databases | All 10 pages render without exception |

The 3 skipped tests are the opt-in live Ollama tests; they pass when enabled (see above). No test was reported as passing without being executed.

## 1. Repository audit

Inspected: packaging and configuration, all `src/` packages (core, dataset, database, execution, monitoring, evaluation, analytics, dashboard, reporting), `scripts/`, `tests/`, and the documentation set.

- Architecture layers are separated as documented: configuration and paths, Ollama client and streaming parser, inference engine and orchestrator, SQLite repositories, deterministic evaluation plus blind judge, analytics/scoring, dashboard read layer, and reporting exporters.
- Import graph is acyclic at the package level. No orphaned source modules were found by entry-point and cross-module import inspection. `tests/performance/` contains only a placeholder `.gitkeep`; `src/local_llm_benchmark.egg-info/` is generated and gitignored.
- No `TODO`, `FIXME`, `XXX`, `NotImplementedError`, or stub functions were found under `src/`.
- No unused third-party dependency was identified; each entry in `pyproject.toml` is imported by application code, except `pypdf` (dev) which is used by PDF tests.
- Schema, database, inference, monitoring, evaluation, analytics, reporting, and dashboard interfaces agree with their consumers. Two interface inconsistencies were recorded rather than changed (see Limitations):
  - `benchmark infer` accepts the judge tag as a model; `scripts/run_experiment.py --models` accepts candidate tags only.
  - `health` reports Ollama unavailability as a warning, so its exit code stays 0.
- No secrets, credentials, or tokens were found in the repository. `.env.example` is a template with a `[TEMPLATE]` header and is not loaded by the application.

## 2. Unit tests

The full default suite passed: 308 passed, 3 skipped. Coverage includes configuration validation and hashing, logging, dataset models/loader/hash/schema checks, PDF/HTML/CSV/JSON exporters, analytics dimensions and recommendation gates, SQLite repositories, streaming parser, inference timing and cancellation, hardware summaries, judge parsing, deterministic evaluators, and sandbox behavior.

## 3. Integration tests

Executed through the suite and through real commands against Ollama and SQLite:

| Area | Evidence | Result |
|---|---|---|
| Ollama connection | live `version` + `/api/tags` test | PASS |
| Model management | `models verify`, live model inventory; CLI tests | PASS for installed models; 2 missing candidates correctly reported |
| Dataset loading | `validate_dataset.py`, loader tests, canonical JSON + JSONL | PASS |
| Inference | live streaming smoke, 10-question experiment, stored records | PASS |
| Streaming | live NDJSON stream test; invalid/final-chunk tests | PASS |
| Performance metrics | stored TTFT/latency/token counts; analytics tests | PASS |
| Hardware monitoring | live sampler test (5 samples typical) and 19 stored summaries | PASS |
| Evaluation | deterministic checks, real judge in the 10-question run, failure paths | PASS |
| Database persistence | 4 experiments, reopen after process exit, `PRAGMA integrity_check` = ok, no FK violations | PASS |
| Analytics | `analyze` on the 10-question run and stored deltas | PASS |
| Dashboard | 10 pages on empty and populated databases, export generation | PASS after fix |
| Reporting | CSV/JSON/HTML/PDF + manifest for a completed experiment; incomplete refused | PASS |

## 4. End-to-end scenarios

All scenarios used real Ollama inference except where explicitly fault-injected.

| Scenario | Execution | Result |
|---|---|---|
| A. One-question benchmark across three models | `infer` with installed `llama3.1:8b` + `qwen2.5:3b` on `gk-002`; requested third model was unavailable | PASS for 2 models: 4 requests (2 warm-up), 2 measured, 2 successful, completed. Three configured models could not be run because two candidates are not installed; `llama3.2:3b` was rejected as outside the configured roster. |
| B. Ten-question benchmark | `run_experiment.py --models llama3.1:8b --limit 10 --warmup 1` with judge `qwen2.5:3b` | PASS: 1 warm-up + 10 measured responses, 10 evaluations, status completed, no failures. Saved to `phase11-ten`. |
| C. Repeated benchmark, 3 repetitions | `--items gk-001 --repeats 3 --profile consistency --no-judge` | PASS: 3 responses, 3 evaluations, completed (`phase11-repeats`). |
| D. Cold-start experiment | `--items json-001 --cold-start --profile cold_start --no-judge` | PASS: 3 responses (cold-start profile repetition default), all marked `cold_start=true`, completed (`phase11-cold`). |
| E. Warm-start experiment | `--items gk-002 --warmup 1 --no-judge` | PASS: 1 warm-up + 1 measured response, completed (`phase11-warm`). |
| F. Failed model request | Selected installed + uninstalled models; also exercised timeout/connection/error paths through automated tests | FAIL initially (defect 1), then PASS: the command exits 1 with `selected models are not installed in Ollama: qwen2.5:7b-instruct…` and creates **zero** experiment rows. |
| G. Ollama unavailable | Dead endpoint `http://127.0.0.1:59999` via a copied config directory | PASS with caveats: `run_experiment` and `infer` exit 1 with a typed `ERROR: Could not connect…`, no traceback and no partial rows. `health` prints `[WARN]` and exits 0 (see Limitations). |
| H. Invalid JSON response | Real `json-001` evaluation path, plus invalid and schema-valid-but-wrong responses | PASS: invalid JSON → status `failed`, score 0, failed `json_parse`/`schema_valid`/`reference_match`; valid JSON with wrong values → `partial`, score 50, only `reference_match` failed. The stored cold-start JSON run produced 3 evaluations. |
| I. Failed coding test | `code-001` through the real sandbox with a wrong implementation and with the reference solution | PASS: wrong code → passed `False`, score 0, all 3 authored tests failed; reference code → passed `True`, score 100. |
| J. Restart and historical retrieval | New process opened the database, then analytics, comparison, reporting, and dashboard reads | PASS after fix: 4 completed experiments retrieved with correct counts, DB integrity ok, 9 report files + manifest (PDF 7 pages), incomplete experiment refused, all dashboard pages rendered, 10 downloads produced. |

## 5. Performance and reliability

| Check | Method | Result |
|---|---|---|
| Timeouts | Automated engine tests for timeout status; sandbox per-test/total timeouts | PASS: timeouts are recorded as `timeout`, distinct from errors; sandbox enforces 5 s/test and 20 s total |
| Memory/thread leaks | 30 hardware-sampler lifecycles and 50 analytics passes in one process | Thread count 1 → 1; RSS 144.2 → 148.8 MiB (+4.6 MiB). No unbounded growth observed in this short check; not a long soak. |
| Unhandled exceptions | CLI fault scenarios, dashboard exception boundary, evaluation failures | PASS: typed `ERROR:` messages; dashboard shows a page-level error instead of blanking |
| Interrupted experiments | Automated interrupt and resume tests; captured-interrupt path | PASS: open experiment remains resumable and stored requests are not repeated |
| Database locking | Two connections writing concurrently under WAL + 30 s busy timeout | PASS: 0 errors, 100 rows written |
| Duplicate submissions | Dashboard lock-file tests; duplicate response/recommendation tests | PASS: a live child blocks a second start; stale PIDs are not treated as live; identical writes are idempotent, conflicting writes raise |
| Missing metric handling | Automated sparse-dataset analytics test; stored hardware flags | PASS: missing values stay `None`/blank and are never converted to zero |
| Dashboard responsiveness | Headless page timing on 4 experiments | 0.03–0.91 s per page; Reports export (10 downloads) completed without exception |
| Monitoring overhead | 19 real stored hardware summaries | Average sampler overhead 87.15 ms, peak 366.87 ms; 5–16 samples per response |

## 6. Security review

- **Code execution isolation:** generated code runs only in a fresh `python -I` subprocess with a temporary working directory, no shell, a minimal environment, capped output, and per-test/total timeouts. This is documented as best-effort isolation, not a VM boundary; the subprocess retains the current user's filesystem and network permissions.
- **Subprocess timeouts:** evaluated code and `nvidia-smi` invocations are timeout-bounded; the dashboard child is started without a shell and with `CREATE_NO_WINDOW` on Windows.
- **Input validation:** dataset items, configuration, model tags, CSV/JSON report formats, repetition counts, and item identifiers are validated before use; YAML is loaded with `safe_load`.
- **Secrets:** none found in source, configs, tests, or docs. The application reads environment variables and does not load `.env`.
- **File paths:** runtime paths are resolved against the project root; report output is written only to the requested, explicit directory; dataset loading rejects unsupported extensions.
- **Unsafe model output:** responses are never executed in-process; HTML reports use Jinja2 autoescaping (explicitly enabled for `.j2`); the judge treats candidate text as untrusted JSON data with a prompt-injection warning in the system message.
- **Static scan:** no `shell=True`, `eval`, `exec`, `os.system`, `os.popen`, `pickle`, `yaml.load`, or `verify=False` occurrences were found under `src/`.

## 7. Defects found and fixed

### Defect 1 — A missing model permanently failed the experiment

- **Issue:** selecting a configured candidate that was not installed created an experiment row, marked its run failed, and aborted the remaining models. Because `failed` is a closed status, the experiment could not be resumed and the remaining models never ran.
- **Reproduction:** `run_experiment.py --models qwen2.5:7b-instruct` created `experiments.status = failed`, `runs.status = failed`, and an orchestration error.
- **Fix:** `src/benchmark/execution/orchestrator.py` now checks installed availability through the model manager before creating or resuming an experiment, and raises one clear `BenchmarkError` listing every missing model. Connectivity failures are left to the execution path so they still produce a typed error.
- **Regression test:** `tests/integration/test_experiment_persistence.py::test_missing_model_is_rejected_before_any_experiment_is_created` asserts the error, no experiment rows, and no engine calls.
- **Verification:** the scenario now exits 1 with no experiment rows; the integration suite passes.

### Defect 2 — Experiment History page crashed with two completed experiments

- **Issue:** `src/benchmark/dashboard/views/history.py` read `comparison.limitations`, but `HistoricalComparison` defines no such field. Any database with at least two completed experiments raised `AttributeError`, which the app boundary converted into “This page could not be rendered because of a backend failure.”
- **Reproduction:** Streamlit `AppTest` against the Phase 11 database with four completed experiments.
- **Fix:** removed the invalid aggregate access; per-comparison limitations were already rendered inside the loop.
- **Regression test:** `tests/dashboard/test_dashboard_app.py::test_history_page_renders_comparison_with_two_completed_experiments` seeds two completed experiments and asserts the delta table renders with no exception or error.
- **Verification:** the populated History page renders; Reports export still produces 10 downloads.

## 8. Known failures

There are no unresolved failing tests or unhandled defects.

Expected non-zero or warning results, documented rather than treated as pass:

- `benchmark models verify` exits 1 because two configured candidates are not installed. The installed candidate and judge verify successfully.
- `benchmark health` exits 0 with a warning for those missing candidates, and also exits 0 when Ollama is unreachable (see Limitations).
- The default `pytest` run reports 3 skipped live Ollama tests; they pass with `BENCHMARK_LIVE_OLLAMA=1`.

## 9. Known limitations and tests not executed

- Only one candidate model is installed. No three-model comparison, model ranking, or workload-winner conclusion was produced in this phase; only the single-model quality/composite value exists for the 10-question run.
- A full 100-item quality run, performance probe (9 prompts × 5 repeats), consistency probe (10 items × 3 repeats), and repeated full runs were not executed because of time cost on this CPU-only laptop. Scenario C used one item × 3 repetitions.
- No multi-hour soak, memory-profiler run, or long-duration leak test was performed; the memory check is a short 30-cycle observation.
- The dashboard was validated through Streamlit's `AppTest` and a headless server health check, not by interactive browser inspection or new screenshots.
- No lint or type checker is configured in `pyproject.toml` (no ruff, mypy, black, flake8, or pre-commit). Only byte-compilation and the test suite were available.
- No code-coverage measurement was produced (no coverage tool configured).
- The per-request timeout is applied per HTTP read by `httpx`. A server that keeps sending keep-alive bytes without finishing could exceed the nominal total request duration.
- `run_experiment.py --models` rejects the judge tag while `benchmark infer` accepts it; the judge cannot serve as a fast benchmark candidate.
- `health` treats Ollama unavailability as a warning, so automation that relies on its exit code will not stop when the service is down.
- Hardware values are system-wide; GPU/VRAM attribution on Windows WDDM remains coarse, and model-specific VRAM is unavailable for some responses (flags are stored per row).
- The code sandbox is a restricted subprocess, not a security boundary against determined adversarial code.

## 10. Manual test results

| Manual action | Result |
|---|---|
| `python -m benchmark --help`, `scripts/run_experiment.py --help` | Expected commands and options displayed |
| `python -m benchmark health` | Pass with the expected missing-candidate warning |
| `python scripts/validate_dataset.py` | `Status: VALID`, 100 items, both probes valid |
| `python -m benchmark models verify` | Installed models verified; two missing candidates reported |
| Reports for `phase11-ten` | 9 files plus `manifest.json`; CSV/JSON/HTML/PDF exist and are non-empty |
| Dashboard headless start | Server started on port 8765; health endpoint 200 `ok`; root page 200 |
| Dashboard AppTest (empty and populated) | No uncaught exceptions on any of the 10 pages; History comparison and Reports export verified |
| Database after restart | 4 experiments, 19 responses, 17 evaluations; integrity and foreign keys clean |

## 11. Remaining risks

1. Benchmark conclusions are provisional until the two missing candidates are installed and compared under identical conditions.
2. Results come from a single machine and short runs; variance across runs, thermal state, and memory pressure is not quantified.
3. The blind judge is a small 3B model; judge agreement with human review was not measured in this phase. Judge failures are recorded as missing, not as zero, but low judge coverage lowers confidence.
4. All measured latencies are single-sample or few-sample estimates, not statistically robust distributions.
5. The sandbox limits runtime and output but does not isolate the filesystem or network.
6. SQLite remains a single-writer store; concurrent experiments are prevented only by the dashboard lock file, not by a database-level lease. A stale lock can require manual dismissal.
7. Timeout semantics are per-read rather than wall-clock total, so a pathological stream can outlive the configured timeout.
8. Dashboard caching is keyed on the database file version; unusual filesystems with coarse timestamps could delay fresh reads.

## 12. Readiness assessment

The application is functional and internally consistent for controlled local use: configuration, dataset validation, streaming inference, hardware monitoring, hybrid evaluation, persistence, resumption, analytics, reporting, and the ten-page dashboard all execute against real Ollama and real SQLite. The two defects found during this phase are fixed with regression tests, and the full suite, byte-compilation, dataset validation, health check, live tests, and a real Streamlit startup pass.

It is **not yet ready for three-model comparative conclusions** because two configured candidates are absent and no multi-model ranking was measured. The next step should be an explicit, user-approved model pull followed by a complete 100-item run across all three candidates with repeated measurements.

## Changed files in this phase

- `src/benchmark/execution/orchestrator.py` — pre-flight installed-model check.
- `src/benchmark/dashboard/views/history.py` — removed access to a non-existent attribute.
- `tests/integration/test_experiment_persistence.py` — regression test for the missing-model refusal.
- `tests/dashboard/test_dashboard_app.py` — regression test for the History comparison page.
- `tests/integration/test_ollama_live.py` — opt-in live streaming and hardware sampling tests.
- `README.md` — documents the refusal to start with a missing model.
- `docs/TESTING_REPORT.md` — this report.
