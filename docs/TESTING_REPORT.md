# Testing Report - Phase 12

Date: 2026-11-14

Scope: final validation of the complete application after the Phase 12 experiments, analysis, reporting, and documentation work. This report records executed checks only. No test or benchmark result is reported without a corresponding run.

## Environment

| Item | Value |
|---|---|
| Operating system | Windows 11, PowerShell 5.1 |
| Python | 3.13.5 (`.venv`) |
| Ollama | 0.40.16 at `http://127.0.0.1:11434` |
| Installed configured models | `llama3.1:8b`, `qwen2.5:7b-instruct`, `mistral:7b-instruct-v0.3-q4_K_M`, `qwen2.5:3b` |
| Dataset | `benchmark-v1.0.0`, 100 items, SHA-256 `ceafe917235213c710da9e3b6a1ceecb86ba519b214581f683191c51358e46f1`, status VALID |
| Configuration hash | `34c5793a1e4075824beb5db8f207dd02ca641bd9d1705f49b53ff6f2e1b039ac` |
| Phase 12 database | `storage/phase12/phase12.db` (gitignored) |
| Report bundles | `data/exports/phase12-main/`, `phase12-coldstart/`, `phase12-consistency/` (gitignored) |

No model was pulled during Phase 12. All four configured models were already installed and verified.

## Validation summary

| Check | Command | Result |
|---|---|---|
| Byte-compile | `python -m compileall -q src tests scripts` | PASS (exit 0) |
| Full test suite | `python -m pytest -q` | **330 passed, 3 skipped**, 83.12 s (exit 0) |
| Live Ollama tests | `$env:BENCHMARK_LIVE_OLLAMA='1'; python -m pytest tests/integration/test_ollama_live.py -q` | **3 passed**, 19.25 s (exit 0) |
| Health | `python -m benchmark health` | PASS; all 4 configured models installed; 160.4 GiB free |
| Model verification | `python -m benchmark models verify` | PASS; all 4 models verified |
| Dataset validation | `python scripts/validate_dataset.py` | VALID; 100 items; probes valid |
| Database integrity | `PRAGMA integrity_check` / `PRAGMA foreign_key_check` | `ok` / no rows |
| Dashboard health | `streamlit run src/benchmark/dashboard/app.py --server.headless=true --server.port=8507` | `/_stcore/health` = 200 `ok`; root page = 200 |
| Report bundles | Inspect files and manifests | 10 files per bundle (9 artifacts + manifest); PDFs 9 pages |

The 3 skipped tests are the opt-in live Ollama tests. They pass when `BENCHMARK_LIVE_OLLAMA=1` is set, as shown above.

## 1. Automated test suite

The default suite passed with **330 passed, 3 skipped in 83.12 s**. The suite covers:

- Configuration loading, validation, hashing, path resolution, and non-UTF-8 errors.
- Logging setup and redaction behavior.
- Dataset models, loader, JSON Schema checks, hash verification, and probe validation.
- Ollama client behavior, streaming NDJSON parsing, cancellation, keep-alive lines, and total deadlines.
- Inference engine timing, TTFT, token counts, throughput, and failure records.
- Model manager discovery, verification, metadata, and pull approval.
- Hardware monitor summaries, fake sensors, NVIDIA fallback, and sampler behavior.
- Deterministic evaluators: exact match, numeric, code tests, JSON Schema, instruction constraints, reference overlap, and refusal checks.
- Judge parsing, invalid JSON handling, scoring, and category-level judge behavior.
- SQLite repositories, migrations, immutable runs, resume behavior, and error ownership.
- Analytics dimensions, filters, metrics, workload profiles, recommendation gates, confidence, and comparison.
- Reporting CSV/JSON/HTML/PDF exports, manifest hashes, incomplete-run handling, and spreadsheet-formula neutralization.
- Dashboard pages, data loading, and experiment-runner locking.

No test was skipped for a missing dependency or an unexplained reason. The only skips are the three live Ollama tests that require explicit opt-in.

## 2. Phase 12 regression tests

The main Phase 12 defect was the keep-alive deadline reset:

- `tests/unit/test_ollama_client.py::test_blank_keepalive_lines_respect_total_deadline` verifies that blank keep-alive lines cannot extend a stream beyond the configured wall-clock deadline.
- The fix is `OllamaClient._deadline_lines` in `src/benchmark/execution/ollama_client.py`. It checks the deadline before every line, including blank lines, so `httpx` per-read timeout resets cannot defeat the total request timeout.
- The main Qwen run recorded a timeout on `lr-007` after **1,839 s** before the fix, even though `request_timeout_seconds` is 300. The timeout remains counted as a failure in the main results. Post-fix cold-start and consistency runs had zero timeouts.

Phase 12 also added coverage for the new `--database` option:

- `tests/analytics/test_analytics_cli.py` verifies that `benchmark analyze` and `benchmark compare` honor an explicit database path instead of the configured default.

## 3. Integration and live Ollama tests

| Area | Evidence | Result |
|---|---|---|
| Ollama connectivity | Live `version` and `/api/tags` | PASS |
| Model inventory | `benchmark health`, `benchmark models verify` | PASS for all 4 configured models |
| Streaming inference | Live NDJSON stream test | PASS |
| Inference smoke | Stored responses from the three Phase 12 experiments | PASS |
| Dataset loading | `validate_dataset.py`; canonical JSON and JSONL | PASS |
| Persistence | SQLite rows for 3 experiments, 9 runs, 438 responses | PASS |
| Resume/idempotency | Test suite and prior interrupted-run checks | PASS |
| Reporting | Three generated bundles; manifests verified | PASS |
| Dashboard | Health endpoint and root page | PASS |

## 4. Database validation

`storage/phase12/phase12.db` reported:

| Check | Result |
|---|---|
| `PRAGMA integrity_check` | `ok` |
| `PRAGMA foreign_key_check` | no rows |
| Experiments | 3 |
| Runs | 9 |
| Responses | 438 (435 measured + 3 warm-ups) |
| Evaluations | 435 |
| Stored errors | 0 |

The main experiment has 303 responses and 300 evaluations. The cold-start and consistency experiments have 45 and 90 measured responses respectively, each with matching evaluations. The Qwen timeout is recorded as a failed response and is not counted as a successful evaluation.

## 5. Reporting validation

Each Phase 12 bundle contains:

- `experiment_summary.csv`
- `model_comparison.csv`
- `performance_metrics.csv`
- `hardware_metrics.csv`
- `evaluation_results.csv`
- `responses.csv`
- versioned JSON
- self-contained HTML
- ReportLab PDF (9 pages)
- `manifest.json` with byte sizes and SHA-256 hashes

All three manifests name the expected experiment IDs:

| Bundle | Experiment ID |
|---|---|
| `phase12-main` | `20261114T074040Z-9edf76` |
| `phase12-coldstart` | `20261114T095758Z-c3ae3a` |
| `phase12-consistency` | `20261114T102018Z-e114a1` |

## 6. Dashboard validation

Streamlit started headless on port 8507. The health endpoint returned HTTP 200 with body `ok`, and the root page returned HTTP 200. The process was stopped after the check.

The dashboard has ten pages: Overview, Model Comparison, Performance Analysis, Quality Evaluation, Hardware Monitoring, Response Inspection, Experiment History, Reports, Recommendations, and Run New Experiment. It reads stored responses, evaluations, and metrics; no benchmark value is hardcoded.

## 7. Defects found and fixed

| Defect | Impact | Fix | Verification |
|---|---|---|---|
| Blank keep-alive lines reset the HTTP read timeout | One Qwen request ran 1,839 s against a 300 s limit | Enforce the wall-clock deadline before every line in `_deadline_lines` | Regression test passes; post-fix subsets had zero timeouts |
| `analyze`/`compare` lacked `--database` | Required copying configuration or databases to analyze a non-default DB | Added `--database` to both CLI commands | New CLI test passes; Phase 12 analytics used the explicit path |
| Qwen `lr-007` timeout | Qwen success rate 99.0%; latency/throughput exclude the failed request | Recorded as a failure; no retry; disclosed in results | Stored error row; 0 unexplained errors |

## 8. Limitations of this validation

- No lint, static type checker, or coverage tool is configured. Validation used byte-compilation, the full test suite, live Ollama tests, health checks, dataset validation, database integrity checks, report inspection, and dashboard health.
- The full test suite does not exercise the full 100-item experiment; that was validated by the three real Phase 12 experiments and their stored rows.
- The dashboard check confirms startup and HTTP health, not a manual click-through of every page.
- Hardware monitoring depends on `nvidia-smi`/NVML and the GTX 1650; fake-sensor tests cover fallback logic but not every GPU model.
- The live tests use the installed models and current Ollama version; they do not pin model behavior across versions.
- Detached background experiment processes died silently during Phase 12. All final experiments were run in the foreground. Any rerun should use foreground execution and verify the process remains alive.
- The consistency run used temperature 0.0, not the probe's 0.7. This is a methodology limitation, not a test failure.

## 9. Conclusion

Phase 12 validation passed. The application compiles, the default and live test suites pass, all configured models are installed and verified, the dataset is valid, the database is internally consistent, all three report bundles are complete with verified manifests, and the dashboard starts and serves its health endpoint. The remaining limitations are disclosed rather than hidden: reduced workload, single machine, judge coverage and bias, consistency temperature, and missing lint/type/coverage tooling.
