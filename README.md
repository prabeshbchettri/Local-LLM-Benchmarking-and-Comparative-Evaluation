# Local LLM Benchmarking and Comparative Evaluation

Phase 10 project state for a local benchmark comparing Llama 3.1 8B, Qwen 2.5 7B, and Mistral 7B Instruct v0.3 through Ollama. The versioned 100-item dataset, probes, streaming inference engine, response records, JSONL logging, hardware monitoring, resource summaries, hybrid response evaluation, SQLite persistence with a resumable experiment orchestrator, workload-specific analytics, scoring, recommendations, historical comparison, CSV/JSON/HTML/PDF reporting with reproducibility manifests, and the interactive Streamlit dashboard (ten pages) are implemented.

## Requirements

- Windows 11
- Python 3.13.x
- Ollama 0.40.15 or compatible, reachable at `http://127.0.0.1:11434`
- At least 12 GiB free disk space for the two missing candidate models

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

## Commands

```powershell
.\.venv\Scripts\python.exe -m benchmark init
.\.venv\Scripts\python.exe -m benchmark health
.\.venv\Scripts\python.exe -m benchmark models list
.\.venv\Scripts\python.exe -m benchmark models list --all
.\.venv\Scripts\python.exe -m benchmark models verify
.\.venv\Scripts\python.exe -m benchmark models show llama3.1:8b
.\.venv\Scripts\python.exe -m benchmark models test qwen2.5:3b --max-tokens 16 --unload
.\.venv\Scripts\python.exe -m benchmark models load qwen2.5:3b
.\.venv\Scripts\python.exe -m benchmark models ps
.\.venv\Scripts\python.exe -m benchmark models unload qwen2.5:3b
.\.venv\Scripts\python.exe -m benchmark models select judge
.\.venv\Scripts\python.exe -m benchmark models snapshot
.\.venv\Scripts\python.exe -m benchmark infer --model llama3.1:8b --limit 5 --warmup 1
.\.venv\Scripts\python.exe -m benchmark infer --model llama3.1:8b --item gk-002 --item lr-001 --output data\results\smoke.jsonl
.\.venv\Scripts\python.exe -m benchmark infer --model llama3.1:8b --item gk-002 --baseline-seconds 1 --hardware-interval-ms 500
.\.venv\Scripts\python.exe -m benchmark infer --model llama3.1:8b --item gk-002 --no-monitor
.\.venv\Scripts\python.exe -m benchmark monitor --seconds 5
.\.venv\Scripts\python.exe -m benchmark monitor --seconds 3 --interval-ms 1000 --json
.\.venv\Scripts\python.exe -m benchmark analyze EXPERIMENT_ID
.\.venv\Scripts\python.exe -m benchmark analyze EXPERIMENT_ID --all-profiles --json
.\.venv\Scripts\python.exe -m benchmark analyze EXPERIMENT_ID --profile coding --persist
.\.venv\Scripts\python.exe -m benchmark analyze EXPERIMENT_ID --run-profile quality --include-cold-start
.\.venv\Scripts\python.exe -m benchmark compare BASELINE_ID CANDIDATE_ID --json
.\.venv\Scripts\python.exe -m benchmark report EXPERIMENT_ID
.\.venv\Scripts\python.exe -m benchmark report EXPERIMENT_ID --format csv --format json --output-dir data\exports\review
.\.venv\Scripts\python.exe scripts\generate_report.py EXPERIMENT_ID --format html --json
.\.venv\Scripts\python.exe scripts\preflight.py
.\.venv\Scripts\python.exe scripts\validate_dataset.py
.\.venv\Scripts\python.exe scripts\validate_dataset.py --json
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m streamlit run src\benchmark\dashboard\app.py
```

Model downloads are never automatic. A pull requires an explicit confirmation:

```powershell
.\.venv\Scripts\python.exe -m benchmark models pull qwen2.5:7b-instruct --yes
.\.venv\Scripts\python.exe scripts\pull_models.py --yes
```

The pull command refuses to download without `--yes`. The helper script pulls only configured candidates that are not already installed.

## Dashboard

Start the dashboard from the repository root:

```powershell
.\.venv\Scripts\python.exe -m streamlit run src\benchmark\dashboard\app.py
```

The dashboard reads `storage/benchmark.db` and the validated dataset. It has ten pages: Overview, Model Comparison, Performance Analysis, Quality Evaluation, Hardware Monitoring, Response Inspection, Experiment History, Reports, Recommendations, and Run New Experiment. Every value is computed from stored responses by the analytics engine and repositories; no benchmark result is hardcoded.

- The Overview page shows registered models, completed experiments, benchmark question count, the latest experiment, the overall ranking, the best model for each configured workload, and aggregate performance statistics.
- Model Comparison and Performance Analysis use the selected scoring profile and can include or exclude cold-start runs explicitly. Missing measurements remain blank rather than being treated as zero.
- Quality Evaluation separates deterministic checks from blind LLM judge scores and reports coding-test, JSON-validity, and instruction-following pass rates from stored evaluations.
- Hardware Monitoring shows average and peak CPU, RAM, GPU, VRAM, and temperature values plus per-response timelines. Hardware is system-wide and is not exclusively attributable to a model.
- Response Inspection shows the stored prompt, expected or reference answer, generated answer, deterministic checks, judge dimensions, latency, token counts, errors, and hardware summary for one model and question.
- Experiment History compares completed experiments on the same profile and shows absolute and percent deltas per model.
- Reports builds CSV, JSON, HTML, and PDF downloads from the selected experiment through the same reporting service as the CLI. Only completed experiments are exported as final results; incomplete experiments are refused.
- Recommendations recompute workload-specific rankings and explanations from the database on every render; the dashboard does not persist new recommendations.
- Run New Experiment starts `scripts/run_experiment.py` in a child process. Streamlit is never blocked by inference. A lock file under `storage/dashboard_runs/active_run.json` prevents duplicate submissions, and the child log is available on the page. The form accepts a dataset, models, repetition count, question scope, warm-up requests, and cold-start execution. If the child fails before creating an experiment, stored partial results are preserved and the log explains the failure.

The dashboard caches reads briefly but includes the SQLite file version in the cache key, so rows written by a running experiment appear without restarting Streamlit. Use **Refresh data** in the sidebar to clear cached queries.

## Configuration

- `configs/benchmark.yaml` contains dataset, repetition, timeout, generation, runtime path, hardware monitoring settings (`hardware_monitoring_enabled`, `hardware_sample_interval_ms`, `hardware_baseline_seconds`), `default_scoring_profile`, and the five validated `workload_profiles`. Profile weights must total 100 and category weights must total 1.0.
- `configs/models.yaml` contains the Ollama endpoint, keep-alive value, candidate tags, judge tag, expected digests, and expected quantization.
- `configs/evaluation.yaml` contains judge settings, validation fractions, and category judge weights.

Environment overrides:

- `BENCHMARK_PROJECT_ROOT`
- `BENCHMARK_CONFIG_DIR`
- `BENCHMARK_LOG_LEVEL`

`.env.example` documents the supported variables. The application reads process environment variables and does not load `.env` automatically.

## Dataset

- Canonical dataset: `data/datasets/benchmark_v1.json` (`benchmark-v1.0.0`, schema `1.0.0`).
- Compatibility export: `data/datasets/benchmark_v1.0.0.jsonl` plus `benchmark_v1.0.0.manifest.json`.
- Performance probe: `data/datasets/probes/performance_v1.0.0.json` (9 prompts, 5 repeats).
- Consistency probe: `data/datasets/probes/consistency_v1.0.0.json` (10 stratified items, 3 repeats).
- Item JSON Schema: `data/datasets/schemas/benchmark_item.schema.json`.
- The canonical dataset has exactly 100 items: 15 each in general knowledge, logical reasoning, mathematical reasoning, coding, instruction following, and structured JSON, plus 10 summarization items.
- Difficulty distribution: 30 easy, 47 medium, 23 hard. Six items are unanswerable and use `refusal_check`.
- The loader verifies the SHA-256 content hash and every item-defined JSON Schema. See `docs/DATASET_GUIDE.md`.

## Implemented behavior

- `health` checks Python, dependencies including `psutil` and `nvidia-ml-py`, configuration, runtime directories, the dataset hash/schema, disk space, `nvidia-smi`, and Ollama connectivity.
- `scripts/validate_dataset.py` validates the dataset, both probes, the content hash, and all item-defined JSON Schemas; `--json` prints a machine-readable summary.
- The dataset loader supports canonical JSON and JSONL plus a sidecar manifest, deterministic seeded ordering, and atomic JSONL export.
- `models list` shows configured candidates and the judge, including missing models. `--all` also shows unconfigured installed models.
- `models verify` compares installed digests and quantization with `configs/models.yaml`.
- `models show` displays tag, digest, size, family, parameter count, quantization, context length, embedding length, capabilities, license, Ollama version, verification, and a heuristic memory-fit assessment.
- `models load` and `models unload` confirm state changes with `/api/ps`.
- `models test` sends a short non-streaming `/api/chat` request.
- `models snapshot` writes `storage/model_metadata.json` by default. Use `--output` to choose another path and `--all` to include unconfigured installed models.
- `infer` streams `/api/chat` responses, measures TTFT and total latency with a monotonic clock, captures Ollama token counts and durations, and appends structured records to a JSONL file under `data/results/` by default.
- `infer` supports selected models and item IDs, repeats, warm-up requests, cold-start unloads, cooperative cancellation, and `--json` summaries. The default selection is five ordered dataset items.
- `infer` monitors hardware by default. Each response record carries a `hardware` summary with baseline, inference, and overall windows, raw samples, availability flags, and sampler overhead. Use `--no-monitor` to disable sampling, or `--hardware-interval-ms` and `--baseline-seconds` to override the configured interval and baseline.
- `monitor` samples hardware without running inference; it supports `--seconds`, `--interval-ms`, and `--json`.
- Hardware collection covers system-wide CPU and RAM, Ollama process RSS, GPU utilization, system VRAM, GPU temperature, and Ollama model VRAM from `/api/ps`.
- Response records distinguish `success`, `empty`, `timeout`, `error`, and `cancelled` outcomes. Warm-up records are stored but excluded from measured counts.
- Response schema `1.1.0` adds `hardware` and `hardware_error` to each record.
- `analyze` computes stored-response statistics, quality coverage, resource usage, min-max normalized dimensions, workload-specific composites, and ranked recommendations. It supports `general_purpose`, `coding`, `reasoning`, `json_generation`, and `resource_constrained`, or `--all-profiles`.
- `compare` reports per-model deltas between experiments on the same scoring profile. Use `--json` for machine-readable output.
- `report` and `scripts\generate_report.py` write experiment_summary, model_comparison, performance_metrics, hardware_metrics, evaluation_results, and responses CSVs, plus versioned JSON, self-contained HTML, and ReportLab PDF, and a `manifest.json` with byte sizes and SHA-256 hashes. See `docs/REPORTING_GUIDE.md`.
- Reports refuse non-completed experiments unless `--allow-incomplete` is set, in which case every format is marked preliminary. Missing measurements are exported as blank cells or JSON `null`, never zero.
- `--format` is repeatable; the default is all four formats. `--profile`, `--run-profile`, `--include-cold-start`, and `--include-smoke` apply the same analytics filters as `analyze`.
- HTML escapes all stored text; Unicode response text is preserved in CSV, JSON, and HTML. The PDF is checked for readability and section completeness in tests.
- Analytics exclude smoke and cold-start runs by default; use `--include-smoke` or `--include-cold-start` to include them. `--run-profile` selects a stored inference profile.
- A model above a 20% failure rate cannot rank first. Scores within 2 points are disclosed as practical ties. Missing relative metrics are dropped for every compared model rather than imputed.
- `--persist` stores recommendations in SQLite with the full evidence payload.
- Database schema `1.1.0` additively adds `recommendations.details_json`; see `docs/SCORING_AND_RECOMMENDATIONS.md`.
- Live tests are opt-in: set `BENCHMARK_LIVE_OLLAMA=1` before running pytest.

The two missing candidate models remain absent until the user explicitly approves a pull. The heuristic compatibility estimate is not a substitute for measuring actual layer placement with `/api/ps`.

## Hardware monitoring

- CPU, RAM, and GPU values are system-wide and include the operating system and other applications. They are not exclusively attributable to the model.
- Ollama RSS is process-specific only when a matching Ollama process is found; matched process names are stored with the summary.
- NVML is preferred for GPU telemetry, with `nvidia-smi` as a fallback. Unavailable values remain `null` and carry availability flags.
- Windows WDDM may report GPU utilization or other sensors as unavailable or zero. A real run can show coarse baseline values and missing inference samples; the stored data records what was observed rather than inferring values.
- The default sampling interval is 1000 ms; the CLI accepts 250-1000 ms. The default baseline is 2.0 seconds.
- A real laptop check with `llama3.1:8b` produced a successful monitored response: 28 samples (2 baseline, 26 inference) with a 1.0 s baseline, CPU average 71.4% (peak 94.3%), GPU utilization average 25.6% (peak 33.0%), system VRAM average 2.42 GiB, RAM average 12.09 GiB, GPU temperature average 40.3 C, Ollama model VRAM 2.36 GiB of 5.63 GiB (42.0%), and sampler overhead average 106.07 ms (peak 239.26 ms). This is engineering validation, not a comparative result.

## Persistence

- The system of record is `storage/benchmark.db` (SQLite, WAL, foreign keys enabled), created automatically on first use. See `docs/DATABASE_SCHEMA.md`.
- `scripts/run_experiment.py` runs a persisted experiment: models execute sequentially, every response and its metrics are saved as it completes, and evaluations are stored after the model unloads.
- An unfinished experiment can be resumed with `--resume EXPERIMENT_ID`. Completed runs reject new responses and completed experiments reject new runs.
- An experiment refuses to start when a selected model is not installed, so a missing model cannot leave a permanently failed experiment behind.
- `analyze --persist` stores ranked recommendations and full evidence in the `recommendations` table. Existing schema `1.0.0` databases migrate additively on open.

## Scope

Phases 01-10 provide configuration, logging, health checks, Ollama access, model discovery, metadata collection, verification, explicit pull support, lifecycle checks, metadata snapshots, the validated 100-item dataset with performance and consistency probes, the streaming inference engine with structured response records, hardware monitoring with baseline/inference summaries, hybrid evaluation with deterministic evaluators and a blind Qwen 2.5 3B judge, SQLite persistence with a resumable orchestrator, workload-specific analytics, scoring, recommendations, historical comparison, CSV/JSON/HTML/PDF reporting with reproducibility manifests, and the interactive Streamlit dashboard. See `docs/SCORING_AND_RECOMMENDATIONS.md`, `docs/EVALUATION_METHODOLOGY.md`, `docs/REPORTING_GUIDE.md`, and `docs/DATABASE_SCHEMA.md`.
