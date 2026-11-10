# Local LLM Benchmarking and Comparative Evaluation

Phase 04 project state for a local benchmark comparing Llama 3.1 8B, Qwen 2.5 7B, and Mistral 7B Instruct v0.3 through Ollama. The versioned 100-item dataset, probes, streaming inference engine, response records, JSONL logging, and tests are implemented.

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
.\.venv\Scripts\python.exe scripts\preflight.py
.\.venv\Scripts\python.exe scripts\validate_dataset.py
.\.venv\Scripts\python.exe scripts\validate_dataset.py --json
.\.venv\Scripts\python.exe -m pytest -q
```

Model downloads are never automatic. A pull requires an explicit confirmation:

```powershell
.\.venv\Scripts\python.exe -m benchmark models pull qwen2.5:7b-instruct --yes
.\.venv\Scripts\python.exe scripts\pull_models.py --yes
```

The pull command refuses to download without `--yes`. The helper script pulls only configured candidates that are not already installed.

## Configuration

- `configs/benchmark.yaml` contains dataset, repetition, timeout, generation, and runtime path settings.
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

- `health` checks Python, dependencies, configuration, runtime directories, the dataset hash/schema, disk space, and Ollama connectivity.
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
- Response records distinguish `success`, `empty`, `timeout`, `error`, and `cancelled` outcomes. Warm-up records are stored but excluded from measured counts.
- Live tests are opt-in: set `BENCHMARK_LIVE_OLLAMA=1` before running pytest.

The two missing candidate models remain absent until the user explicitly approves a pull. The heuristic compatibility estimate is not a substitute for measuring actual layer placement with `/api/ps`.

## Scope

Phases 01-04 provide configuration, logging, health checks, Ollama access, model discovery, metadata collection, verification, explicit pull support, lifecycle checks, metadata snapshots, the validated 100-item dataset with performance and consistency probes, and the streaming inference engine with structured response records. Evaluation, database storage, analytics, dashboard, and reporting are implemented in later phases.
