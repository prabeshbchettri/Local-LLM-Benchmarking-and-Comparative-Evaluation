# Database Schema

**Status:** Phase 07 implementation (schema version `1.0.0`)
**Engine:** SQLite 3 through SQLAlchemy 2.0 ORM
**Default file:** `storage/benchmark.db`

## Purpose

The database is the durable record of every benchmark experiment. Each experiment
stores the exact configuration that produced it, every model run, every prompt
response (including failures and warm-ups), timing and token metrics, hardware
samples when monitoring is enabled, deterministic and judge evaluations, and the
recommendations generated from those results.

JSONL files remain useful for streaming inspection, but they are no longer the
system of record. Historical rows are never overwritten: a completed run rejects
new responses, a completed or failed experiment rejects new runs, and re-inserting
an identical record is idempotent while a conflicting payload raises an error.

## Connection Behavior

`open_database(settings, path=None)` creates the parent directory, opens one engine
per process, and applies:

| Pragma | Value | Reason |
| --- | --- | --- |
| `foreign_keys` | `ON` | Reject orphan runs, responses, metrics, and evaluations |
| `journal_mode` | `WAL` | Allow readers while an experiment writes |
| `synchronous` | `NORMAL` | Durable enough with WAL, faster than `FULL` |
| `busy_timeout` | `30000` ms | Wait instead of failing under short write contention |

Sessions are short: `Database.session_scope()` commits once on success and rolls
back the whole unit on any exception. Repositories never expose detached ORM
objects; they return Pydantic read views.

## Schema Versioning

`schema_versions` records the applied version. On startup `apply_schema` creates
missing tables, inserts `1.0.0` for a new database, and refuses to open a database
whose stored version is newer than the running code. Future changes are registered
in `MIGRATIONS` and must be additive (new tables, indexes, or nullable columns).

There is no Alembic dependency. A backup of `storage/benchmark.db` is the required
step before upgrading a database used for published results.

## Tables

### `models`

One row per candidate or judge artifact.

| Column | Type | Notes |
| --- | --- | --- |
| `id` | integer PK | |
| `tag` | text, unique, indexed | Ollama tag, for example `llama3.1:8b` |
| `name` | text | Configured display name |
| `role` | text, indexed | `candidate` or `judge` |
| `installed` | boolean | Verified from Ollama at registration time |
| `digest` | text, indexed | Model digest when available |
| `family`, `parameter_count`, `parameter_size` | text/integer | From Ollama metadata |
| `quantization`, `size_bytes`, `context_length`, `embedding_length` | | From Ollama metadata |
| `license`, `ollama_version` | text | Provenance |
| `model_info` | JSON | Raw `/api/show` payload |
| `first_seen_at`, `last_seen_at` | datetime | Registration history |

Re-registering a model refreshes metadata and `last_seen_at`; it never changes an
existing run or response.

### `datasets`

| Column | Type | Notes |
| --- | --- | --- |
| `id` | integer PK | |
| `name`, `version` | text | Unique together (`uq_dataset_name_version`) |
| `schema_version` | text | Dataset schema version |
| `content_sha256` | text, indexed | Canonical dataset hash |
| `item_count` | integer | Number of items |
| `source_path` | text | Path used at registration |
| `random_seed`, `registered_at` | integer/datetime | |

Registering the same name and version with a different hash raises
`DatabaseConflictError`; a new version is required for changed content.

### `experiments`

| Column | Type | Notes |
| --- | --- | --- |
| `id` | integer PK | |
| `experiment_id` | text, unique, indexed | External identifier, for example `20261110T101500Z-a1b2c3` |
| `name` | text, nullable | Operator label |
| `dataset_id` | FK → `datasets.id` | |
| `config_hash` | text, indexed | Hash from the validated configuration |
| `status` | text, indexed | `created`, `running`, `completed`, `failed`, `cancelled` |
| `created_at`, `started_at`, `completed_at` | datetime | |
| `config_json` | JSON | Full validated settings snapshot |
| `host_json` | JSON | Hostname, platform, CPU count, memory |
| `software_json` | JSON | Package versions |
| `notes` | text | Failure message or operator notes |

`config_hash` is intentionally **not unique**: repeating an experiment with the same
configuration is a legitimate historical event. This differs from the original
architecture sketch and is recorded here as a deliberate decision.

### `runs`

One model's execution group inside an experiment.

| Column | Type | Notes |
| --- | --- | --- |
| `id` | integer PK | |
| `experiment_id` | FK → `experiments.id`, indexed | |
| `model_id` | FK → `models.id`, indexed | |
| `profile` | text, indexed | `quality`, `performance`, `consistency`, `cold_start`, `smoke` |
| `status` | text, indexed | `running`, `completed`, `failed`, `cancelled` |
| `cold_start` | boolean | Part of run identity |
| `warmup_requests`, `repetitions` | integer | Request plan |
| `started_at`, `completed_at` | datetime | |
| `notes`, `error_message` | text | Failure evidence |

Unique on `(experiment_id, model_id, profile, cold_start)` (`uq_run_identity`) and
indexed on `(experiment_id, status)`.

### `responses`

One prompt-response record, including failed requests and warm-ups.

| Column | Type | Notes |
| --- | --- | --- |
| `id` | integer PK | |
| `run_id` | FK → `runs.id`, indexed | |
| `response_uuid` | text, unique | Application-level response id |
| `question_id` | text | Dataset item id or probe id |
| `category` | text, indexed | Dataset category when applicable |
| `prompt`, `system_prompt` | text | Exact request content |
| `response_text` | text | Empty for failures |
| `finish_reason`, `status` | text | `success`, `timeout`, `connection_error`, `http_error`, `invalid_response`, `cancelled` |
| `error_type`, `error_message` | text | Failure evidence |
| `is_warmup`, `cold_start` | boolean | Part of response identity |
| `repetition` | integer | 1-based |
| `started_at`, `completed_at` | datetime | |
| `stream_chunk_count` | integer | Streaming evidence |
| `config_hash` | text, nullable | Per-request configuration hash |
| `raw_json` | JSON | Full `ResponseRecord` payload |
| `created_at` | datetime | Insert time |

Unique on `(run_id, question_id, repetition, is_warmup)` (`uq_response_identity`)
and indexed by `(run_id, question_id)` and `question_id`.

### `performance_metrics`

One row per response (unique `response_id`). Stores `ttft_seconds`,
`total_latency_seconds`, Ollama `load_duration_seconds`,
`prompt_eval_duration_seconds`, `eval_duration_seconds`, prompt and eval counts,
derived `output_tokens_per_second` and `prompt_tokens_per_second`, and the effective
`options_json`.

A metric that could not be measured is `NULL`, never zero.

### `hardware_measurements`

One row per response (unique `response_id`) when hardware monitoring was enabled.
Stores the summary fields (`cpu_*`, `ram_*`, `gpu_utilization_*`, `vram_*`,
`gpu_temperature_*`, Ollama RSS and model VRAM), availability flags, GPU identity
and source, sampler overhead, error notes, plus the full `summary_json` and every
raw `samples_json` entry.

Rows are omitted entirely when no sampler ran. Availability flags make missing
GPU, temperature, or VRAM data explicit.

### `evaluations`

One row per response (unique `response_id`).

| Column | Type | Notes |
| --- | --- | --- |
| `id` | integer PK | |
| `response_id` | FK → `responses.id`, unique | |
| `question_id`, `category` | text, indexed | Denormalized for querying |
| `evaluation_type` | text | Deterministic method |
| `evaluator`, `evaluator_version` | text | Method provenance |
| `status` | text, indexed | `passed`, `partial`, `failed`, `skipped`, `error` |
| `deterministic_score`, `deterministic_max_score`, `deterministic_normalized_score` | float | |
| `passed` | boolean | Deterministic pass flag |
| `judge_status`, `judge_model_tag`, `judge_model_digest`, `judge_same_family` | text/boolean | Judge provenance |
| `judge_raw_score`, `judge_normalized_score` | float | Null when the judge did not run |
| `judge_missing` | boolean | Judge coverage flag |
| `composite_score` | float | Category-weighted combination |
| `checks_json` | JSON | Per-check deterministic evidence |
| `judge_dimensions_json`, `judge_config_json` | JSON | Judge rubric evidence |
| `errors_json` | JSON | Structured evaluation errors |
| `created_at` | datetime | |

### `recommendations`

| Column | Type | Notes |
| --- | --- | --- |
| `id` | integer PK | |
| `experiment_id` | FK → `experiments.id`, indexed | |
| `model_id` | FK → `models.id`, indexed | |
| `workload` | text, indexed | Workload name |
| `rank` | integer | 1 = best |
| `score`, `confidence` | float/text | |
| `weights_json` | JSON | Scoring weights used |
| `rationale` | text | Explanation |
| `created_at` | datetime | |

Unique on `(experiment_id, workload, rank)` (`uq_recommendation_rank`). Phase 08
computes recommendations; Phase 07 only provides durable storage.

### `errors`

Structured failures that are not tied to one response.

| Column | Type | Notes |
| --- | --- | --- |
| `id` | integer PK | |
| `experiment_id` | FK → `experiments.id`, nullable, indexed | |
| `run_id` | FK → `runs.id`, nullable, indexed | |
| `stage` | text | `orchestrator`, `evaluation`, ... |
| `error_type`, `message` | text | |
| `occurred_at` | datetime | |

## Relationship Diagram

```text
datasets 1 ─── n experiments 1 ─── n runs 1 ─── n responses 1 ─── 1 performance_metrics
                                     │              │        ├── 1 hardware_measurements
                                     │              │        └── 1 evaluations
                                     └── n errors   │
models 1 ─── n runs                                 │
models 1 ─── n recommendations                      │
experiments 1 ─── n recommendations                 │
experiments 1 ─── n errors ─────────────────────────┘
```

All foreign keys are enforced by SQLite for connections opened through
`Database`. A raw `sqlite3` shell connection does not inherit these pragmas, so run
`PRAGMA foreign_keys=ON;` before manual writes. Deleting a parent row is not part of
any workflow; there is no cascade delete because historical evidence must be preserved.

## Integrity Rules

| Situation | Behavior |
| --- | --- |
| Same response saved twice with identical payload | Returns the existing id (idempotent) |
| Same response key with different payload | `DatabaseDuplicateError` |
| Response saved into a completed run | `ExperimentStateError` |
| New run in a completed, failed, or cancelled experiment | `ExperimentStateError` |
| Same evaluation saved twice with identical scores/status | Returns existing id |
| Different evaluation for the same response | `DatabaseDuplicateError` |
| Dataset version re-registered with a different hash | `DatabaseConflictError` |
| Same recommendation rank with different model/score | `DatabaseDuplicateError` |
| Foreign key points at a missing row | SQLite `IntegrityError`, transaction rolled back |
| Failure inside a multi-row save | Whole transaction rolled back |

Evaluations remain appendable after a run is completed, so an interrupted or
manually triggered evaluation pass can be finished later.

## Example Records

```text
experiments
  experiment_id: 20261110T101500Z-a1b2c3
  name:          phase07-demo
  config_hash:   6f1c...e9
  status:        completed
  config_json:   {"benchmark": {...}, "models": {...}, "evaluation": {...}}

runs
  id 1 | llama3.1:8b | profile quality | status completed | repetitions 1

responses
  question_id gk-001 | status success | repetition 1 | is_warmup 0

performance_metrics
  total_latency_seconds 3.42 | ttft_seconds 0.41 | eval_count 128

evaluations
  question_id gk-001 | evaluation_type exact_match | status passed
  deterministic_normalized_score 1.0 | judge_status not_eligible | composite 1.0
```

Failed requests are stored with `status != success`, an `error_type`, and an
evaluation row whose status is `skipped` (when evaluation ran).

## SQL Inspection

```bash
sqlite3 storage/benchmark.db ".tables"
sqlite3 storage/benchmark.db ".schema responses"
sqlite3 storage/benchmark.db "SELECT version, applied_at FROM schema_versions;"
sqlite3 storage/benchmark.db "SELECT experiment_id, status, created_at FROM experiments ORDER BY id DESC LIMIT 5;"
sqlite3 -header -column storage/benchmark.db \
  "SELECT r.id, m.tag, r.profile, r.status, COUNT(p.id) AS responses
   FROM runs r
   JOIN models m ON m.id = r.model_id
   LEFT JOIN responses p ON p.run_id = r.id
   GROUP BY r.id;"
sqlite3 -header -column storage/benchmark.db \
  "SELECT question_id, status, total_latency_seconds
   FROM responses JOIN performance_metrics ON performance_metrics.response_id = responses.id
   WHERE run_id = 1 LIMIT 10;"
```

The `sqlite3` shell may not be installed on Windows; the same queries can be run
through Python:

```bash
.venv/Scripts/python.exe -c "import sqlite3; c=sqlite3.connect('storage/benchmark.db'); print(c.execute('SELECT name FROM sqlite_master WHERE type=\"table\"').fetchall())"
```

## Manual Persistence Test

```bash
.venv/Scripts/python.exe scripts/run_experiment.py --limit 1 --warmup 0 --no-judge --name persistence-demo
.venv/Scripts/python.exe -c "from benchmark.core.config import load_settings; from benchmark.database import open_database; from benchmark.database.repositories import Repositories; s=load_settings(); d=open_database(s); r=Repositories(d); print(r.experiments.history()); print(r.reads.model_results(r.experiments.history(1)[0].experiment_id))"
```

Restarting the process and rerunning the second command must show the same rows,
which proves the record is on disk rather than in memory.

## Known Deviations From `docs/ARCHITECTURE.md`

- `experiment_configs` is folded into `experiments.config_json` plus `config_hash`.
- `dataset_items` are not stored row-by-row; the dataset is registered by version and
  content hash, and item content is recoverable from the versioned dataset file.
- `hardware_samples` are stored as JSON inside `hardware_measurements` rather than as
  one row per sample, because there can be hundreds of samples per response.
- `report_exports` is not implemented in Phase 07; report generation is a later phase.
- `experiments.config_hash` is indexed but not unique, so repeated experiments remain
  possible.
