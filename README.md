# Local LLM Benchmarking and Comparative Evaluation

A reproducible local benchmark for comparing Llama 3.1 8B Instruct, Qwen 2.5 7B Instruct, and Mistral 7B Instruct v0.3 through Ollama. The project measures quality, latency, throughput, hardware use, reliability, cold-start behavior, and repeated-response consistency on a consumer laptop, then stores every result in SQLite and exports auditable CSV, JSON, HTML, and PDF reports.

Phase 12 is complete. The final workload was reduced by explicit user choice from the planned 900 measured responses to **435 measured responses** (300 full-quality, 45 cold-start, 90 consistency). All reported values come from stored experiment rows. See [`docs/PHASE12_RESULTS.md`](docs/PHASE12_RESULTS.md) for the full final report.

---

## Contents

1. [Overview](#1-overview)
2. [Key findings](#2-key-findings)
3. [Measured workload](#3-measured-workload)
4. [Features](#4-features)
5. [Architecture](#5-architecture)
6. [Repository layout](#6-repository-layout)
7. [Requirements](#7-requirements)
8. [Environment used](#8-environment-used)
9. [Installation](#9-installation)
10. [Configuration](#10-configuration)
11. [Dataset and probes](#11-dataset-and-probes)
12. [Models and verification](#12-models-and-verification)
13. [Running experiments](#13-running-experiments)
14. [Main benchmark results](#14-main-benchmark-results)
15. [Category analysis](#15-category-analysis)
16. [Cold-start results](#16-cold-start-results)
17. [Consistency results](#17-consistency-results)
18. [Evaluation methodology and judge validation](#18-evaluation-methodology-and-judge-validation)
19. [Reports, dashboard, and artifacts](#19-reports-dashboard-and-artifacts)
20. [Reproducibility, testing, limitations, and next steps](#20-reproducibility-testing-limitations-and-next-steps)

---

## 1. Overview

The benchmark answers a practical question: **which of three locally served 7B-8B instruction models gives the best balance of quality, speed, resource use, and reliability on this machine?**

The application provides:

- A versioned 100-item benchmark dataset with seven task categories and authored evaluation rules.
- A streaming Ollama client with client-side TTFT, latency, and token-throughput measurement.
- Hardware sampling for CPU, RAM, GPU utilization, VRAM, Ollama model VRAM, and GPU temperature.
- Deterministic evaluators plus a blind Qwen 2.5 3B judge.
- A resumable experiment orchestrator backed by SQLite.
- Workload-specific analytics, scoring, recommendations, and historical comparison.
- CSV, JSON, HTML, and PDF reporting with reproducibility manifests.
- A ten-page Streamlit dashboard that reads the same stored data.

The benchmark is an engineering measurement, not a formal randomized controlled trial. Results are specific to the recorded hardware, software, model digests, dataset hash, and configuration.

---

## 2. Key findings

- **Mistral 7B Instruct v0.3 is the best balanced model on this machine.** It ranked first in every configured workload profile, with a general-purpose composite of 77.13/100. Its advantages are latency (P95 37.24 s), throughput (9.76 tok/s), and hardware efficiency.
- **Qwen 2.5 7B Instruct is the strongest quality-first alternative.** It led the discriminating deterministic coding checks (9/15 code tests) and structured-JSON checks (13/15 schemas), and had the highest measured quality score (53.47/100). It ranked second overall with 65.29/100.
- **Llama 3.1 8B Instruct was the slowest and least efficient.** It had the highest mean latency (19.31 s), the highest P95 latency (59.16 s), the lowest throughput (8.00 tok/s), and the highest average RAM use (13,392 MiB). It ranked third with 34.92/100.
- **Quality differences are small and judge-limited.** The top two quality scores differ by 4.0 points, the judge covers only 9% of measured responses, and a manual spot-check found at least one clear judge false negative. Do not treat the quality ranking as definitive.
- **Cold start is expensive.** Loading the model before each request added 10-14 s to TTFT and increased mean latency by 1.6x-2.5x depending on the model.
- **Mistral is faster but less textually consistent at temperature 0.0.** Its repeated-response similarity was 0.780, versus 0.894 for Qwen and 0.944 for Llama. Consistency is not a quality measure.
- **A real timeout defect was found and fixed.** A stalled Qwen request exceeded its 300 s timeout by more than 30 minutes because blank keep-alive lines reset the HTTP read timeout. The stream deadline is now enforced on every line, including blank lines.

---

## 3. Measured workload

| Workload | Experiment ID | Models | Items | Repeats | Warm-up | Measured |
|---|---|---|---:|---:|---:|---:|
| Full quality benchmark | `20261114T074040Z-9edf76` | 3 | 100 | 1 | 1/model | 300 |
| Cold-start subset | `20261114T095758Z-c3ae3a` | 3 | 5 | 3 | 0 | 45 |
| Consistency subset | `20261114T102018Z-e114a1` | 3 | 10 | 3 | 0 | 90 |
| **Total** | | | | | | **435** |

The full run has 303 stored response rows (300 measured plus 3 warm-ups) and 300 evaluation records. The cold-start and consistency subsets have 45 and 90 measured responses respectively, with matching evaluation records.

Not executed: the 9-prompt x 5-repeat performance probe, the planned three full repetitions, a multi-hour soak, and a multi-repetition full-dataset reliability run. These omissions are disclosed in section 20.

---

## 4. Features

- **Dataset validation:** SHA-256 content hash, JSON Schema validation, duplicate-ID rejection, and category/difficulty distributions.
- **Model management:** explicit discovery, verification, pull approval, load/unload, metadata snapshots, and digest checks.
- **Streaming inference:** NDJSON parsing, TTFT, total latency, Ollama token counts, throughput, cancellation, and typed failure records.
- **Hardware monitoring:** baseline and inference windows, CPU/RAM/GPU/VRAM/temperature samples, Ollama model VRAM, and sampler overhead.
- **Hybrid evaluation:** exact match, numeric, code tests, JSON Schema, instruction constraints, reference overlap, refusal checks, and a blind judge.
- **Persistence:** SQLite WAL, foreign keys, immutable completed runs, resumable experiments, and stored errors.
- **Analytics:** normalized dimensions, workload profiles, recommendation gates, confidence, and limitations.
- **Reporting:** six CSV exports, versioned JSON, self-contained HTML, ReportLab PDF, and a manifest with sizes and SHA-256 hashes.
- **Dashboard:** Overview, Model Comparison, Performance Analysis, Quality Evaluation, Hardware Monitoring, Response Inspection, Experiment History, Reports, Recommendations, and Run New Experiment.
- **CLI:** `benchmark` commands for health, models, infer, monitor, analyze, compare, and report; standalone scripts for experiments, dataset validation, reports, preflight, and model pulls.

---

## 5. Architecture

```text
configs/ + data/datasets/
        |
        v
benchmark.core.config  -->  validation, hashing, paths
        |
        v
benchmark.execution    -->  Ollama client, streaming engine, model manager, orchestrator
        |                         |
        |                         +--> benchmark.monitoring (CPU/RAM/GPU/VRAM)
        |                         +--> benchmark.evaluation (deterministic + judge)
        v
benchmark.database     -->  SQLite repositories, schema migrations, immutable runs
        |
        +--> benchmark.analytics  -->  statistics, quality, resources, scoring, recommendations
        +--> benchmark.reporting  -->  CSV, JSON, HTML, PDF, manifest
        +--> benchmark.dashboard  -->  Streamlit read layer and experiment runner
```

Key design rules:

- One candidate model is measured at a time.
- Every response is persisted as soon as it completes.
- Warm-ups are stored but excluded from measured aggregates.
- Missing values are `null`, never zero.
- Completed experiments and runs are immutable through the application.
- Reports and recommendations are computed from stored rows, not hardcoded values.

---

## 6. Repository layout

```text
configs/                     benchmark, model, and evaluation YAML
data/datasets/               canonical dataset, probes, schemas, manifest
data/results/                JSONL inference output (gitignored)
data/exports/                generated report bundles (gitignored)
docs/                        architecture, methodology, schema, testing, results
scripts/                     experiment, report, dataset, preflight, pull helpers
src/benchmark/               application package
tests/                       unit, integration, evaluation, analytics, reporting, dashboard
storage/                     SQLite databases, logs, snapshots (gitignored)
pyproject.toml               package metadata and test markers
```

The installed package is `local-llm-benchmark`; the CLI module is `benchmark`.

---

## 7. Requirements

- Windows 11 (the validated environment) or a compatible OS.
- Python 3.13.x.
- Ollama 0.40.16 or compatible at `http://127.0.0.1:11434`.
- NVIDIA GPU with a working driver for GPU telemetry (CPU/RAM monitoring works without one).
- At least 12 GiB free disk space for the three candidate models and the judge.
- 16 GB system RAM recommended for 7B-8B Q4 models with partial CPU offload.

No model is downloaded automatically. A pull requires explicit `--yes` confirmation.

---

## 8. Environment used

| Item | Value |
|---|---|
| OS | Windows 11 `10.0.26300` |
| CPU | AMD Ryzen 5 5600H, 6 cores / 12 threads |
| RAM | 14,877,257,728 bytes |
| GPU | NVIDIA GeForce GTX 1650, 4,096 MiB |
| GPU driver | 555.99 |
| Python | 3.13.5 |
| Ollama | 0.40.16 |
| Free disk at final health check | 160.4 GiB |
| Dataset | `benchmark-v1.0.0`, SHA-256 `ceafe917235213c710da9e3b6a1ceecb86ba519b214581f683191c51358e46f1` |
| Configuration hash | `34c5793a1e4075824beb5db8f207dd02ca641bd9d1705f49b53ff6f2e1b039ac` |

---

## 9. Installation

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m benchmark init
.\.venv\Scripts\python.exe -m benchmark health
```

Verify the environment and models:

```powershell
.\.venv\Scripts\python.exe scripts\validate_dataset.py
.\.venv\Scripts\python.exe -m benchmark models verify
```

Pull a missing configured model only after explicit approval:

```powershell
.\.venv\Scripts\python.exe -m benchmark models pull qwen2.5:7b-instruct --yes
.\.venv\Scripts\python.exe scripts\pull_models.py --yes
```

The pull command refuses to download without `--yes`. The helper script pulls only configured candidates that are not already installed.

---

## 10. Configuration

- `configs/benchmark.yaml`: dataset path, random seed, request timeout, repetition defaults, cooldowns, generation settings, runtime paths, hardware monitoring, default scoring profile, and five workload profiles.
- `configs/models.yaml`: Ollama endpoint, keep-alive, candidate tags, judge tag, expected digests, and expected quantization.
- `configs/evaluation.yaml`: judge settings, validation fractions, and category judge weights.

Environment overrides:

- `BENCHMARK_PROJECT_ROOT`
- `BENCHMARK_CONFIG_DIR`
- `BENCHMARK_LOG_LEVEL`

`.env.example` documents the supported variables. The application reads process environment variables and does not load `.env` automatically.

Generation settings used for the final run: temperature `0.0`, top-p `1.0`, top-k `40`, repeat penalty `1.1`, `num_ctx` `4096`, `num_predict` `512`, request timeout `300` s, keep-alive `30m`, and 60 s cooldown between models.

---

## 11. Dataset and probes

- Canonical dataset: `data/datasets/benchmark_v1.json` (`benchmark-v1.0.0`, schema `1.0.0`).
- Compatibility export: `data/datasets/benchmark_v1.0.0.jsonl` plus `benchmark_v1.0.0.manifest.json`.
- Performance probe: `data/datasets/probes/performance_v1.0.0.json` (9 prompts, 5 repeats; not executed in Phase 12).
- Consistency probe: `data/datasets/probes/consistency_v1.0.0.json` (10 stratified items, 3 repeats).
- Item JSON Schema: `data/datasets/schemas/benchmark_item.schema.json`.

The canonical dataset has exactly 100 items:

| Category | Items |
|---|---:|
| General knowledge | 15 |
| Logical reasoning | 15 |
| Mathematical reasoning | 15 |
| Coding | 15 |
| Instruction following | 15 |
| Structured JSON | 15 |
| Summarization | 10 |

Difficulty distribution: 30 easy, 47 medium, 23 hard. Six items are unanswerable and use `refusal_check`.

The loader verifies the SHA-256 content hash and every item-defined JSON Schema. See [`docs/DATASET_GUIDE.md`](docs/DATASET_GUIDE.md).

---

## 12. Models and verification

| Role | Tag | Digest | Size | Parameters | Quantization |
|---|---|---|---:|---|---|
| Candidate | `llama3.1:8b` | `46e0c10c039e019119339687c3c1757cc81b9da49709a3b3924863ba87ca666e` | 4,920,753,328 B | 8.0B | Q4_K_M |
| Candidate | `qwen2.5:7b-instruct` | `845dbda0ea48ed749caafd9e6037047aa19acfcfd82e704d7ca97d631a0b697e` | 4,683,087,332 B | 7.6B | Q4_K_M |
| Candidate | `mistral:7b-instruct-v0.3-q4_K_M` | `6577803aa9a036369e481d648a2baebb381ebc6e897f2bb9a766a2aa7bfbc1cf` | 4,372,824,384 B | 7.2B | Q4_K_M |
| Judge | `qwen2.5:3b` | `357c53fb659c5076de1d65ccb0b397446227b71a42be9d1603d46168015c9e4b` | 1,929,912,432 B | 3.1B | Q4_K_M |

`benchmark models verify` reported all four models as verified in the final environment. `benchmark health` passed with all four configured models installed.

The judge shares the Qwen family with the Qwen candidate. It is not treated as neutral, and its scores are reported separately from deterministic scores.

---

## 13. Running experiments

Run a full quality benchmark:

```powershell
.\.venv\Scripts\python.exe scripts\run_experiment.py `
  --database storage\phase12\phase12.db `
  --dataset data\datasets\benchmark_v1.json `
  --models llama3.1:8b qwen2.5:7b-instruct mistral:7b-instruct-v0.3-q4_K_M `
  --profile quality --repeats 1 --warmup 1 --cooldown 60 `
  --name phase12-full-100x1
```

Run a cold-start subset:

```powershell
.\.venv\Scripts\python.exe scripts\run_experiment.py `
  --database storage\phase12\phase12.db `
  --dataset data\datasets\benchmark_v1.json `
  --models llama3.1:8b qwen2.5:7b-instruct mistral:7b-instruct-v0.3-q4_K_M `
  --profile cold_start --cold-start --warmup 0 --repeats 3 --cooldown 60 `
  --items gk-002 lr-001 mr-001 code-001 json-001 `
  --name phase12-coldstart-5x3
```

Run the consistency subset:

```powershell
.\.venv\Scripts\python.exe scripts\run_experiment.py `
  --database storage\phase12\phase12.db `
  --dataset data\datasets\benchmark_v1.json `
  --models llama3.1:8b qwen2.5:7b-instruct mistral:7b-instruct-v0.3-q4_K_M `
  --profile consistency --warmup 0 --repeats 3 --cooldown 60 `
  --items gk-002 gk-009 lr-005 lr-010 mr-006 code-009 if-006 json-011 sum-004 sum-010 `
  --name phase12-consistency-10x3
```

Resume an interrupted experiment:

```powershell
.\.venv\Scripts\python.exe scripts\run_experiment.py --resume EXPERIMENT_ID
```

Analyze a stored experiment:

```powershell
.\.venv\Scripts\python.exe -m benchmark analyze EXPERIMENT_ID --database storage\phase12\phase12.db
.\.venv\Scripts\python.exe -m benchmark analyze EXPERIMENT_ID --database storage\phase12\phase12.db --all-profiles --json
```

Compare two experiments:

```powershell
.\.venv\Scripts\python.exe -m benchmark compare BASELINE_ID CANDIDATE_ID --database storage\phase12\phase12.db --json
```

Smoke inference without persistence:

```powershell
.\.venv\Scripts\python.exe -m benchmark infer --model llama3.1:8b --limit 5 --warmup 1
.\.venv\Scripts\python.exe -m benchmark infer --model llama3.1:8b --item gk-002 --item lr-001 --output data\results\smoke.jsonl
```

---

## 14. Main benchmark results

Experiment `20261114T074040Z-9edf76`; 100 items per model; one measured repetition; one warm-up per model; temperature 0.0. Duration: 2 h 16 min 55 s, including a 30-minute Qwen timeout.

| Model | Success | Mean latency | Median | P95 | Mean TTFT | Output tok/s | CPU avg | RAM avg | System VRAM avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `llama3.1:8b` | 100.0% | 19.31 s | 13.82 s | 59.16 s | 1.40 s | 8.00 | 63.5% | 13,392 MiB | 2,476 MiB |
| `qwen2.5:7b-instruct` | 99.0% | 15.21 s | 8.96 s | 51.95 s | 1.21 s | 9.46 | 54.4% | 11,842 MiB | 2,418 MiB |
| `mistral:7b-instruct-v0.3-q4_K_M` | 100.0% | 12.55 s | 9.00 s | 37.24 s | 1.20 s | 9.76 | 55.8% | 12,307 MiB | 2,445 MiB |

Quality:

| Model | Deterministic | Judge | Quality | Evaluated | Judge coverage |
|---|---:|---:|---:|---:|---:|
| `llama3.1:8b` | 46.58 | 84.03 | 49.79 | 100/100 | 9/100 |
| `qwen2.5:7b-instruct` | 50.02 | 88.19 | **53.47** | 99/100 | 9/99 |
| `mistral:7b-instruct-v0.3-q4_K_M` | 45.75 | **91.67** | 49.44 | 100/100 | 9/100 |

General-purpose recommendation (quality 40%, latency 20%, throughput 15%, efficiency 10%, reliability 15%):

| Rank | Model | Composite | Confidence |
|---:|---|---:|---|
| 1 | `mistral:7b-instruct-v0.3-q4_K_M` | 77.13 | medium |
| 2 | `qwen2.5:7b-instruct` | 65.29 | medium |
| 3 | `llama3.1:8b` | 34.92 | medium |

The Qwen failure was a timeout on `lr-007`. It is excluded from latency and throughput statistics and counted in the 99.0% success rate.

---

## 15. Category analysis

Category composites (0-100). Summarization combines deterministic overlap and the judge; other categories use deterministic evaluators only.

| Category | Llama | Qwen | Mistral | Interpretation |
|---|---:|---:|---:|---|
| General knowledge | 13.3 | 13.3 | 13.3 | Exact match passed 2/15 for all; not discriminative. |
| Logical reasoning | 0.0 | 0.0 | 0.0 | Exact match passed 0/15 for all; not discriminative. |
| Mathematical reasoning | 93.3 | 93.3 | 93.3 | Numeric passed 13/14 for all. |
| Coding | 48.3 | **60.0** | 40.0 | Qwen passed 9/15 code tests; Llama 7/15; Mistral 6/15. |
| Instruction following | **68.3** | 64.4 | 65.6 | All passed 6/14 constraint sets; partial scores differ. |
| Structured JSON | 73.3 | **90.0** | 80.0 | Qwen passed 13/15 schemas; Mistral 11/15; Llama 10/15. |
| Summarization | 51.9 | 53.2 | **53.8** | Small differences; judge-limited. |

The identical general-knowledge, logical-reasoning, and mathematical-reasoning scores are an evaluator limitation, not evidence that the models are equally capable. Qwen is the strongest candidate on the discriminating coding and structured-JSON checks. Llama leads instruction following. Mistral's overall win is driven by speed, throughput, and efficiency.

Workload-specific recommendations:

| Workload | Rank 1 | Rank 2 | Rank 3 |
|---|---|---|---|
| General purpose | Mistral 77.13 | Qwen 65.29 | Llama 34.92 |
| Coding | Mistral 70.85 | Qwen 66.96 | Llama 36.38 |
| Reasoning | Mistral 65.02 | Qwen 59.07 | Llama 37.67 |
| JSON generation | Mistral 87.23 | Qwen 82.21 | Llama 51.17 |
| Resource constrained | Mistral 78.22 | Qwen 69.99 | Llama 24.94 |

Mistral ranks first in every profile because its speed and efficiency advantages outweigh small quality differences. If quality on coding or JSON is the priority, Qwen is the better evidence-backed choice.

---

## 16. Cold-start results

Experiment `20261114T095758Z-c3ae3a`; five items, three repetitions, model unloaded before every measured request, no warm-up. 45 measured responses, all successful. Duration: 22 min 06 s.

| Model | Mean latency | Median | P95 | Mean TTFT | Output tok/s | CPU avg | RAM avg |
|---|---:|---:|---:|---:|---:|---:|---:|
| `llama3.1:8b` | 31.59 s | 28.46 s | 46.35 s | 14.87 s | 8.92 | 40.6% | 10,517 MiB |
| `qwen2.5:7b-instruct` | 23.04 s | 21.80 s | 32.40 s | 14.85 s | 10.30 | 32.1% | 9,990 MiB |
| `mistral:7b-instruct-v0.3-q4_K_M` | 17.61 s | 17.21 s | 24.46 s | 11.34 s | 10.79 | 32.9% | 9,747 MiB |

Cold-start overhead versus the same five items in the warm main run:

| Model | Warm mean | Cold mean | Ratio | TTFT overhead |
|---|---:|---:|---:|---:|
| `llama3.1:8b` | 19.62 s | 31.59 s | 1.61x | +13.55 s |
| `qwen2.5:7b-instruct` | 10.12 s | 23.04 s | 2.28x | +13.52 s |
| `mistral:7b-instruct-v0.3-q4_K_M` | 7.12 s | 17.61 s | 2.47x | +10.24 s |

Cold-start general-purpose ranking: Mistral 73.67, Qwen 71.78, Llama 39.00. The cold-start deterministic scores cover only five items and no judge calls, so they are not comparable to the 100-item quality scores.

---

## 17. Consistency results

Experiment `20261114T102018Z-e114a1`; ten stratified items, three repetitions, warm model, no warm-up. 90 measured responses, all successful. Duration: 26 min 23 s.

| Model | Mean latency | P95 | Mean TTFT | Output tok/s | Response consistency | Quality | Composite |
|---|---:|---:|---:|---:|---:|---:|---:|
| `llama3.1:8b` | 13.21 s | 39.05 s | 0.61 s | 8.50 | **0.944** | 50.3 | 36.12 |
| `qwen2.5:7b-instruct` | 13.14 s | 38.04 s | 0.56 s | 9.07 | 0.894 | 50.2 | 56.27 |
| `mistral:7b-instruct-v0.3-q4_K_M` | 11.69 s | 33.93 s | 0.56 s | 9.31 | 0.780 | 44.2 | 75.56 |

Response consistency is the mean pairwise text similarity across repeated identical prompts. The run used the global temperature of **0.0**, not the 0.7 value in the consistency probe file, because the experiment CLI does not load per-probe generation overrides. It measures determinism and runtime stability at greedy decoding, not robustness under sampled output. Mistral produced the most textual variation; Llama produced the least. Consistency is not a quality score.

---

## 18. Evaluation methodology and judge validation

Deterministic evaluators:

| Evaluation type | Llama | Qwen | Mistral |
|---|---:|---:|---:|
| Code tests | 7/15 | **9/15** | 6/15 |
| Exact match | 0/27 | 0/27 | 0/27 |
| Instruction constraints | 6/14 | 6/14 | 6/14 |
| JSON schema | 10/15 | **13/15** | 11/15 |
| Numeric | 13/14 | 13/14 | 13/14 |
| Reference overlap | 0/9 | 0/9 | 0/9 |
| Refusal check | **5/6** | 4/6 | 4/6 |

The exact-match and reference-overlap evaluators are deliberately strict and penalize correct paraphrases. For example, Llama's `sum-005` response was a correct two-sentence water-cycle summary, but lexical key-fact matching found 0 of 5 facts and the deterministic overlap score was 14.94/100.

Judge validation:

- The judge ran on 27 responses: nine summarization responses per model (9% coverage).
- Judge mean normalized score: 87.96; deterministic overlap mean: 20.91; Pearson correlation: 0.172; mean gap: +67.05 points.
- Manual spot-check of ten judged responses found one clear false negative (Llama `sum-005`, 18.75/100 for a correct summary) and one borderline low completeness score (Qwen `sum-001`). The other eight were broadly plausible; no clear false positive was found.
- The judge is a 3B Qwen model and shares a family with the Qwen candidate. It is not neutral.

The quality ranking (Qwen 53.47, Llama 49.79, Mistral 49.44) is close and judge-limited. The performance ranking is much more stable. See [`docs/EVALUATION_METHODOLOGY.md`](docs/EVALUATION_METHODOLOGY.md) and [`docs/PHASE12_RESULTS.md`](docs/PHASE12_RESULTS.md).

---

## 19. Reports, dashboard, and artifacts

Generate a report bundle:

```powershell
.\.venv\Scripts\python.exe -m benchmark report 20261114T074040Z-9edf76 `
  --database storage\phase12\phase12.db `
  --output-dir data\exports\phase12-main `
  --format csv --format json --format html --format pdf
```

Or use the standalone script:

```powershell
.\.venv\Scripts\python.exe scripts\generate_report.py 20261114T074040Z-9edf76 `
  --database storage\phase12\phase12.db `
  --output-dir data\exports\phase12-main
```

Each bundle contains:

- `experiment_summary.csv`
- `model_comparison.csv`
- `performance_metrics.csv`
- `hardware_metrics.csv`
- `evaluation_results.csv`
- `responses.csv`
- versioned JSON
- self-contained HTML
- ReportLab PDF
- `manifest.json` with byte sizes and SHA-256 hashes

Phase 12 artifacts:

| Artifact | Path |
|---|---|
| Main report bundle | `data/exports/phase12-main/` (PDF 9 pages) |
| Cold-start bundle | `data/exports/phase12-coldstart/` (PDF 9 pages) |
| Consistency bundle | `data/exports/phase12-consistency/` (PDF 9 pages) |
| Experiment database | `storage/phase12/phase12.db` (gitignored) |
| Full results report | `docs/PHASE12_RESULTS.md` |
| Test report | `docs/TESTING_REPORT.md` |

Start the dashboard:

```powershell
.\.venv\Scripts\python.exe -m streamlit run src\benchmark\dashboard\app.py
```

The dashboard reads `storage/benchmark.db` by default and has ten pages: Overview, Model Comparison, Performance Analysis, Quality Evaluation, Hardware Monitoring, Response Inspection, Experiment History, Reports, Recommendations, and Run New Experiment. It computes values from stored responses and evaluations; no benchmark result is hardcoded.

Reports refuse incomplete experiments unless `--allow-incomplete` is set, in which case every format is marked preliminary. Missing measurements are exported as blank cells or JSON `null`, never zero.

---

## 20. Reproducibility, testing, limitations, and next steps

### Testing

Final validation for Phase 12:

| Check | Result |
|---|---|
| `python -m compileall -q src tests scripts` | PASS |
| `python -m pytest -q` | **330 passed, 3 skipped** in 83.12 s |
| `BENCHMARK_LIVE_OLLAMA=1 pytest tests/integration/test_ollama_live.py -q` | **3 passed** in 19.25 s |
| `benchmark health` | PASS; all four configured models installed |
| `benchmark models verify` | PASS; all four models verified |
| `scripts/validate_dataset.py` | VALID; 100 items, probes valid |
| SQLite `integrity_check` / `foreign_key_check` | `ok` / no rows |
| Report generation | 3 bundles, 9 files each plus manifest; PDFs 9 pages |
| Database totals | 3 experiments, 438 responses, 435 evaluations, 0 errors |

The 3 skipped tests are the opt-in live Ollama tests; they pass when enabled.

### Defect fixed in Phase 12

`OllamaClient` now enforces the wall-clock deadline on every stream line, including blank keep-alive lines. The main Qwen run had one timeout on `lr-007` that lasted 1,839 s instead of 300 s because blank lines reset the HTTP read timeout and were skipped before the engine deadline check. The fix is in `src/benchmark/execution/ollama_client.py` with regression test `test_blank_keepalive_lines_respect_total_deadline`. Post-fix subsets produced zero timeouts. The original timeout remains counted as a failure in the main results.

The `analyze` and `compare` CLI commands now accept `--database`, removing the earlier need to copy configuration files or databases.

### Limitations

1. **Reduced workload.** The full benchmark has one repetition per item, not three. Run-to-run variance is not estimated.
2. **Single machine.** Results apply to this Windows laptop, GTX 1650, 16 GB RAM, and Ollama 0.40.16.
3. **Quality uncertainty.** Exact-match and reference-overlap evaluators penalize correct paraphrase. The judge covers 9% of measured responses and produced at least one clear false negative. Quality differences are small and low-confidence.
4. **Judge bias.** The judge is a 3B Qwen model and shares a family with the Qwen candidate. It is not a neutral human panel.
5. **Consistency temperature.** The consistency run used temperature 0.0, not the probe's 0.7. It measures determinism/runtime stability, not sampled-output robustness.
6. **Small subsets.** Cold-start uses 5 items x 3 repeats; consistency uses 10 items x 3 repeats.
7. **No performance probe.** The 9-prompt x 5-repeat performance probe was not executed.
8. **No long soak.** No multi-hour memory/thermal soak was performed.
9. **System-wide hardware.** CPU, RAM, GPU, and VRAM values include other processes and are not exclusively attributable to a model.
10. **Timeout history.** The main Qwen timeout occurred before the keep-alive deadline fix. It is counted as a failure, not hidden.
11. **No lint/type/coverage tooling.** Validation used byte-compilation, the test suite, live Ollama tests, health checks, dataset validation, and database integrity checks.
12. **Sandbox.** Generated code runs in a restricted subprocess, not a security boundary against adversarial code.

### Recommended use

- **Balanced local assistant:** Mistral 7B Instruct v0.3. Best latency, throughput, and efficiency; competitive quality; 100% success in these runs.
- **Quality-first coding or structured JSON:** Qwen 2.5 7B Instruct. Highest measured quality and the strongest deterministic coding/JSON results, at the cost of higher P95 latency than Mistral.
- **Instruction-following and repeated-response stability:** Llama 3.1 8B Instruct. Best instruction-constraint score and highest response consistency, but slowest and most memory-hungry on this machine.

### Next steps

1. Run three full repetitions per model to estimate run-to-run variance.
2. Execute the 9-prompt x 5-repeat performance probe for cleaner latency distributions.
3. Add a second judge or a human review sample and report judge agreement.
4. Run the consistency probe at the intended temperature 0.7 to measure sampled-output robustness.
5. Repeat on a machine with more VRAM to separate GPU-resident performance from CPU offload.

See [`docs/PHASE12_RESULTS.md`](docs/PHASE12_RESULTS.md), [`docs/TESTING_REPORT.md`](docs/TESTING_REPORT.md), [`docs/EXPERIMENTAL_METHODOLOGY.md`](docs/EXPERIMENTAL_METHODOLOGY.md), [`docs/EVALUATION_METHODOLOGY.md`](docs/EVALUATION_METHODOLOGY.md), and [`docs/SCORING_AND_RECOMMENDATIONS.md`](docs/SCORING_AND_RECOMMENDATIONS.md) for the full methodology, evidence, and limitations.
