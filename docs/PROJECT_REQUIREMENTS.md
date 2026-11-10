# Project Requirements — Local LLM Benchmarking and Comparative Evaluation Framework

> **Document purpose:** Define the finalized functional and non-functional requirements, scope, verified environment baseline, constraints, and acceptance criteria for the project. This document is the contract that later phases must satisfy.

| Field | Value |
|---|---|
| Document version | 1.0.0 |
| Status | Phase 00 baseline — approved pending user review |
| Date | 2026-11-10 |
| Project owner | Student developer (Computer Engineering) |
| Development model | Phase-gated; each phase ends with tests, a full project scan, and a completion report |

---

## 1. Executive summary

This project builds a reproducible local benchmarking application that compares three instruction-tuned 7B–8B language models served by Ollama:

1. Meta Llama 3.1 8B Instruct
2. Alibaba Qwen 2.5 7B Instruct
3. Mistral 7B Instruct v0.3

The comparison is performed under identical prompts, generation settings, execution order rules, and measurement procedures on a single Windows 11 laptop with a 4 GB NVIDIA GTX 1650. The application records quality, latency, throughput, hardware consumption, and reliability in a local SQLite database, then produces dashboards, workload-specific recommendations, and exportable reports.

The central question is:

> Among Llama 3.1 8B, Qwen 2.5 7B, and Mistral 7B, which model provides the best balance of response quality, inference performance, hardware efficiency, and reliability under my local hardware constraints?

The system must answer this with measured evidence, not opinion. It must also expose missing or unreliable measurements instead of hiding them.

---

## 2. Objectives

### 2.1 Primary objectives

- Compare the three candidate models on a fixed, versioned 100-question benchmark.
- Measure response quality with deterministic evaluators wherever possible.
- Use an LLM-as-a-judge only for subjective dimensions that cannot be checked deterministically.
- Measure client-observed latency and Ollama-reported runtime metadata separately.
- Measure system-wide and Ollama-process resource consumption during inference.
- Preserve every experiment result, including failures, timeouts, and outliers.
- Provide workload-specific recommendations with configurable weights and explicit confidence.
- Export CSV, JSON, HTML, and PDF reports containing methodology, findings, and limitations.

### 2.2 Learning objectives

- Practice reproducible experiment design under real hardware constraints.
- Understand the difference between client-side and runtime-side metrics.
- Learn to build deterministic evaluators and to use LLM judges responsibly.
- Learn schema design, append-only experiment storage, and evidence-based reporting.
- Learn to test and scan an application after every development phase.

### 2.3 Success criteria

The project succeeds when a user can:

1. Open the dashboard.
2. Select candidate models.
3. Select a versioned benchmark dataset.
4. Configure generation and experiment settings.
5. Run the experiment sequentially with visible progress.
6. Inspect raw responses and evaluation details.
7. Compare quality, latency, throughput, hardware use, and reliability.
8. Review historical experiments without overwriting them.
9. Receive workload-specific recommendations with limitations.
10. Export reports that another person could use to reproduce the experiment.

---

## 3. Scope

### 3.1 In scope

- Local Ollama runtime integration through its HTTP API.
- Model discovery, availability checks, explicit user-controlled pulls, metadata inspection, load/unload.
- A 100-item benchmark dataset with seven categories and embedded unanswerable questions.
- Streaming inference with client-side TTFT and total latency measurement.
- Warm-start and cold-start experiment modes.
- CPU, RAM, GPU, VRAM, and temperature sampling during inference.
- Deterministic evaluation: exact match, numeric validation, code tests, JSON Schema validation, instruction constraints, reference overlap, refusal checks.
- LLM-as-a-judge for relevance, completeness, coherence, and reasoning quality.
- SQLite storage through SQLAlchemy with append-only historical results.
- Analytics, normalized weighted scoring, and workload profiles.
- Streamlit dashboard with ten required views.
- CSV, JSON, HTML, and PDF exports.
- Pytest-based unit and integration tests.
- Documentation of methodology, metrics, architecture, and roadmap.

### 3.2 Out of scope

- Training, fine-tuning, quantizing, or converting models.
- Distributed or multi-machine benchmarking.
- Serving models to external users or exposing a public API.
- Agent frameworks, tool-use loops, RAG pipelines, or LangChain/LangGraph.
- Docker, Kubernetes, FastAPI, or cloud deployment.
- Mobile or cross-platform packaging.
- True multi-tenant security isolation.
- Formal statistical significance claims from very small samples.
- Energy or electricity-cost measurement, because reliable laptop power telemetry is not guaranteed.
- Judging factual truth solely with an LLM.

### 3.3 Explicit non-goals

- Producing a universal "best model" ranking independent of workload and hardware.
- Comparing models at different quantization levels without recording and disclosing the difference.
- Hiding failed requests, timeouts, or unavailable sensors.
- Using a judge score to override a deterministic failure.

---

## 4. Verified environment baseline

The following values were measured on 2026-11-10 during Phase 00. The application must detect these values at runtime and store them with each experiment; they must not be hardcoded as fixed assumptions.

| Component | Verified value | Notes |
|---|---|---|
| Laptop | Lenovo IdeaPad Gaming 3 (82K2) | User-provided model identification |
| CPU | AMD Ryzen 5 5600H, 6 cores / 12 logical processors | Detected via Win32_Processor |
| System RAM | 16 GB nominal; 14,877,257,728 bytes visible (~13.85 GiB) | `psutil` will report total and available at run time |
| GPU | NVIDIA GeForce GTX 1650, 4,096 MiB VRAM | Detected via nvidia-smi |
| GPU driver | 555.99 | Windows WDDM driver model |
| CUDA runtime reported by driver | 12.5 | Driver capability, not a project dependency |
| GPU compute capability | 7.5 | Turing generation |
| OS | Windows 11 Home, build 26300 | `platform.platform()` reported Windows-11-10.0.26300 |
| Python | 3.13.5 (Anaconda distribution) | `C:\Users\LENOVO\Anaconda3\python.exe` |
| Ollama | 0.40.15 | Verified with `ollama --version` |
| Ollama API | `http://127.0.0.1:11434` | Local only |
| Free disk on C: | ~171.7 GB at Phase 00 inspection | Must be checked before pulling models |

### 4.1 Hardware constraints that shape the design

- **VRAM is the binding constraint.** A 7B–8B model at Q4_K_M is roughly 4.4–4.9 GB on disk. The GPU has 4,096 MiB total, part of which is used by Windows desktop composition. Full GPU residency is not guaranteed.
- **Partial CPU offload is expected.** The application must record Ollama's layer placement (`size` and `size_vram` from `/api/ps`) and must not assume that all layers are on the GPU.
- **Models run sequentially.** No two candidate models may be loaded for measurement at the same time.
- **System-wide VRAM is not model VRAM.** NVML reports total GPU memory in use by all processes, including the desktop. Ollama's own `/api/ps` accounting is reported separately.
- **Windows WDDM limits telemetry.** Per-process GPU memory may be unavailable or approximate; GPU utilization sampling may be coarse. Unavailable values are stored as `null` and labeled, never invented.
- **Thermals and background load vary.** Final experiments require repeated runs, recorded order, and a cooldown policy.

---

## 5. Candidate and judge models

### 5.1 Candidate model tags

The canonical tags for this project are:

| Candidate | Canonical Ollama tag | Local status at Phase 00 | Registry observation |
|---|---|---|---|
| Llama 3.1 8B Instruct | `llama3.1:8b` | Installed | Manifest `46e0c10c039e019119339687c3c1757cc81b9da49709a3b3924863ba87ca666e`, 4.9 GB, 131,072 context, 8.0B, Q4_K_M |
| Qwen 2.5 7B Instruct | `qwen2.5:7b-instruct` | Not installed | Manifest `845dbda0ea48ed749caafd9e6037047aa19acfcfd82e704d7ca97d631a0b697e`, 4.7 GB, 32,768 context; registry config reports 7.6B, Q4_K_M |
| Mistral 7B Instruct v0.3 | `mistral:7b-instruct-v0.3-q4_K_M` | Not installed | Manifest `6577803aa9a036369e481d648a2baebb381ebc6e897f2bb9a766a2aa7bfbc1cf`, 4.4 GB, 32,768 context; registry config reports 7.2B, Q4_K_M |

Verified local metadata for the installed Llama model:

| Field | Value |
|---|---|
| Tag | `llama3.1:8b` |
| Manifest digest | `46e0c10c039e019119339687c3c1757cc81b9da49709a3b3924863ba87ca666e` |
| Size | 4,920,753,328 bytes |
| Architecture | `llama` |
| Parameters | 8,030,261,312 (8.0B) |
| Quantization | Q4_K_M |
| Context length | 131,072 tokens |
| Embedding length | 4,096 |
| Capabilities | completion, tools |

The Qwen 7B and Mistral v0.3 models are not yet installed. Their registry manifests were resolved during Phase 00, and their registry config blobs report Q4_K_M, but **local quantization and metadata must be verified with `/api/show` after the explicit pull in Phase 02**. The application must never assume that all three tags use identical quantization.

The Ollama library page displays the shorthand `mistral:7b-instruct-v0.3`, but that exact tag returned 404 from the registry manifest endpoint during Phase 00. The explicit tag `mistral:7b-instruct-v0.3-q4_K_M` resolved successfully and matches the library page's manifest digest prefix `6577803aa9a0`. The project therefore uses the explicit tag for reproducibility and records any alias behavior observed during the actual pull.

Registry model-layer sizes observed during Phase 00: Qwen 4,683,073,952 bytes; Mistral 4,372,811,712 bytes; Llama 4,920,738,944 bytes. The local Llama `/api/tags` total is 4,920,753,328 bytes, which includes non-model layers; these values are for verification, not for cross-model size comparisons.

### 5.2 Judge model

- Preferred judge: `qwen2.5:3b`, already installed.
- Verified metadata: digest `357c53fb659c5076de1d65ccb0b397446227b71a42be9d1603d46168015c9e4b`, 1.93 GB, Q4_K_M, 3.1B parameters, 32,768 context.
- The judge is not one of the candidate models.
- Qwen 2.5 3B is the same model family as the Qwen 2.5 7B candidate. This creates a possible self-preference bias. The project must disclose it and mitigate it with blinding, a fixed rubric, a manual validation sample, and an optional secondary cross-check with `llama3.2:3b` (installed) on a sample. The cross-check judge shares the Llama family with the Llama candidate, so it is not treated as neutral either; it is only a secondary sample check.

### 5.3 Metadata collection requirements

For every model used in an experiment, record:

- Exact Ollama tag.
- Manifest digest from `/api/tags` (full SHA-256).
- Model file size in bytes.
- Architecture family.
- Parameter count and human-readable parameter size.
- Quantization level.
- Context length and embedding length.
- Ollama runtime version.
- Whether the model was already present or pulled during the experiment.
- Model capabilities reported by Ollama.
- Layer placement and VRAM accounting from `/api/ps` after load, after warm-up, and at the end of the run.

Metadata inspection methods:

- `GET /api/tags` for installed models, digest, size, and summary details.
- `POST /api/show` with `{"model": "<tag>"}` for details and `model_info`.
- `POST /api/show` with `{"model": "<tag>", "verbose": true}` for tensor-level information.
- CLI equivalents: `ollama list`, `ollama show <tag>`, `ollama show --verbose <tag>`.
- `GET /api/ps` for currently loaded models, `size`, `size_vram`, context, and expiry.
- `GET /api/version` for the runtime version.

Registry pages display truncated digest prefixes. The full manifest digest must be recorded from `/api/tags` at pull time and stored with the model record; a registry page prefix is not sufficient for verification.

---

## 6. Functional requirements

Requirement identifiers are stable and will be referenced by tests and later phase reports.

### 6.1 A — Model management

| ID | Requirement |
|---|---|
| A-01 | Discover installed Ollama models via the API. |
| A-02 | Verify that each selected candidate and the judge model are available. |
| A-03 | Pull missing models only after explicit user confirmation. |
| A-04 | Display pull progress and record success or failure. |
| A-05 | Inspect and store model metadata listed in section 5.3. |
| A-06 | Load a model with a defined `keep_alive` value. |
| A-07 | Unload a model and confirm it is absent from `/api/ps` before cold-start measurement. |
| A-08 | Record the Ollama version and warn if it changes between runs. |
| A-09 | Handle connection failures, missing models, and malformed metadata without crashing the experiment. |
| A-10 | Never silently substitute a different tag or quantization. |
| A-11 | Record `/api/ps` layer placement after load, after warm-up, and at the end of a model run; flag placement changes. |

### 6.2 B — Benchmark dataset

| ID | Requirement |
|---|---|
| B-01 | Provide a versioned dataset of exactly 100 items. |
| B-02 | Use the category counts: general knowledge 15, logical reasoning 15, mathematical reasoning 15, coding 15, instruction following 15, structured JSON 15, summarization 10. |
| B-03 | Assign each item a difficulty level, expected answer or reference, evaluation method, and category metadata. |
| B-04 | Embed unanswerable items inside existing categories without increasing the total count. |
| B-05 | Validate the dataset against a schema before any run. |
| B-06 | Compute and store a SHA-256 content hash for the dataset version. |
| B-07 | Keep the dataset version fixed for the duration of an experiment. |
| B-08 | Require a new version for any content change. |
| B-09 | Store prompts, references, evaluation configuration, and unanswerable flags in the database. |
| B-10 | Avoid copying copyrighted benchmark content; author items for this project. |

### 6.3 C — Inference engine

| ID | Requirement |
|---|---|
| C-01 | Send identical prompts to all candidate models. |
| C-02 | Apply identical generation options to all candidates. |
| C-03 | Use streaming responses. |
| C-04 | Measure client-side TTFT from request start to first non-empty content token. |
| C-05 | Measure total client-side latency to the final stream chunk. |
| C-06 | Capture prompt token count and output token count from Ollama. |
| C-07 | Calculate prompt processing speed from Ollama metadata. |
| C-08 | Calculate output tokens per second from Ollama metadata. |
| C-09 | Record model load duration when reported. |
| C-10 | Record all Ollama runtime timing fields from the final response. |
| C-11 | Enforce request timeouts and record them distinctly from other failures. |
| C-12 | Record interrupted and cancelled requests without corrupting the run. |
| C-13 | Persist the raw response and raw final metadata for audit. |
| C-14 | Never retry a failed candidate request by default; retries would bias reliability metrics. |
| C-15 | Execute candidate models sequentially. |
| C-16 | Prefix repeated performance prompts with a unique nonce to prevent prompt-prefix cache reuse; record `prompt_eval_count` and flag anomalies. |
| C-17 | Run the performance probe before the quality and consistency profiles to reduce thermal drift. |
| C-18 | Record the finish/done reason and flag output-length truncation. |

### 6.4 D — Hardware monitoring

| ID | Requirement |
|---|---|
| D-01 | Sample CPU utilization system-wide. |
| D-02 | Sample system RAM used and available. |
| D-03 | Sample Ollama process-tree RSS where available, labeled approximate. |
| D-04 | Sample GPU utilization where available. |
| D-05 | Sample GPU memory utilization where available. |
| D-06 | Sample total system VRAM used and total VRAM. |
| D-07 | Sample GPU temperature where available. |
| D-08 | Record Ollama-reported per-model `size_vram` separately from NVML totals. |
| D-09 | Compute peak and average values over the inference window. |
| D-10 | Record baseline samples before each request. |
| D-11 | Store unavailable measurements as `null` with an availability flag and note. |
| D-12 | Record sampling interval and sample count. |
| D-13 | Account for Windows WDDM limitations in labels and documentation. |

### 6.5 E — Evaluation engine

| ID | Requirement |
|---|---|
| E-01 | Implement exact-match evaluation with documented normalization. |
| E-02 | Implement numeric answer validation with tolerance. |
| E-03 | Implement code execution with test cases in a restricted subprocess. |
| E-04 | Implement JSON parsing and schema validation. |
| E-05 | Implement instruction-following constraint checks. |
| E-06 | Implement reference-based overlap scoring. |
| E-07 | Implement refusal and uncertainty checks for unanswerable items. |
| E-08 | Use deterministic evaluation as the primary score for objective categories. |
| E-09 | Use an LLM judge only for relevance, completeness, coherence, and reasoning quality. |
| E-10 | Blind the judge to model identity and randomize response order. |
| E-11 | Validate a sample of judge scores manually and record agreement. |
| E-12 | Document judge bias risks and limitations. |
| E-13 | Never allow the judge to override a deterministic failure. |
| E-14 | Record evaluator name, version, score, pass/fail, and details for every evaluation. |

### 6.6 F — Database

| ID | Requirement |
|---|---|
| F-01 | Use SQLite with SQLAlchemy. |
| F-02 | Store model metadata, dataset versions, configurations, runs, model runs, responses, hardware samples, hardware summaries, evaluations, errors, and recommendations. |
| F-03 | Preserve historical results; never overwrite a completed run. |
| F-04 | Use foreign keys and indexes for run and response lookups. |
| F-05 | Enable SQLite foreign-key enforcement and WAL mode. |
| F-06 | Store raw Ollama metadata as JSON for audit. |
| F-07 | Support resuming an interrupted run without duplicating completed responses. |
| F-08 | Record schema version and support additive migrations. |

### 6.7 G — Analytics

| ID | Requirement |
|---|---|
| G-01 | Calculate mean, median, and P95 latency for warm-start performance measurements. |
| G-02 | Calculate mean TTFT. |
| G-03 | Calculate output tokens per second and prompt processing speed. |
| G-04 | Calculate quality scores by category and overall. |
| G-05 | Calculate success, failure, timeout, and cancellation rates. |
| G-06 | Calculate response consistency from repeated prompts. |
| G-07 | Calculate resource efficiency indicators. |
| G-08 | Support configurable weighted scoring. |
| G-09 | Produce workload-specific recommendations. |
| G-10 | Report sample sizes and missing-data caveats with every aggregate. |
| G-11 | Report quality coverage and a failure-as-zero quality sensitivity score. |

### 6.8 H — Dashboard

| ID | Requirement |
|---|---|
| H-01 | Overview page. |
| H-02 | Model comparison page. |
| H-03 | Performance analysis page. |
| H-04 | Quality analysis page. |
| H-05 | Hardware monitoring page. |
| H-06 | Individual response inspection page. |
| H-07 | Experiment history page. |
| H-08 | Recommendations page. |
| H-09 | New experiment execution page. |
| H-10 | Report export page. |
| H-11 | Clearly label unavailable metrics. |
| H-12 | Never block the UI while a long experiment runs; show progress from the database or logs. |

### 6.9 I — Reporting

| ID | Requirement |
|---|---|
| I-01 | Export CSV files for responses, metrics, and evaluations. |
| I-02 | Export JSON with the full experiment structure. |
| I-03 | Generate HTML reports with methodology, hardware, results, findings, recommendations, and limitations. |
| I-04 | Generate PDF reports with the same content. |
| I-05 | Include model tags, digests, quantization, Ollama version, dataset hash, generation settings, and host metadata. |
| I-06 | Include failed and missing measurements in reports. |
| I-07 | Avoid claiming statistical significance that the sample cannot support. |

---

## 7. Non-functional requirements

| ID | Requirement |
|---|---|
| N-01 | Python 3.13 compatibility. |
| N-02 | Direct Ollama HTTP API; no LangChain, LangGraph, FastAPI, Docker, or Kubernetes. |
| N-03 | Every Python file begins with the required module docstring header. |
| N-04 | Code is simple, readable, typed where useful, and free of placeholders. |
| N-05 | Configuration is YAML-based and validated with Pydantic. |
| N-06 | Dependencies are pinned and documented; no unnecessary packages. |
| N-07 | The application runs locally without network access except for explicit model pulls. |
| N-08 | Experiments are reproducible from the stored configuration, dataset hash, model digests, and host metadata. |
| N-09 | The application remains usable when GPU telemetry is unavailable. |
| N-10 | Logs are structured, rotate on size, and never contain secrets. |
| N-11 | The full test suite and project scan pass after every phase. |
| N-12 | Documentation is updated whenever interfaces or methodology change. |

---

## 8. Experimental validity requirements

These requirements are mandatory and are detailed further in `EXPERIMENTAL_METHODOLOGY.md`.

- Identical prompts and generation options for all candidates.
- Exact model tags, digests, quantization, and runtime version recorded.
- Fixed dataset version per experiment.
- Cold-start and warm-start experiments stored separately.
- Sequential model execution only.
- Repeated measurements for latency and throughput.
- Failures, timeouts, and outliers retained and reported.
- Client TTFT and Ollama runtime timings reported separately.
- System-wide VRAM and per-model VRAM reported separately.
- Warm-up requests discarded from measurement.
- Model order and item order recorded.
- Random seeds recorded where applicable.
- Unavailable metrics labeled, never fabricated.
- No results published before the actual runs are completed.

---

## 9. Security requirements

| ID | Requirement |
|---|---|
| S-01 | Never execute model-generated code directly in the host Python process. |
| S-02 | Run coding tests in a subprocess with a temporary working directory, a timeout, output limits, and a restricted environment. |
| S-03 | Treat the subprocess as best-effort isolation, not a security boundary; document the limitation. |
| S-04 | Do not use `shell=True`. |
| S-05 | Validate all dataset, configuration, and model-response inputs. |
| S-06 | Do not let model-generated JSON control privileged operations. |
| S-07 | Do not log secrets; the project uses no remote API keys. |
| S-08 | Limit response and error sizes stored in the database and displayed in the UI. |
| S-09 | For untrusted use, run the application in a dedicated low-privilege Windows account or a virtual machine. |

---

## 10. Data, privacy, and licensing

- All benchmark items are authored for this project or use permissively licensed material with attribution.
- No personal or confidential data is sent to external services. Ollama runs locally.
- Model licenses differ and are recorded from each artifact's `/api/show` metadata rather than assumed. The installed Llama 3.1 artifact reports the Llama 3.1 Community License, and the installed Qwen 2.5 3B judge reports the Qwen Research License. The licenses of the two pulled candidates must be recorded when their metadata is verified. The project does not redistribute model weights.
- Generated responses are stored locally and may be exported; the user is responsible for reviewing exports before sharing.
- The database and reports contain no secrets.

---

## 11. Deliverables

| Deliverable | Phase |
|---|---|
| Requirements and methodology documentation | Phase 00 |
| Project skeleton, configuration, dependency environment | Phase 01 |
| Model metadata collection and pull workflow | Phase 02 |
| Versioned 100-item dataset | Phase 03 |
| Streaming inference engine | Phase 04 |
| Hardware monitoring engine | Phase 05 |
| Deterministic evaluation and judge | Phase 06 |
| Database and experiment management | Phase 07 |
| Analytics and recommendations | Phase 08 |
| Streamlit dashboard | Phase 09 |
| CSV/JSON/HTML/PDF reporting | Phase 10 |
| Test suite, integration validation, reliability checks | Phase 11 |
| Final experiments and documentation | Phase 12 |

---

## 12. Acceptance criteria for the whole project

The project is accepted only when all of the following are true:

1. All three candidate models can be discovered, verified, and benchmarked sequentially.
2. A complete 100-item run can be executed, interrupted, resumed, and audited.
3. Warm-start and cold-start results are stored separately.
4. Every successful response has raw text, timing metadata, hardware samples, and evaluation records; every failed request has a failure record.
5. Deterministic evaluators are tested with known-good and known-bad fixtures.
6. Judge scores are stored with the judge model tag and digest.
7. Historical runs are immutable through the application.
8. Dashboard pages load from the database and show real experiment data.
9. Reports include methodology, hardware, metrics, findings, recommendations, and limitations.
10. The full test suite passes and the full project scan reports no known integration breakage.
11. The documentation states what was measured, what was not measured, and why.
12. No fabricated benchmark results appear anywhere in the repository.

---

## 13. Assumptions

- Ollama remains installed and reachable at `127.0.0.1:11434`.
- The user can pull the two missing models when disk space and network permit.
- The GTX 1650 remains available; the application still runs on CPU if it is not.
- The laptop stays on AC power during experiments.
- No other heavy GPU or CPU workload runs during measurement.
- The user reviews judge outputs and report conclusions before publishing them.

---

## 14. Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Models do not fit in 4 GB VRAM | Slow inference, CPU offload, thermal throttling | Record layer placement; separate cold/warm; allow cooldown; report hardware limits |
| Windows WDDM hides per-process GPU data | Incomplete hardware attribution | Use NVML total plus Ollama `size_vram`; store nulls with labels |
| Background desktop GPU use | Noisy VRAM baseline | Record baseline; report deltas; repeat measurements |
| Thermal throttling | Drift in latency | Cooldown policy; randomized/balanced order; repeated runs |
| Judge self-preference (Qwen judging Qwen) | Biased subjective scores | Blind judge, fixed rubric, manual validation, optional second judge |
| Code execution escapes subprocess controls | Host compromise | Best-effort sandbox, timeouts, low-privilege account/VM for untrusted use, document limitation |
| Dataset contamination in training data | Inflated quality | Author original items; disclose contamination risk |
| Small sample size | Unstable P95 and rankings | Report n, use repeated runs, bootstrap intervals, avoid significance claims |
| Model tag changes upstream | Non-reproducible results | Record digest; verify digest before each run; require explicit override |
| Pulling models consumes disk/time | Interrupted setup | Check disk space; explicit confirmation; progress display |

---

## 15. Unresolved questions for later phases

1. Confirm local digests, quantization, parameter count, and context length of `qwen2.5:7b-instruct` and `mistral:7b-instruct-v0.3-q4_K_M` after pulling them. Phase 02 implemented the verification and snapshot workflow, but the pull remains pending explicit user approval. Registry config blobs already report Q4_K_M for both, but local verification is still required.
2. Decide whether the optional secondary judge (`llama3.2:3b`) is used for a cross-check sample or only manual validation is performed.
3. Confirm the final number of full repeated runs (proposed: three full runs for final experiments, one for development).
4. Confirm whether `/api/chat` is used for all candidate prompts (proposed) or `/api/generate` for selected tasks.
5. Decide whether code evaluation runs under the normal user account with best-effort isolation or under a dedicated low-privilege account/VM for final experiments.
6. Confirm that GPU power draw and CPU temperature are treated as unavailable unless a reliable sensor is detected.
7. Confirm in Phase 01 that `jsonschema` supports Python 3.13 and pin it. The architecture uses `jsonschema` for arbitrary item-defined JSON Schemas; Pydantic handles configuration and dataset-item models.
8. Confirm the exact pinned dependency versions in Phase 01.
9. Confirm the dataset authoring sources and the final distribution of unanswerable items.
10. Confirm the report template design and whether PDF fonts need bundling.

---

## 16. Glossary

| Term | Meaning |
|---|---|
| Candidate model | A model being evaluated: Llama 3.1 8B, Qwen 2.5 7B, or Mistral 7B v0.3 |
| Judge model | A separate model used only for subjective scoring |
| TTFT | Time to first token, measured client-side |
| Warm start | Model already loaded and warmed before measurement |
| Cold start | Model unloaded before the request; includes load time |
| Manifest digest | Ollama's SHA-256 identifier for a model manifest |
| Quantization | Compression format such as Q4_K_M |
| VRAM | GPU memory |
| WDDM | Windows Display Driver Model; affects GPU telemetry |
| Deterministic evaluator | A rule-based check with a reproducible pass/fail or score |
| Workload profile | A weighted scoring preset for a use case |
| Experiment run | One execution of a configuration across selected models |
| Model run | The portion of an experiment run for one model |
| Response record | One prompt-response pair with timings and metadata |
| Performance probe | A fixed prompt subset used for repeated latency measurements |
| Consistency probe | A fixed subset repeated at nonzero temperature to measure similarity |

---

## 17. Change control

- This document is versioned. Material changes require a new version and a note in the phase report.
- The dataset, generation settings, judge rubric, and scoring weights are experimental variables. Changing any of them requires a new experiment configuration and, where applicable, a new dataset version.
- Historical experiment records are never edited to match new methodology.
