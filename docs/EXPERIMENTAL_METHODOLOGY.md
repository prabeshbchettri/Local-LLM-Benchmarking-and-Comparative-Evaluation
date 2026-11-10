# Experimental Methodology — Local LLM Benchmarking and Comparative Evaluation Framework

> **Document purpose:** Define the experimental design, controls, dataset structure, generation settings, execution protocol, judge methodology, and validity safeguards used to compare the three candidate models. This document explains how measurements are produced so that results can be reproduced and criticized.

| Field | Value |
|---|---|
| Document version | 1.2.0 |
| Status | Phase 03 dataset and Phase 04 inference engine implemented (pending approval) |
| Date | 2026-11-10 |
| Related documents | `PROJECT_REQUIREMENTS.md`, `METRICS_DEFINITIONS.md`, `ARCHITECTURE.md`, `DATASET_GUIDE.md` |

---

## 1. Research question

> Among Llama 3.1 8B Instruct, Qwen 2.5 7B Instruct, and Mistral 7B Instruct v0.3, which model provides the best balance of response quality, inference performance, hardware efficiency, and reliability on a Lenovo IdeaPad Gaming 3 with an NVIDIA GTX 1650 4 GB, 16 GB RAM, Windows 11, and Ollama 0.40.15?

"Best balance" is workload-dependent. The project therefore reports a default balanced score plus workload-specific profiles, and it reports the underlying dimensions separately so a reader can disagree with the weights.

### 1.1 Pre-registered expectations

These are hypotheses to be tested, not results. They may be falsified.

| ID | Expectation | Rationale |
|---|---|---|
| H1 | Llama 3.1 8B will score highest on general knowledge and instruction following. | Larger parameter count and strong instruction tuning. |
| H2 | Mistral 7B v0.3 will have the lowest latency and smallest memory footprint. | Smallest artifact of the three (about 4.4 GB) and a simpler 7B architecture. |
| H3 | Qwen 2.5 7B will be strongest on structured JSON and coding tasks. | Qwen 2.5 is marketed and commonly observed as strong on structured and code tasks; this must be verified locally. |
| H4 | All three models will show partial CPU offload on the 4 GB GPU, and throughput will be CPU-bound rather than GPU-bound. | Model artifacts exceed usable VRAM. |

No hypothesis will be reported as confirmed until the measured runs are complete. The project will report effect sizes and uncertainty, not only ranks.

---

## 2. Experimental design summary

The design is a controlled, repeated, within-subject comparison:

- **Subjects:** three candidate models.
- **Conditions:** identical dataset items, prompts, generation options, timeouts, and measurement procedures.
- **Execution:** strictly sequential; one model loaded at a time.
- **Measurement:** client-side timing, Ollama runtime metadata, system/GPU sampling, deterministic evaluation, and blinded judge scoring.
- **Replication:** repeated performance measurements, optional consistency measurements, and three independent full runs for final conclusions.
- **Storage:** every response and failure is persisted; historical results are immutable.

The experiment is not a formal randomized controlled trial. It is an engineering benchmark with documented controls and limitations.

---

## 3. Variables

### 3.1 Independent variables

- Candidate model: Llama 3.1 8B, Qwen 2.5 7B, Mistral 7B v0.3.
- Workload category: seven benchmark categories.
- Prompt difficulty: easy, medium, hard.
- Run type: warm-start, cold-start, consistency.
- Optional controlled factors: none in the primary analysis; all candidates receive identical settings.

### 3.2 Dependent variables

- Quality: deterministic score, judge score, composite score.
- Latency: TTFT, total latency, mean/median/P95.
- Throughput: output tokens per second, prompt processing speed.
- Resource use: CPU, system RAM, Ollama RSS, GPU utilization, VRAM, GPU temperature.
- Reliability: success, failure, timeout, cancellation, consistency.
- Recommendation score and rank by workload profile.

### 3.3 Controlled variables

- Dataset version and item order.
- Prompt text and output-format instructions.
- Generation options: temperature, top-p, top-k, seed, repeat penalty, context length, maximum output tokens.
- Request timeout and keep-alive policy.
- Sampling interval and hardware monitoring method.
- Model execution order and cooldown policy.
- Ollama version and host environment.
- Judge model, rubric, temperature, and blinding.
- Evaluation code version.

### 3.4 Nuisance variables

- Ambient temperature and laptop thermal state.
- Background Windows processes, browser GPU use, antivirus scans, and Windows updates.
- Disk cache state and model load order.
- GPU clock variability and WDDM scheduling.
- Ollama scheduler behavior and layer offload decisions.
- Network stack overhead, even on localhost.
- Judge stochasticity and bias.

Nuisance variables are recorded where possible, controlled by protocol where practical, and disclosed as limitations otherwise.

---

## 4. Models and verification

### 4.1 Candidate tags

| Candidate | Canonical tag | Notes |
|---|---|---|
| Llama 3.1 8B Instruct | `llama3.1:8b` | Installed; Q4_K_M; 8.03B parameters; 131,072 context |
| Qwen 2.5 7B Instruct | `qwen2.5:7b-instruct` | Must be pulled; registry config reports 7.6B, Q4_K_M; verify local metadata after pull |
| Mistral 7B Instruct v0.3 | `mistral:7b-instruct-v0.3-q4_K_M` | Must be pulled; registry config reports 7.2B, Q4_K_M; verify local metadata after pull |

The shorthand `mistral:7b-instruct-v0.3` returned 404 from the registry manifest endpoint during Phase 00, while `mistral:7b-instruct-v0.3-q4_K_M` resolved and matched the library page digest prefix. The explicit tag is used for reproducibility.

### 4.2 Judge tag

| Role | Tag | Notes |
|---|---|---|
| Primary judge | `qwen2.5:3b` | Installed; Q4_K_M; 3.1B parameters; 32,768 context |
| Optional cross-check judge | `llama3.2:3b` | Installed; used only on a sample if approved; never a candidate |

The primary judge shares the Qwen family with one candidate, and the optional cross-check judge shares the Llama family with another. Neither is treated as neutral. The primary judge scores all eligible responses; the optional cross-check is used only on a sample, and its agreement with the primary is reported. Manual validation remains the final check.

### 4.3 Verification procedure before every experiment

1. Query `GET /api/version` and record the Ollama version.
2. Query `GET /api/tags` and record the full manifest digest, size, family, parameter size, and quantization for each required model.
3. Query `POST /api/show` for each model and record architecture, parameter count, context length, embedding length, quantization, and capabilities.
4. Compare the digest to the digest stored in the experiment configuration. If it changed, stop and require explicit user approval to use the new artifact.
5. Verify the judge model independently.
6. Verify free disk space before any pull.
7. Never substitute a different tag, family, or quantization silently.
8. After loading each candidate, record `/api/ps` layer placement; record it again after warm-up and at the end of the model run, and flag any change in `size_vram`.

### 4.4 Quantization policy

- The project compares the models as distributed by their canonical Ollama tags.
- Quantization is recorded, not assumed. If the three artifacts use different quantization levels, the comparison is a comparison of those specific artifacts, not of the underlying models in the abstract.
- Changing a tag to force equal quantization would be a separate experiment and would require a new configuration and disclosure.

---

## 5. Benchmark dataset design

### 5.1 Version and size

- Dataset name: `benchmark-v1.0.0`.
- Total items: exactly 100.
- Canonical format: `data/datasets/benchmark_v1.json`, a versioned container with dataset metadata, `content_sha256`, and an `items` array.
- Compatibility export: `data/datasets/benchmark_v1.0.0.jsonl` with one validated item per line, plus `benchmark_v1.0.0.manifest.json` containing dataset metadata and the SHA-256 content hash.
- The JSON file is the source of truth; the JSONL export is generated deterministically and validated through the same Pydantic models.
- Any content change creates a new version. Items are never edited in place during an experiment.

### 5.2 Category distribution

| Category | Items | Difficulty mix (easy / medium / hard) | Primary evaluation |
|---|---:|---|---|
| General knowledge | 15 | 5 / 7 / 3 | Exact match, plus refusal checks for unanswerable items |
| Logical reasoning | 15 | 4 / 7 / 4 | Final-answer exact match, plus one refusal check |
| Mathematical reasoning | 15 | 5 / 7 / 3 | Numeric validation with tolerance |
| Coding | 15 | 4 / 7 / 4 | Executed unit tests |
| Instruction following | 15 | 5 / 7 / 3 | Constraint checks |
| Structured JSON generation | 15 | 5 / 7 / 3 | JSON parse plus schema validation |
| Summarization | 10 | 2 / 5 / 3 | Reference overlap plus key-fact coverage; judge for quality |
| **Total** | **100** | **30 / 47 / 23** | |

The table reports the authored distribution, which the dataset model enforces.

### 5.3 Item schema

The dataset container stores `name`, `version`, `schema_version`, `created_at`, `random_seed`, `description`, `content_sha256`, and `items`. Every item contains:

| Field | Type | Default | Meaning |
|---|---|---|---|
| `id` | string | required | Stable unique identifier with a category prefix, e.g. `gk-001` |
| `category` | enum | required | One of the seven categories |
| `difficulty` | enum | required | `easy`, `medium`, `hard` |
| `prompt` | string | required | Exact user prompt sent to every model (minimum 20 characters) |
| `system_instruction` | string or null | `null` | Optional system message |
| `expected_answer` | string or null | `null` | Canonical answer for exact/numeric methods |
| `evaluation_type` | enum | required | Primary evaluator identifier |
| `reference_answer` | string or null | `null` | Reference used by code, JSON, overlap, refusal, or judge evaluation |
| `test_cases` | list of objects | `[]` | Authored unit tests for coding items |
| `json_schema` | object or null | `null` | JSON Schema for structured JSON items |
| `required_constraints` | list of objects | `[]` | Deterministic instruction constraints |
| `maximum_output_tokens` | integer | `512` | Per-item generation cap (1–4096) |
| `tags` | list of strings | `[]` | Topic tags for analysis |
| `is_unanswerable` | boolean | `false` | True for hallucination-resistance items |
| `judge_eligible` | boolean | `false` | Whether subjective judging applies |
| `max_score` | number | `1.0` | Maximum points for the item |
| `source` | string | `"authored"` | `authored`, or a licensed source with attribution |
| `notes` | string or null | `null` | Author notes, not shown to models |
| `accepted_answers` | list of strings | `[]` | Additional accepted exact-match answers |
| `key_facts` | list of strings | `[]` | Required facts for reference-overlap scoring |
| `forbidden_claims` | list of strings | `[]` | Claims that invalidate a refusal response |
| `function_name` | string or null | `null` | Required function name for coding items |
| `function_signature` | string or null | `null` | Required function signature for coding items |
| `numeric_tolerance` | number or null | `null` | Absolute tolerance for numeric evaluation |
| `expected_unit` | string or null | `null` | Expected unit for numeric evaluation |

Method-specific requirements are enforced by the model: exact-match and numeric items require `expected_answer`; coding items require `function_name`, `function_signature`, `test_cases`, and `reference_answer`; JSON items require `json_schema` and `reference_answer`; instruction items require `required_constraints`; reference-overlap items require `reference_answer` and `key_facts`; refusal items require `reference_answer`, `forbidden_claims`, and `is_unanswerable`. The `evaluation_config()` method derives the evaluator configuration; there is no stored `evaluation_config` field.

### 5.4 Evaluation method identifiers

- `exact_match`
- `numeric`
- `code_tests`
- `json_schema`
- `instruction_constraints`
- `reference_overlap`
- `refusal_check`
- `composite`

The dataset loader rejects unknown identifiers. Authored evaluation-type counts: `exact_match` 27, `numeric` 14, `code_tests` 15, `instruction_constraints` 14, `json_schema` 15, `reference_overlap` 9, `refusal_check` 6, `composite` 0.

### 5.5 Unanswerable items

Unanswerable items are embedded inside existing categories and replace answerable items; they do not increase the total count. Authored distribution:

| Category | Count | Scenario |
|---|---:|---|
| General knowledge | 2 | `gk-008` asks for the population of the fictional city of Eldoria in Valoria; `gk-014` asks for an exact historical population of the fictional village of Veyra. |
| Logical reasoning | 1 | `lr-014` gives insufficient premises to decide whether the pet is a cat. |
| Mathematical reasoning | 1 | `mr-014` gives a rectangle's area but not enough information to determine its length. |
| Instruction following | 1 | `if-013` asks for an exact population of the fictional city of Noria in exactly five words. |
| Summarization | 1 | `sum-009` asks for findings from a nonexistent 2026 clinical trial of the fictional drug Neuroclear. |
| **Total** | **6** | |

Expected behavior:

- State that the answer cannot be determined from the provided information, or ask for the missing information.
- Do not invent a confident answer.
- A refusal is not automatically correct: the response must also avoid asserting a fabricated answer.

Scoring:

- Deterministic refusal patterns are checked first.
- A judge may assess whether the refusal is appropriate and whether unsupported claims were made.
- The deterministic result is primary; the judge is secondary.
- Hallucination resistance is reported as `correct_refusal_rate` over unanswerable items, with the raw responses available for inspection.

### 5.6 Dataset authoring rules

- Write original items; do not copy protected benchmark content.
- Keep prompts self-contained and unambiguous.
- State the required output format explicitly in the prompt when format matters.
- Avoid prompts that require tools, browsing, or external files.
- Coding items must specify the function name, signature, and input assumptions; the evaluation environment uses Python 3.13.
- JSON items must include a complete schema or an unambiguous example and must forbid extra keys unless explicitly allowed.
- Summarization items must include the source text in the prompt and state the length constraint.
- Include the answer or a reference for every answerable item.
- Avoid answer leakage in the prompt.
- Keep prompts within a size that fits the configured context length with room for output.
- Document likely training-data contamination risk; originality reduces but does not eliminate it.

### 5.7 Prompt protocol

- Zero-shot prompting; no few-shot examples in the primary benchmark.
- No custom system message by default. Each model's Ollama template supplies its own default behavior. This preserves the models as distributed but means template differences are part of the measured behavior.
- The user prompt contains all task instructions, output format, and constraints.
- The same exact prompt string is sent to every candidate.
- No model-specific prompt tuning in the primary benchmark. Any prompt adaptation is a separate experiment.
- Raw prompts are stored with each response.

---

## 6. Generation settings

### 6.1 Primary warm-start settings

| Setting | Value | Rationale |
|---|---|---|
| Temperature | 0.0 | Maximizes reproducibility and fair deterministic scoring. |
| Top-p | 1.0 | Disables nucleus truncation; temperature 0 already makes sampling greedy. |
| Top-k | 40 | Ollama's common default, set explicitly so it is recorded. |
| Seed | 42 | Recorded for reproducibility; with temperature 0 it has little effect but is still passed. |
| Repeat penalty | 1.1 | Ollama's common default; reduces loops without heavily altering answers. |
| Context length (`num_ctx`) | 4096 | Keeps KV-cache memory controlled on a 4 GB GPU; all prompts fit with room for output. |
| Maximum output (`num_predict`) | 512 | Allows complete answers and code while limiting runaway generation. |
| Streaming | Enabled | Required for TTFT measurement. |
| Keep-alive | 30 minutes | Keeps the model warm during a model run; unloaded explicitly between models. |
| Request timeout | 300 seconds | Generous for CPU-offloaded 7B–8B inference; prevents indefinite hangs. |
| Connect timeout | 10 seconds | Localhost connection should be immediate. |
| System prompt | None (default template) | Avoids overriding model-specific behavior; documented as a design choice. |

### 6.2 Consistency settings

The consistency probe uses a nonzero temperature to measure response variability:

| Setting | Value |
|---|---|
| Temperature | 0.7 |
| Top-p | 0.9 |
| Top-k | 40 |
| Seed | 42 |
| Repeat penalty | 1.1 |
| Context length | 4096 |
| Maximum output | 512 |
| Repeats | 3 per selected item |

### 6.3 Cold-start settings

Cold-start requests use the primary settings except:

- The model is unloaded before each request.
- `keep_alive` is 0 for the measured request so the model unloads afterward.
- A short prompt with `num_predict = 64` is used to isolate load and first-token behavior.
- Load duration is read from the final Ollama response.

### 6.4 Why these settings are fixed

The goal is to compare models, not to tune each model to its best possible output. Fixed settings reduce researcher degrees of freedom. If a model performs poorly under these settings, that is a finding about the model under the stated conditions, not proof that no better settings exist. A separate hyperparameter study would be needed to make the latter claim.

The 512-token output cap may truncate very long code or explanations. Coding items are authored to fit within the cap, and any response that ends because of the cap is recorded as a limitation rather than silently scored as complete.

### 6.5 Settings that are not forced

- GPU layer count (`num_gpu`) is not forced. Ollama chooses placement, and the resulting `size_vram` is recorded.
- Stop sequences are not overridden; model templates may supply their own.
- Sampling parameters not listed above use Ollama defaults and are recorded as "default" rather than assumed values.

---

## 7. Experiment profiles

The full benchmark suite is composed of four profiles. A warm experiment run contains Q, P, and S. A cold experiment run contains C and is stored separately.

Within a warm model run, profiles execute in the order **P → Q → S**. The performance probe runs first, immediately after warm-up, to reduce the effect of thermal drift from the long quality pass. Quality is less sensitive to thermal state, and consistency runs last.

### 7.1 Profile Q — Quality

| Property | Value |
|---|---|
| Purpose | Measure response quality on all 100 items |
| Items | 100 |
| Repeats per item | 1 |
| Temperature | 0.0 |
| Model state | Warm; after warm-up and Profile P |
| Requests per model | 100 |
| Requests for three models | 300 |
| Used for | Deterministic scores, judge scores, quality analysis |

One pass is used because temperature 0 makes repeated quality passes largely redundant and expensive. Repeated runs for final conclusions are handled at the whole-run level.

### 7.2 Profile P — Performance

| Property | Value |
|---|---|
| Purpose | Measure stable latency and throughput |
| Prompts | 9 fixed performance prompts: 3 short (18–19 words, ~25 tokens), 3 medium (101–107 words, ~140 tokens), 3 long (419–456 words, ~550–600 tokens) |
| Repeats per prompt | 5 |
| Temperature | 0.0 |
| Model state | Warm, after one discarded warm-up; P runs before Q |
| Requests per model | 45 |
| Requests for three models | 135 |
| Used for | Mean, median, P95 latency; TTFT; tokens/s; prompt processing speed |

The performance prompts are fixed and stored with the dataset version or as a separate versioned probe file. They are not used for quality scoring. Each prompt includes an explicit target output length (approximately 60, 120, and 200 words for the short, medium, and long prompt groups) so output length is comparable across models; `num_predict = 512` remains the hard cap. Each repeat uses the same base prompt and settings. All repeats are retained; the first measured repeat is not discarded.

**Prompt-cache control:** Ollama may reuse a prompt prefix across identical requests, which can inflate prompt-processing speed and reduce TTFT on later repeats. To prevent this, each performance request is prefixed with a short nonce containing the performance prompt ID and repeat index (for example `[perf P3 R2]`). The nonce is identical across models for the same prompt/repeat pair, and the full prompt is stored with the response. Record `prompt_eval_count` for every request and flag implausibly low `prompt_eval_duration` values as possible cache reuse. Consistency repeats deliberately use identical prompts and are not used for latency or throughput metrics.

### 7.3 Profile C — Cold start

| Property | Value |
|---|---|
| Purpose | Measure model load and first-request behavior from an unloaded state |
| Prompt | 1 short fixed prompt |
| Repeats per model | 3, with unload between each |
| Maximum output | 64 tokens |
| Model state | Unloaded before each request; `keep_alive = 0` |
| Requests per model | 3 |
| Requests for three models | 9 |
| Used for | Load duration, cold TTFT, cold total latency |

Cold start is a separate experiment run. It must not be mixed with warm-start aggregates. Because Windows file caches are not cleared, this is a "cold model, warm OS cache" measurement; the limitation is recorded.

### 7.4 Profile S — Consistency

| Property | Value |
|---|---|
| Purpose | Measure response variability under sampling |
| Items | 10 items stratified across categories |
| Repeats per item | 3 |
| Temperature | 0.7 |
| Model state | Warm; after P and Q |
| Requests per model | 30 |
| Requests for three models | 90 |
| Used for | Similarity across repeats; consistency score |

### 7.5 Request budget

| Run type | Per model | Three models |
|---|---:|---:|
| Warm run (P + Q + S) | 175 | 525 |
| Cold run (C) | 3 | 9 |
| One complete suite | 178 | 534 |

Final experiments repeat the complete suite three times: approximately 1,602 candidate requests; judge calls are additional. Development and debugging use a reduced smoke profile that is clearly labeled and never used for conclusions.

### 7.6 Smoke profile (not for conclusions)

- 10 dataset items, one pass, warm only.
- 1 performance prompt, 1 repeat.
- No cold start, no consistency probe.
- Purpose: verify plumbing and UI, not model ranking.
- Stored with `is_smoke = true` and excluded from recommendations.

### 7.7 Runtime expectations

The final experiment is intentionally expensive. As a planning estimate only, at an illustrative 8 output tokens per second and an average of 150 generated tokens per quality or consistency response, one complete suite is roughly 2-4 hours including warm-ups and judging; three suites are roughly 8-15 hours. Actual duration depends on measured throughput, output lengths, thermal state, and judge calls, and must be recorded from the run. The application must support resuming interrupted runs. Final runs may span multiple sessions. If time is constrained, use the smoke profile for development and disclose any reduction in the final design; do not silently shrink the experiment.

---

## 8. Warm-up and cooldown procedures

### 8.1 Warm-up

1. Load the model with `keep_alive = "30m"`.
2. Wait until `/api/ps` reports the model loaded.
3. Send one warm-up request not drawn from the benchmark: a short generic prompt, `num_predict = 32`, temperature 0.
4. Discard the warm-up response from all metrics.
5. Wait 5 seconds for clocks and caches to settle.
6. Begin Profile P (performance), then Profile Q (quality), then Profile S (consistency).

Warm-up is performed once per model run, not once per item. The warm-up prompt and response are stored in logs but marked `is_warmup = true` and excluded from analysis.

### 8.2 Cooldown

- 60 seconds between candidate models in a warm run.
- 5 minutes between complete final runs.
- 60 seconds between cold-start repetitions.
- The laptop stays on AC power and no other heavy workload runs during the experiment.
- The operator records any observed thermal throttling or background activity in the run notes.

### 8.3 Model unload

1. Request unload using Ollama's supported mechanism (`keep_alive = 0` on a request, or the `ollama stop` CLI).
2. Poll `/api/ps` every second for up to 60 seconds.
3. Confirm the model is absent before a cold-start measurement or before loading the next candidate.
4. Record unload success or failure.

---

## 9. Execution order

### 9.1 Model order

- Development runs: fixed order Llama, Qwen, Mistral.
- Final runs: balanced order across three runs:
  - Run 1: Llama → Qwen → Mistral
  - Run 2: Qwen → Mistral → Llama
  - Run 3: Mistral → Llama → Qwen
- The order is recorded in the experiment run.
- The purpose is to distribute thermal and cache order effects across models.

### 9.2 Item order

- Dataset items are shuffled once per experiment configuration using seed 42.
- The same shuffled order is used for all three candidates.
- The order is stored with the configuration and the response records.
- Performance prompts are ordered short, medium, long, with repeats sequential.
- Consistency items are selected by a seeded stratified procedure and stored.

### 9.3 Concurrency

- No concurrent candidate requests.
- No concurrent model loads.
- The dashboard may display progress, but it must not issue inference requests during a run.
- Hardware sampling runs in a background thread or async task and must not block the inference stream.

---

## 10. Measurement windows and timing

### 10.1 Client timing

- `t0`: immediately before issuing the HTTP request, using `time.perf_counter_ns()`.
- `t_first`: arrival of the first stream chunk containing non-empty content.
- `t_last`: arrival of the final chunk with `done = true`.
- `TTFT = t_first − t0`.
- `total_latency = t_last − t0`.
- UTC wall-clock timestamps are stored for auditing; monotonic values are used for durations.
- Empty content chunks, role-only chunks, and metadata-only chunks do not count as the first token.
- Phase 04 records the first non-empty content chunk as `t_first`; the final `done = true` chunk supplies `t_last` and raw Ollama metadata.

### 10.2 Ollama runtime timing

The final response chunk supplies:

- `total_duration`
- `load_duration`
- `prompt_eval_count`
- `prompt_eval_duration`
- `eval_count`
- `eval_duration`

These are stored in nanoseconds and converted to seconds for analysis. They describe Ollama's internal timing and are not interchangeable with client timings.

### 10.3 Hardware sampling window

- Baseline: 2 seconds before the request, sampled every 500 ms.
- Inference: from `t0` to `t_last`, sampled every 500 ms.
- A final sample is taken at completion.
- The sampling interval is configurable between 250 ms and 1000 ms and is recorded.
- If the sampler cannot keep up, the actual sample count and interval are reported.

### 10.4 What is excluded

- Warm-up requests.
- Judge requests from candidate performance metrics.
- Requests that failed before any response, from latency and throughput aggregates, while still counting them in reliability.
- Retried requests, if any are explicitly enabled for non-benchmark purposes.

### 10.5 Phase 04 implementation notes

- The interim response log is append-only JSONL under `data/results/` until Phase 07 storage exists.
- Each record captures experiment, model, prompt, repetition, warm-up and cold-start flags, status, response text, finish reason, client timing, token metrics, generation options, Ollama metadata, and error details.
- Prompt and output token rates are calculated from the matching Ollama token count and duration; they are not inferred from text length.
- Warm-up records are stored and marked but excluded from measured summaries.
- `benchmark infer` defaults to the first configured candidate and five ordered dataset items; explicit `--model` and `--item` selections are required for targeted runs.
- A five-question smoke with `llama3.1:8b` produced all-success records in experiment `20261110T052647Z-afd1e4d4`; it is engineering validation, not a comparative result.

---

## 11. Evaluation methodology

### 11.1 Principles

1. Deterministic checks are primary for objective tasks.
2. The judge is used only for subjective dimensions.
3. The judge never overrides a deterministic failure.
4. Every evaluation stores its evaluator name, version, score, pass/fail, and details.
5. Raw responses remain available for manual review.
6. Evaluation failures are recorded, not silently converted to zero unless the rubric explicitly says so.

### 11.2 Category evaluation matrix

| Category | Primary deterministic method | Secondary method | Judge weight in composite |
|---|---|---|---:|
| General knowledge | Key-fact matching against reference; normalized exact match where applicable | Judge for completeness and coherence | 0.30 |
| Logical reasoning | Final-answer exact match | Judge for reasoning quality | 0.30 |
| Mathematical reasoning | Numeric validation with tolerance | Judge for explanation quality only | 0.10 |
| Coding | Extracted code executed against unit tests | Judge for readability only, never for correctness | 0.10 |
| Instruction following | Constraint checks | Judge for relevance only | 0.10 |
| Structured JSON | Parse plus schema validation | Judge for field completeness only | 0.10 |
| Summarization | Token overlap plus key-fact coverage | Judge for relevance, completeness, coherence | 0.40 |

Composite weight = `1 − judge_weight` for the deterministic component. If the judge is unavailable, the deterministic component is used alone and the missing judge score is flagged. The exact composite formula is in `METRICS_DEFINITIONS.md`.

### 11.3 Deterministic evaluators

#### Exact match

- Normalize with Unicode NFKC, lowercase, whitespace collapse, and removal of leading/trailing punctuation.
- Compare to the accepted answer set.
- Store both raw and normalized strings.
- Partial credit only when the item's evaluation configuration defines accepted alternatives.

#### Numeric validation

- Extract the final numeric answer using a documented parser.
- Accept integers, decimals, fractions, and scientific notation.
- Tolerance: absolute `1e-6` or relative `0.5%`, whichever is larger, unless the item overrides it.
- Check units when the item requires them.
- If multiple numbers appear, the final explicitly stated answer is used; ambiguous responses fail.

#### Code tests

- Extract the first complete Python code block, or the full response if no block is present.
- If the item specifies a function signature, verify it exists.
- Run each test in a fresh subprocess with:
  - a temporary working directory,
  - Python isolated mode (`-I`),
  - a minimal environment,
  - a per-test timeout of 5 seconds and a total timeout of 20 seconds,
  - stdout/stderr capture capped at 10 KB,
  - no shell execution.
- A test passes only if it exits cleanly with the expected result.
- `pass@1` is the primary coding score.
- The subprocess is best-effort isolation, not a security boundary. Final untrusted runs should use a dedicated low-privilege account or VM.
- Test code is authored by the project and stored with the dataset item.

#### JSON validation

- Extract JSON from the response, allowing a fenced code block.
- Parse strictly; do not repair malformed JSON for scoring.
- Validate against the item's schema.
- Partial credit may be defined for valid JSON but invalid schema, or schema-valid but wrong values. The item's configuration defines the breakdown.
- Extra keys are rejected unless the schema allows them.
- Model-generated JSON is never used to control privileged operations.

#### Instruction constraints

- Constraints are explicit and machine-checkable: word count, sentence count, bullet count, required phrases, forbidden phrases, language, case, prefix/suffix, ordering, and format.
- Each constraint is scored separately, then combined as defined by the item.
- Subjective style is not scored deterministically.

#### Reference overlap

- Tokenize to lowercase words.
- Compute unigram precision, recall, and F1 against the reference.
- Compute a longest-common-subsequence F1 as a lightweight ROUGE-L analogue.
- Compute key-fact coverage from a configured list of required facts.
- Report all components; the item configuration defines the composite.
- This is a lexical measure and does not understand paraphrase; the judge complements it for summarization.

#### Refusal check

- Detect explicit uncertainty or refusal patterns.
- Check that the response does not assert a specific fabricated answer.
- Require both appropriate refusal and absence of fabricated specifics for full credit.
- Judge review is secondary.

### 11.4 LLM-as-a-judge

#### Judge configuration

| Setting | Value |
|---|---|
| Model | `qwen2.5:3b` |
| Temperature | 0.0 |
| Seed | 42 |
| Top-p | 1.0 |
| Top-k | 40 |
| Repeat penalty | 1.1 |
| Context length | 4096 |
| Maximum output | 300 |
| Calls per response | 1 by default |
| Reliability sample | 10% of judged responses judged 3 times at temperature 0.7 with three distinct recorded seeds |

Primary judge scores use temperature 0. The reliability sample uses temperature 0.7 and three distinct seeds so repeated calls can reveal judge variance; those calls are stored separately and excluded from the primary judge scores.

#### Dimensions and scale

Each dimension is scored as an integer from 1 to 5.

| Dimension | 1 | 3 | 5 |
|---|---|---|---|
| Relevance | Off-topic or does not address the request | Partially addresses the request with omissions | Directly addresses the request with no irrelevant material |
| Completeness | Missing most required elements | Covers some required elements | Covers all required elements |
| Coherence | Disorganized or contradictory | Understandable but with structural or clarity problems | Clear, organized, and internally consistent |
| Reasoning quality | No valid reasoning or invalid logic | Partly sound reasoning with gaps | Sound, sufficient, and well-connected reasoning |

Scores 2 and 4 are intermediate. The judge must return a short rationale for each dimension.

#### Judge input

The judge receives:

1. The exact task prompt.
2. The reference answer or reference text, when available.
3. The candidate response, clearly delimited as untrusted data.
4. The rubric and the required JSON output format.

The judge does not receive:

- The candidate model name or tag.
- Other candidates' responses.
- Any expected score or ranking.
- Any instruction to prefer a particular style.

#### Judge output

Strict JSON:

```json
{
  "relevance": {"score": 1, "rationale": "..."},
  "completeness": {"score": 1, "rationale": "..."},
  "coherence": {"score": 1, "rationale": "..."},
  "reasoning_quality": {"score": 1, "rationale": "..."}
}
```

The parser validates scores are integers from 1 to 5. Invalid output gets one retry with a stricter instruction. If the retry fails, the judge evaluation is stored as failed and excluded from judge aggregates; it is not replaced with a zero.

#### Blinding and ordering

- Candidate identity is removed from judge prompts.
- Responses are scored individually, not side-by-side, which avoids position bias.
- If multiple responses are judged in one session, their order is randomized with a recorded seed.
- The judge model's own family is disclosed as a possible source of self-preference.

#### Manual validation

- A stratified random sample of at least 10% of judged responses is reviewed manually.
- The reviewer records scores without seeing the judge's scores first.
- Report exact agreement, agreement within one point, and rank correlation where sample size permits.
- Disagreements are discussed in the limitations section.
- Manual scores are stored separately and never overwrite judge scores.

#### Known bias risks

- **Self-preference:** Qwen 2.5 3B may favor Qwen 2.5 7B outputs.
- **Verbosity bias:** Longer answers may receive higher completeness scores.
- **Style bias:** Fluent but incorrect answers may receive higher coherence scores.
- **Leniency:** The judge may avoid low scores.
- **Prompt injection:** A candidate response may attempt to manipulate the judge; delimiters and instructions mitigate but do not eliminate this.
- **Language and cultural bias:** The dataset is English-only.
- **Reference bias:** The judge may over-reward similarity to the reference even when another correct form exists.

Mitigations include blinding, single-response scoring, a fixed rubric, rationale requirements, a manual validation sample, optional secondary-judge cross-checks, and reporting deterministic and judge scores separately.

---

## 12. Reliability and failure policy

### 12.1 Failure classes

- `connection_error`
- `http_error`
- `model_not_found`
- `timeout`
- `stream_parse_error`
- `empty_response`
- `ollama_error`
- `cancelled_by_user`
- `internal_error`

### 12.2 Policy

- Candidate requests are not retried by default.
- A failed request is stored with its error class, message, stage, and timestamps.
- Timeouts count as reliability failures but are excluded from latency and throughput aggregates.
- Cancellations are reported separately and excluded from the reliability denominator, with the count disclosed.
- If a run is interrupted, completed responses remain valid; the run can be resumed without duplicating them.
- A model with a failure rate above 20% on a workload is not eligible for a top recommendation; the recommendation is marked "insufficient reliability."
- Outliers are retained. A sensitivity analysis may report trimmed statistics, clearly labeled as secondary.
- No missing latency is imputed.

### 12.3 Thresholds

| Condition | Action |
|---|---|
| Failure rate ≤ 5% | Normal confidence |
| Failure rate 5–20% | Reduced confidence; note in report |
| Failure rate > 20% | No top recommendation; reliability warning |
| Missing metric weight > 30% of a profile | Recommendation marked "insufficient data" |
| Judge invalid output after retry | Judge score excluded; deterministic score retained |
| Dataset hash mismatch | Run blocked |
| Model digest mismatch | Run blocked unless explicitly overridden and recorded |

---

## 13. Reproducibility checklist

Before a run:

- [ ] Ollama version recorded.
- [ ] Model tags and digests recorded.
- [ ] Quantization and context lengths recorded.
- [ ] Dataset version and SHA-256 hash recorded.
- [ ] Generation settings recorded.
- [ ] Seed and item order recorded.
- [ ] Judge tag, digest, and rubric version recorded.
- [ ] Host CPU, RAM, GPU, driver, OS, and Python versions recorded.
- [ ] Free disk space recorded.
- [ ] Sampling interval and timeout recorded.
- [ ] Cold or warm run type recorded.
- [ ] Model order recorded.
- [ ] No other heavy workload running; operator note stored.

After a run:

- [ ] Raw responses stored.
- [ ] Raw Ollama final metadata stored.
- [ ] Hardware samples stored.
- [ ] Evaluation details stored.
- [ ] Failures and timeouts stored.
- [ ] Aggregates include sample sizes.
- [ ] Report includes limitations.
- [ ] Historical records are not modified.

---

## 14. Standard run protocol

1. **Preflight.** Confirm AC power, close GPU-heavy applications, check disk space, verify Ollama is reachable, and confirm the dataset hash.
2. **Model verification.** Discover models, verify digests, pull missing models only with explicit approval, and record metadata.
3. **Configuration.** Create an immutable experiment configuration: dataset version, model tags, generation options, profiles, judge settings, scoring weights, and run type.
4. **Warm run.**
   - For each model in the recorded order:
     - Load the model.
     - Record `/api/ps` layer placement after load.
     - Run and discard one warm-up.
     - Record `/api/ps` layer placement again after warm-up.
     - Execute Profile P.
     - Execute Profile Q.
     - Execute Profile S.
     - Record `/api/ps` at the end of the run and flag any placement change.
     - Unload the model.
     - Cooldown.
5. **Cold run (separate experiment run).**
   - For each model:
     - Confirm unloaded.
     - Execute three cold-start repetitions with unload between.
     - Record load duration and first-token latency.
6. **Evaluation.** Run deterministic evaluators, then the judge on eligible responses, then compute scores.
7. **Analytics.** Compute latency, throughput, resource, reliability, and quality aggregates with sample sizes.
8. **Recommendations.** Apply workload profiles, missing-data policy, and confidence rules.
9. **Reporting.** Export CSV, JSON, HTML, and PDF.
10. **Archive.** Mark the run complete; do not modify its records.

---

## 15. Validity threats and limitations

### 15.1 Internal validity

- Thermal drift and clock changes across a long run.
- Background Windows activity and GPU desktop use.
- Order effects from model loading and disk cache.
- Model template differences, which are part of the measured behavior but may confound prompt handling.
- Judge bias and stochasticity.
- Dataset contamination in pretraining data.
- Evaluator imperfections, especially lexical overlap and code test coverage.
- Prompt-prefix cache reuse by Ollama, mitigated with nonce-prefixed performance requests and monitored through `prompt_eval_count` and `prompt_eval_duration`.
- GPU layer-placement changes between load and the end of a run, monitored through `/api/ps`.

### 15.2 External validity

- Results apply to these exact quantized artifacts, this Ollama version, this hardware, this OS, and this dataset.
- Results may not generalize to unquantized models, other runtimes, other hardware, or other languages.
- Zero-shot performance may differ from few-shot or tuned-prompt performance.
- A 100-item dataset cannot cover all capabilities.

### 15.3 Construct validity

- Benchmark scores are proxies for quality, not direct measures of usefulness.
- The judge measures judged quality, not truth.
- Code tests measure specific functional correctness, not software engineering quality.
- Throughput on short prompts may not predict long-context performance.
- The 4,096-token context cap means long-context capability is not evaluated.

### 15.4 Statistical validity

- Three models and repeated measurements provide limited statistical power.
- P95 with 45 samples is indicative, not precise.
- Multiple category comparisons increase the chance of apparent differences.
- The project will not claim statistical significance without an appropriate test.
- Final reports may include bootstrap confidence intervals and effect sizes, clearly labeled as exploratory.

### 15.5 Measurement validity

- Client TTFT includes localhost HTTP overhead, scheduler delay, and model load if cold.
- Ollama's `eval_duration` excludes network and client overhead.
- Windows WDDM may make per-process GPU memory unavailable.
- System VRAM includes desktop and other applications.
- Ollama process RSS may not isolate model memory from server overhead.
- CPU utilization is system-wide and sampled, not continuous.
- The 512-token output cap may truncate long code or explanations; coding items are designed for concise solutions, and truncation is recorded.
- Prompt-cache reuse can make repeated prompts look faster; performance requests use nonces, and anomalies are flagged.
- GPU utilization sampling may miss short spikes.
- Laptop power and CPU temperature are not reliably measured.

### 15.6 Mitigation summary

- Record everything, including nulls.
- Separate client and runtime metrics.
- Separate system and model VRAM.
- Repeat measurements.
- Balance model order.
- Use fixed seeds and settings.
- Blind the judge.
- Validate a judge sample manually.
- Report limitations with every conclusion.
- Never fabricate a missing value.

---

## 16. What would falsify the conclusions

A conclusion would be weakened or falsified if:

- A rerun with the same configuration and digests produces materially different rankings.
- The model digest or quantization differs from the recorded artifact.
- The judge's manual validation shows low agreement.
- Failures or timeouts are concentrated in one model and change the ranking.
- Thermal throttling or background load explains the performance differences.
- Deterministic and judge scores disagree strongly and the composite is sensitive to the weights.
- The dataset is found to be contaminated or ambiguous.

The project must report these possibilities rather than hide them.

---

## 17. Changes to this methodology

- Any change to prompts, settings, evaluators, judge rubric, dataset content, or scoring weights requires a new configuration version.
- Changes are documented in the phase report and in this document's version history.
- Historical runs retain the methodology version used at the time.
- The project does not retroactively edit results to match a new methodology.
