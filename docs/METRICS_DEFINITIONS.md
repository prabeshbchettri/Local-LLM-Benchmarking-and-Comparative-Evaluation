# Metrics Definitions — Local LLM Benchmarking and Comparative Evaluation Framework

> **Document purpose:** Define every metric used by the project, including its formula, data source, aggregation rules, units, and limitations. This document prevents ambiguous reporting and makes client-measured, Ollama-reported, and hardware-sampled values distinguishable.

| Field | Value |
|---|---|
| Document version | 1.1.0 |
| Status | Phase 00 baseline; Phase 05 hardware implementation aligned |
| Date | 2026-11-10 |
| Related documents | `EXPERIMENTAL_METHODOLOGY.md`, `PROJECT_REQUIREMENTS.md`, `ARCHITECTURE.md` |

---

## 1. Conventions

### 1.1 Time

| Symbol | Meaning |
|---|---|
| `t0` | Monotonic timestamp immediately before the HTTP request is issued |
| `t_first` | Monotonic timestamp when the first non-empty content chunk arrives |
| `t_last` | Monotonic timestamp when the final `done = true` chunk arrives |
| `L` | Total client latency |
| `T` | Time to first token |

Durations are computed with `time.perf_counter()` and stored in milliseconds for client metrics. Ollama durations are stored in nanoseconds exactly as received and converted to seconds for derived metrics.

### 1.2 Sample sets

Metrics are computed over explicitly named sample sets. Mixing sets is a reporting error.

| Sample set | Contents |
|---|---|
| `warm_quality` | Profile Q successful responses, warm model |
| `warm_performance` | Profile P successful responses, warm model |
| `warm_consistency` | Profile S responses, warm model, temperature 0.7 |
| `cold_start` | Profile C successful responses, unloaded model before each request |
| `judge` | Judge calls only; never mixed with candidate performance |
| `all_candidate` | All measured candidate requests, including failures, excluding warm-ups and cancellations where noted |

### 1.3 Status classes

| Status | Meaning |
|---|---|
| `success` | HTTP 200, stream completed, final `done = true`, usable content |
| `failure` | Non-timeout error: connection, HTTP, model, parse, or empty response |
| `timeout` | Client-side request timeout |
| `cancelled` | User interrupted the run or request |
| `warmup` | Excluded from all metrics |

### 1.4 Null policy

- Unavailable measurements are stored as `null`.
- Nulls are never converted to zero.
- Aggregates report the number of valid samples.
- A metric with no valid samples is reported as `null` with reason `unavailable`.
- Approximate measurements carry an `approximate` flag and a note.

---

## 2. Metric source taxonomy

| Source | Examples | Trust level |
|---|---|---|
| Client | TTFT, total latency, request start/end timestamps | Directly measured; includes localhost overhead |
| Ollama API | `prompt_eval_count`, `eval_count`, `prompt_eval_duration`, `eval_duration`, `load_duration`, `total_duration` | Runtime-reported; excludes client/network overhead |
| Ollama `/api/ps` | Loaded model `size`, `size_vram`, context, expiry | Ollama's model accounting |
| psutil | CPU, system RAM, Ollama process RSS | System/process sampled; approximate for attribution |
| NVML / nvidia-smi | GPU utilization, GPU memory utilization, VRAM used/total, GPU temperature | System-wide; WDDM limitations |
| Derived | Means, medians, percentiles, rates, normalized scores | Computed from the above with documented formulas |

---

## 3. Latency metrics

### 3.1 Time to first token (TTFT)

**Definition:** Time from request issue to the first stream chunk containing non-empty generated content.

**Formula:**

```
TTFT_ms = (t_first − t0) × 1000
```

**Source:** Client, streaming response.

**Exclusions:** Empty role chunks, metadata-only chunks, and whitespace-only content do not count.

**Caveats:**

- Includes localhost HTTP overhead, Ollama scheduler delay, and prompt processing.
- On cold start, includes model load time.
- Not the same as Ollama's `prompt_eval_duration`, which measures only prompt evaluation.
- A response that produces no content before timeout has no TTFT; it is a failure or timeout.

### 3.2 Total latency

**Definition:** Time from request issue to the final stream chunk.

**Formula:**

```
Latency_ms = (t_last − t0) × 1000
```

**Source:** Client.

**Caveats:**

- Includes TTFT and all output generation time.
- Includes localhost transport and client parsing overhead.
- On cold start, includes model load time.
- Excludes post-processing and evaluation time.

### 3.3 Mean latency

**Definition:** Arithmetic mean of total latency over a sample set.

**Formula:**

```
mean_latency = (1/n) × Σ Latency_i
```

**Sample set:** `warm_performance` by default; also reported for `warm_quality` separately.

**Caveats:** Sensitive to outliers; report median and P95 alongside.

### 3.4 Median latency

**Definition:** 50th percentile of total latency.

**Formula:** Linear-interpolation percentile over sorted successful latencies.

**Caveats:** More robust to outliers than the mean.

### 3.5 P95 latency

**Definition:** 95th percentile of total latency.

**Formula:** Linear-interpolation percentile over sorted successful latencies.

**Caveats:**

- With 45 performance samples, P95 is an order statistic near the maximum and is unstable.
- Report `n` with every P95.
- For `n < 20`, label P95 as indicative only.
- Do not compare P95 values without also reporting the distribution and sample size.

### 3.6 Model load duration

**Definition:** Time Ollama reports for loading the model.

**Formula:**

```
load_duration_s = load_duration_ns / 1e9
```

**Source:** Ollama final response metadata.

**Caveats:**

- Meaningful primarily on cold start; may be zero or omitted when the model is already loaded.
- "Cold start" means the model was unloaded, but the OS file cache may still be warm.
- Do not treat load duration as pure disk-read time.

### 3.7 Ollama total duration

**Definition:** Ollama-reported total time for the request.

**Formula:**

```
ollama_total_duration_s = total_duration_ns / 1e9
```

**Source:** Ollama final response metadata.

**Caveats:** Not identical to client latency; excludes client/network overhead and may exclude queueing outside the runner.

### 3.8 Client versus Ollama timing

| Aspect | Client TTFT | Ollama prompt eval |
|---|---|---|
| What it measures | Request issue to first visible token | Prompt evaluation inside the runner |
| Includes queueing | Yes | Not necessarily |
| Includes model load | Yes on cold start | No |
| Includes HTTP overhead | Yes | No |
| Use | User-perceived responsiveness | Runtime efficiency |

Both are reported. Neither replaces the other.

---

## 4. Throughput metrics

### 4.1 Prompt processing speed

**Definition:** Prompt tokens processed per second, as reported by Ollama.

**Formula:**

```
prompt_tps = prompt_eval_count / (prompt_eval_duration_ns / 1e9)
```

**Guards:**

- If `prompt_eval_duration_ns` is missing or zero, the metric is `null`.
- If `prompt_eval_count` is missing, the metric is `null`.

**Caveats:** Measures runner prompt evaluation, not client-perceived prompt latency. Prompt-prefix cache reuse can inflate this value; performance prompts use unique nonces, and implausibly low `prompt_eval_duration` values are flagged as possible cache reuse.

### 4.2 Effective prompt speed (client)

**Definition:** Prompt tokens per second of client TTFT. This is a lower bound because TTFT includes queueing and load.

**Formula:**

```
effective_prompt_tps = prompt_eval_count / (TTFT_ms / 1000)
```

**Source:** Mixed: Ollama token count plus client time.

**Caveats:** Label clearly as effective; do not present as pure prompt processing speed.

### 4.3 Output tokens per second

**Definition:** Generated tokens per second of Ollama evaluation time.

**Formula:**

```
output_tps = eval_count / (eval_duration_ns / 1e9)
```

**Guards:** `null` if `eval_count` or `eval_duration_ns` is missing or zero.

**Caveats:**

- Excludes prompt processing and model load.
- Excludes client/network overhead.
- Short outputs produce noisier estimates.
- Do not infer from stream chunk count; chunks are not tokens.

### 4.4 Effective output rate (client)

**Definition:** Generated tokens per second after the first token, using client-observed time.

**Formula:**

```
effective_output_tps = eval_count / ((Latency_ms − TTFT_ms) / 1000)
```

**Guards:**

- `null` if `Latency_ms ≤ TTFT_ms`.
- `null` if `eval_count` is missing.

**Source:** Mixed: Ollama token count plus client times.

**Caveats:** Includes client streaming overhead; useful as a user-perceived rate, not as a runtime benchmark.

### 4.5 Token counts

| Field | Meaning | Source |
|---|---|---|
| `prompt_eval_count` | Prompt tokens evaluated | Ollama |
| `eval_count` | Output tokens generated | Ollama |
| `total_tokens` | `prompt_eval_count + eval_count` | Derived |

Token counts are model-tokenizer-specific. Do not compare raw token counts across models as if they were words.

### 4.6 Streaming chunk rate (diagnostic only)

**Definition:** Number of non-empty content chunks per second.

**Use:** Diagnostic only. It is not a token rate and must never be reported as tokens per second.

---

## 5. Reliability metrics

### 5.1 Definitions

Let:

- `attempted` = measured candidate requests started, excluding warm-ups and judge calls.
- `cancelled` = requests cancelled by the user.
- `eligible` = `attempted − cancelled`.
- `success` = requests meeting the success definition.
- `failure` = non-timeout failures.
- `timeout` = client timeouts.

### 5.2 Formulas

```
success_rate = success / eligible
failure_rate = failure / eligible
timeout_rate = timeout / eligible
cancellation_rate = cancelled / attempted
```

### 5.3 Reporting rules

- Report numerator and denominator, not only percentages.
- Report all four rates.
- A timeout is not counted as success.
- A cancellation is not counted as a model failure, but it is disclosed.
- Reliability denominators are per model and per run type; do not pool warm and cold runs.
- A failure rate above 20% on a workload triggers the "insufficient reliability" rule in recommendations.

### 5.4 Response consistency

**Definition:** Similarity of repeated responses to the same prompt under the consistency profile.

For each item with `R` repeats:

- Normalize each response: NFKC, lowercase, whitespace collapse, strip.
- Exact-repeat indicator: `1` if two normalized responses are identical, else `0`.
- Token F1: unigram precision/recall F1 between two responses.
- Sequence similarity: `difflib.SequenceMatcher` ratio, reported as a secondary lexical measure.

Aggregate per item:

```
item_consistency = mean over pairs of token_f1
item_exact_repeat_rate = mean over pairs of exact_repeat_indicator
```

Aggregate per model:

```
consistency = mean over items of item_consistency
exact_repeat_rate = mean over items of item_exact_repeat_rate
```

**Caveats:**

- Lexical similarity penalizes valid paraphrase.
- Temperature 0.7 intentionally introduces variability; low similarity is not automatically bad.
- Report both token F1 and exact-repeat rate.
- Consistency is not a quality measure.

### 5.5 Correct refusal rate (hallucination resistance)

**Definition:** Fraction of unanswerable items for which the model appropriately refuses or states uncertainty without fabricating an answer.

**Formula:**

```
correct_refusal_rate = correct_refusals / unanswerable_items
```

**Source:** Deterministic refusal check, optionally reviewed by the judge.

**Caveats:** Pattern-based refusal detection can miss unusual phrasings; raw responses are retained for review.

---

## 6. Quality metrics

### 6.1 Item score

**Definition:** Points awarded by the item's primary evaluator, from 0 to `max_score`.

**Source:** Deterministic evaluator, or a documented composite.

**Storage:** One evaluation record per evaluator per response, with details.

### 6.2 Deterministic category score

**Formula:**

```
category_score = (Σ item_score_i / Σ max_score_i) × 100
```

**Caveats:** Partial credit depends on the item's evaluation configuration; report the configuration version.

### 6.3 Overall deterministic quality

**Primary definition:** Equal-weight mean of the seven category scores.

```
overall_deterministic = (1/7) × Σ category_score_c
```

**Sensitivity definition:** Item-weighted mean across all items.

```
item_weighted_quality = (Σ item_score_i / Σ max_score_i) × 100
```

Both are reported. The equal-category-weight score is primary because it prevents categories with more items from dominating. The item-weighted score is a sensitivity check.

**Coverage rule:** Quality scores are conditional on successful responses. Report `quality_coverage = evaluated_items / expected_items` per model and per category. Failed items are not imputed as 0 in the primary quality score. A failure-as-zero sensitivity score is reported separately. If quality coverage is below 80%, flag the quality comparison as incomplete; below 50%, do not rank quality. The reliability subscore remains the primary measure of failure behavior.

### 6.4 Judge dimension score

**Definition:** Mean raw judge score for a dimension over judged responses.

```
judge_dimension_raw = mean(score_i) over judged responses
judge_dimension_normalized = ((judge_dimension_raw − 1) / 4) × 100
```

**Caveats:** Judge scores are subjective and model-dependent; report the judge tag and digest.

### 6.5 Judge item score

**Formula:**

```
judge_item_raw = mean(relevance, completeness, coherence, reasoning_quality)
judge_item_normalized = ((judge_item_raw − 1) / 4) × 100
```

### 6.6 Composite quality

For each category `c` with judge weight `w_c`:

```
category_composite_c = (1 − w_c) × category_score_c + w_c × judge_category_score_c
```

If the judge score is unavailable:

```
category_composite_c = category_score_c
```

with a `judge_missing = true` flag.

Overall:

```
overall_composite = (1/7) × Σ category_composite_c
```

**Judge weights (from the methodology):**

| Category | `w_c` |
|---|---:|
| General knowledge | 0.30 |
| Logical reasoning | 0.30 |
| Mathematical reasoning | 0.10 |
| Coding | 0.10 |
| Instruction following | 0.10 |
| Structured JSON | 0.10 |
| Summarization | 0.40 |

**Rule:** A deterministic failure is never converted into a pass by the judge. The judge may add partial credit only within the configured weight, and the deterministic result is always shown separately.

### 6.7 Judge reliability

- Judge parse failure rate.
- Judge retry rate.
- Judge score variance on the 10% triple-judged sample; these calls use temperature 0.7 and recorded seeds so variance is observable.
- Manual agreement: exact, within one point, and rank correlation where applicable.

These are reported as evaluator health metrics, not as model quality.

---

## 7. Hardware metrics

### 7.1 Sampling model

- Baseline window: 2 seconds before `t0` by default, configurable.
- Inference window: `t0` to `t_last`.
- Sampling interval: 1000 ms default, configurable 250-1000 ms.
- Each sample stores a UTC timestamp, phase, and availability flags.
- Peak and average are computed over the inference-window samples only.
- Baseline mean is computed over the baseline window.

### 7.2 CPU utilization

**Definition:** System-wide CPU utilization percentage.

**Source:** `psutil.cpu_percent(interval=None)` sampled after a warm-up call.

**Formula:**

```
cpu_mean = mean(cpu_percent_samples)
cpu_peak = max(cpu_percent_samples)
```

**Caveats:**

- System-wide, not Ollama-specific.
- On a 6-core/12-thread CPU, process CPU can exceed 100%; system CPU is 0–100%.
- Sampling can miss short spikes.

### 7.3 System RAM

**Definition:** Physical memory used and available.

**Source:** `psutil.virtual_memory()`.

**Formulas:**

```
ram_used_bytes = total − available
ram_percent = (used / total) × 100
ram_peak = max(ram_used_bytes_samples)
ram_mean = mean(ram_used_bytes_samples)
```

**Caveats:** "Available" is an OS estimate and may include reclaimable memory; values vary with Windows caching.

### 7.4 Ollama process RSS

**Definition:** Resident set size of Ollama-related processes.

**Source:** `psutil` process iteration, matching the Ollama server and runner process names.

**Formula:**

```
ollama_rss = Σ RSS of matching processes
```

**Caveats:**

- Approximate: may include server overhead and multiple processes.
- May not capture GPU-resident memory.
- Process discovery differs across Ollama versions; record the method and matched process names.
- If no process is found, store `null` and flag unavailable.

### 7.5 GPU utilization

**Definition:** GPU compute utilization percentage.

**Source:** NVML `nvmlDeviceGetUtilizationRates().gpu`; fallback to `nvidia-smi --query-gpu=utilization.gpu`.

**Formulas:**

```
gpu_util_mean = mean(samples)
gpu_util_peak = max(samples)
```

**Caveats:** Windows WDDM may report coarse or zero values; nvidia-smi may report N/A. Store null or approximate with a note.

### 7.6 GPU memory utilization

**Definition:** GPU memory controller utilization percentage.

**Source:** NVML `nvmlDeviceGetUtilizationRates().memory`.

**Caveats:** This is memory-controller utilization, not the percentage of VRAM used. Do not confuse it with `vram_percent`.

### 7.7 System VRAM used

**Definition:** Total GPU memory used by all processes.

**Source:** NVML `nvmlDeviceGetMemoryInfo().used`; fallback nvidia-smi.

**Formulas:**

```
vram_used_peak = max(samples)
vram_used_mean = mean(samples)
vram_total = NVML total
vram_percent = (vram_used / vram_total) × 100
baseline_vram_mean = mean(baseline samples)
delta_vram_peak = vram_used_peak − baseline_vram_mean
```

**Caveats:**

- System-wide; includes Windows desktop, browser, and other applications.
- Do not attribute all of it to the model.
- On WDDM, values may be approximate.

### 7.8 Ollama model VRAM

**Definition:** Ollama's reported `size_vram` for the loaded model.

**Source:** `GET /api/ps`.

**Use:** Model-level VRAM accounting, separate from NVML system totals.

**Caveats:** Ollama's accounting may differ from NVML; report both. `size_vram = 0` means no GPU residency; a value less than model `size` indicates partial offload. Phase 05 reads `/api/ps` after each monitored response and stores the matched model name, artifact size, `size_vram`, and offload ratio. A model that is not listed produces an explicit `ollama_model_vram_error` rather than an inferred value. Sampling at load, after warm-up, and at model-run end remains a later orchestration concern.

### 7.9 GPU offload ratio

**Definition:** Fraction of the model artifact reported as GPU-resident.

**Formula:**

```
offload_ratio = size_vram / size
```

**Caveats:** This is an artifact-size ratio, not a layer-count ratio. If `size_vram` is unavailable, the ratio is null.

### 7.10 GPU temperature

**Definition:** GPU temperature in degrees Celsius.

**Source:** NVML `nvmlDeviceGetTemperature`.

**Formulas:**

```
gpu_temp_peak = max(samples)
gpu_temp_mean = mean(samples)
```

**Caveats:** Laptop sensors may be unavailable or coarse. CPU temperature is not measured by default.

### 7.11 Peak and average rules

- Peak = maximum valid sample in the inference window.
- Average = arithmetic mean of valid samples in the inference window.
- If the inference window has fewer than two samples, report the values but flag `low_sample_count = true`.
- Baseline and inference windows are not pooled.
- Missing samples are skipped, not filled.

### 7.12 Resource efficiency indicators

These are derived indicators, not physical laws:

```
tokens_per_second_per_gb_vram = output_tps / (vram_used_peak / 1024^3)
output_tokens_per_peak_ram_mb = eval_count / (ram_peak / 1024^2)
throughput_per_cpu_percent = output_tps / max(cpu_mean, 1)
```

**Use:** Comparative indicators within the same hardware and run. Do not present them as universal efficiency ratings.

**Caveats:** System VRAM includes other processes; RAM is system-wide; CPU is system-wide.

### 7.13 Phase 05 implementation notes

- The sampler runs per measured request with a baseline window, an inference window from `t0` to `t_last`, and a final sample.
- Each `HardwareSample` stores UTC timestamp, elapsed seconds, phase, raw values, Ollama process names, GPU source, and sample overhead.
- `WindowSummary` stores `sample_count`, `low_sample_count`, `duration_seconds`, averages, and peaks. A window with fewer than two samples is flagged; its values remain descriptive only.
- `HardwareSummary` stores baseline, inference, and overall windows, raw samples, availability flags, system VRAM, Ollama model VRAM, GPU source/name, errors, and the attribution note.
- System-wide CPU, RAM, and GPU values include other processes. Ollama RSS is process-specific only when a matching process is found; Ollama model VRAM comes from `/api/ps`.
- NVML is preferred; `nvidia-smi` is the fallback. Unavailable values are null. Windows WDDM can make GPU utilization unavailable in some samples.
- Monitoring failures are stored in `hardware_error` and do not fail the inference request.
- Sampler overhead is stored as average and peak milliseconds per summary.
- The default interval is 1000 ms; the default baseline is 2.0 seconds. Both are recorded in the summary.

---

## 8. Recommendation methodology

Recommendations are comparative and workload-specific. They are not a claim that one model is universally best. The scoring is implemented in Phase 08 and must follow these definitions.

### 8.1 Dimensions

Five dimensions are scored on a 0–100 scale:

| Dimension | Meaning | Primary inputs |
|---|---|---|
| Quality | Task-relevant response quality | Category composites from deterministic plus judge scoring |
| Speed | Generation throughput | `output_tps`, `prompt_tps` |
| Responsiveness | User-perceived latency | Mean TTFT, mean total latency |
| Reliability | Ability to complete requests consistently | Success rate, timeout rate, consistency |
| Efficiency | Resource cost on this host | Peak system VRAM, peak system RAM, mean CPU |

### 8.2 Quality subscore

The quality subscore uses category composites, not raw item counts. Category composites are defined in section 6.6.

| Workload profile | Category weights for the quality subscore |
|---|---|
| `balanced` | Overall composite (equal category weights) |
| `interactive_chat` | General knowledge 0.40, instruction following 0.30, logical reasoning 0.30 |
| `coding_assistant` | Coding 0.60, structured JSON 0.20, instruction following 0.20 |
| `document_summarization` | Summarization 0.60, instruction following 0.20, general knowledge 0.20 |
| `constrained_hardware` | Overall composite |
| `batch_throughput` | Overall composite |

If a category is missing for a profile, the remaining category weights are renormalized and the missing coverage is disclosed.

### 8.3 Speed subscore

Normalize `output_tps` and `prompt_tps` across the compared models:

```
speed = 0.70 × norm_higher(output_tps) + 0.30 × norm_higher(prompt_tps)
```

`output_tps` is weighted more because it dominates user-perceived generation speed for this workload.

### 8.4 Responsiveness subscore

Normalize mean TTFT and mean total latency; lower is better:

```
responsiveness = 0.60 × norm_lower(mean_ttft) + 0.40 × norm_lower(mean_latency)
```

TTFT is weighted more because it dominates perceived responsiveness in interactive use.

### 8.5 Reliability subscore

Reliability uses absolute rates, not min-max normalization, because a 100% success rate should score 100 regardless of the other models.

```
reliability_base = 100 × success_rate − 20 × timeout_rate
reliability = clip(reliability_base, 0, 100)
```

If the consistency probe is available:

```
reliability_with_consistency = 0.70 × reliability_base + 0.30 × (100 × consistency)
reliability = clip(reliability_with_consistency, 0, 100)
```

If consistency is unavailable, `reliability_base` is used and `consistency_missing = true` is recorded. Timeouts are already excluded from success; the explicit penalty prevents a model with many timeouts from scoring highly merely because other failures are rare.

### 8.6 Efficiency subscore

Normalize peak system VRAM, peak system RAM, and mean CPU; lower is better:

```
efficiency = 0.50 × norm_lower(vram_used_peak)
           + 0.30 × norm_lower(ram_used_peak)
           + 0.20 × norm_lower(cpu_mean)
```

Notes:

- `vram_used_peak` is system-wide, not model-only. `delta_vram_peak` and Ollama `size_vram` are reported separately as sensitivity checks.
- If `vram_used_peak` is unavailable for any candidate, use `delta_vram_peak`; if that is also unavailable, drop the VRAM component and renormalize.
- CPU and RAM are system-wide and include the application overhead.

### 8.7 Normalization

For a metric where higher is better:

```
norm_higher(x) = 100 × (x − min) / (max − min)
```

For a metric where lower is better:

```
norm_lower(x) = 100 × (max − x) / (max − min)
```

Rules:

- If `max == min`, all candidates receive 50 and the dimension is flagged `no_discrimination`.
- Normalization is computed only across the models being compared in the same run and workload.
- Do not normalize across runs with different hardware, datasets, or settings.
- Absolute quality and reliability scores are not min-max normalized; they already use a meaningful 0–100 scale.
- With three candidates, the best and worst values receive 100 and 0 by construction. Normalized scores are relative ranking aids, not absolute measures, and a score of 0 does not mean the model is unusable.
- Min-max normalization is sensitive to outliers. Report raw values alongside normalized scores, use the specified robust aggregates, and flag any result driven by a single extreme observation.

### 8.8 Workload profiles and weights

Weights sum to 1.00 for every profile.

| Profile | Quality | Speed | Responsiveness | Reliability | Efficiency |
|---|---:|---:|---:|---:|---:|
| `balanced` | 0.35 | 0.20 | 0.15 | 0.15 | 0.15 |
| `interactive_chat` | 0.25 | 0.15 | 0.30 | 0.15 | 0.15 |
| `coding_assistant` | 0.45 | 0.15 | 0.10 | 0.20 | 0.10 |
| `document_summarization` | 0.40 | 0.20 | 0.10 | 0.15 | 0.15 |
| `constrained_hardware` | 0.25 | 0.10 | 0.10 | 0.20 | 0.35 |
| `batch_throughput` | 0.30 | 0.35 | 0.05 | 0.15 | 0.15 |

The weights are configuration data, not constants in code. Changing a weight creates a new recommendation configuration and is disclosed in the report.

### 8.9 Weighted score

```
weighted_score = Σ (weight_d × subscore_d)
```

Report the weighted score, every subscore, the weights used, and the sample sizes behind each input.

### 8.10 Missing metrics

- If a dimension is unavailable for one candidate, drop that dimension for all candidates and renormalize the remaining profile weights. This preserves comparability.
- If more than 30% of the original weight is dropped, mark the recommendation `insufficient_data` and do not present a confident rank.
- If a candidate has no data for a dimension because every request failed, do not impute a score; treat the dimension as missing and apply the reliability gate.
- Report the dropped dimensions and the renormalized weights.

### 8.11 Failed requests

- Failed and timed-out requests reduce the reliability subscore.
- Failed requests are excluded from latency, throughput, and hardware aggregates; they are never imputed as fast.
- A model with a failure rate above 20% on the relevant workload is not eligible for rank 1. It is still listed with a reliability warning.
- A model with a timeout rate above 10% receives an explicit warning even if its success rate remains acceptable.
- Cancellations are disclosed but are not counted as model failures.

### 8.12 Confidence

| Confidence | Conditions |
|---|---|
| High | At least 3 independent full runs; failure rate ≤ 5%; no missing dimensions; judge coverage ≥ 90% |
| Medium | 1–2 full runs, or failure rate 5–20%, or one dropped dimension, or judge coverage 70–90% |
| Low | Failure rate > 20%, or >30% missing weight, or judge coverage < 70%, or smoke-only data |

Judge coverage is the fraction of judge-eligible responses that produced a valid judge evaluation. Smoke runs never receive a recommendation. Confidence is reported next to the rank, not hidden in a tooltip.

### 8.13 Ties and stability

- A weighted-score difference of 2 points or less is reported as a practical tie.
- Report the score range across repeated runs when available.
- If the rank order changes across runs, report the instability instead of choosing one run.
- Do not claim statistical significance from three models and a small number of runs.
- Optional bootstrap confidence intervals are exploratory and must be labeled as such.

### 8.14 Recommendation record

Each stored recommendation includes:

- Experiment run ID.
- Workload profile.
- Model ID, tag, and digest.
- Rank.
- Weighted score.
- All dimension subscores.
- Weights used, including renormalization.
- Confidence label.
- Missing dimensions and failure rates.
- Human-readable rationale.
- Methodology version.

---

## 9. Aggregation and exclusion rules

### 9.1 Included

- Successful warm performance responses for latency/throughput aggregates.
- Successful quality responses for quality aggregates.
- Successful cold responses for cold-start aggregates.
- All measured attempts for reliability aggregates.
- All valid hardware samples in the inference window.

### 9.2 Excluded

- Warm-up requests.
- Judge calls from candidate performance metrics.
- Failed responses from latency/throughput aggregates.
- Timed-out responses from latency/throughput aggregates.
- Cancelled requests from reliability denominators, with the count disclosed.
- Smoke runs from final recommendations.

### 9.3 Outliers

- Outliers are retained in primary aggregates.
- Optional trimmed statistics (for example, 5% trim) may be reported as sensitivity analysis.
- Trimming must be labeled and must never replace the primary result.
- A removed outlier must have a documented reason and remain in the database.

### 9.4 Missing data

- A missing metric is `null`, never zero.
- Aggregates report valid `n`.
- If a model has no valid samples for a metric, it is excluded from that metric's comparison and the recommendation renormalizes weights.
- If more than 30% of a profile's weight is missing, the recommendation is marked "insufficient data."

---

## 10. Metric availability matrix

| Metric | Primary source | Usually available on this host | Notes |
|---|---|---|---|
| TTFT | Client | Yes | Streaming required |
| Total latency | Client | Yes | |
| Prompt/output token counts | Ollama | Yes | |
| Prompt/output durations | Ollama | Yes | |
| Load duration | Ollama | Usually | Zero if already loaded |
| CPU utilization | psutil | Yes | System-wide |
| System RAM | psutil | Yes | |
| Ollama RSS | psutil | Usually | Approximate |
| GPU utilization | NVML/nvidia-smi | Usually | WDDM caveat |
| GPU memory utilization | NVML | Usually | Not VRAM percent |
| System VRAM | NVML/nvidia-smi | Yes | System-wide |
| Model VRAM | Ollama `/api/ps` | Yes | Ollama accounting |
| GPU temperature | NVML | Usually | May be unavailable |
| CPU temperature | Not measured | No | Would require extra driver |
| Power draw | Not measured | No | Laptop sensor reliability uncertain |
| Judge scores | Ollama judge | Yes if judge installed | Subjective |
| Manual judge agreement | Human review | Sample only | 10% sample |

---

## 11. Formula summary

| Metric | Formula | Unit |
|---|---|---|
| TTFT | `(t_first − t0) × 1000` | ms |
| Total latency | `(t_last − t0) × 1000` | ms |
| Mean latency | `mean(L)` | ms |
| Median latency | `percentile(L, 50)` | ms |
| P95 latency | `percentile(L, 95)` | ms |
| Load duration | `load_duration_ns / 1e9` | s |
| Prompt TPS | `prompt_eval_count / (prompt_eval_duration_ns / 1e9)` | tokens/s |
| Effective prompt TPS | `prompt_eval_count / (TTFT_ms / 1000)` | tokens/s |
| Output TPS | `eval_count / (eval_duration_ns / 1e9)` | tokens/s |
| Effective output TPS | `eval_count / ((Latency_ms − TTFT_ms) / 1000)` | tokens/s |
| Success rate | `success / (attempted − cancelled)` | ratio |
| Failure rate | `failure / (attempted − cancelled)` | ratio |
| Timeout rate | `timeout / (attempted − cancelled)` | ratio |
| Cancellation rate | `cancelled / attempted` | ratio |
| Consistency | mean pairwise token F1 | 0–1 |
| Exact repeat rate | mean pairwise normalized equality | 0–1 |
| Correct refusal rate | `correct_refusals / unanswerable_items` | ratio |
| Category score | `Σ item_score / Σ max_score × 100` | 0–100 |
| Overall deterministic | equal-weight mean of category scores | 0–100 |
| Judge normalized | `((raw − 1) / 4) × 100` | 0–100 |
| Composite quality | weighted blend per category | 0–100 |
| Quality coverage | `evaluated_items / expected_items` | ratio |
| Failure-as-zero quality | quality with failed items scored 0 | 0–100 |
| VRAM percent | `vram_used / vram_total × 100` | % |
| Offload ratio | `size_vram / size` | 0–1 |
| Delta VRAM peak | `peak − baseline_mean` | bytes |

---

## 12. Reporting requirements

Every aggregate reported in the dashboard or an export must include:

- Metric name and definition version.
- Sample set (`warm_performance`, `warm_quality`, `cold_start`, etc.).
- Model tag and digest.
- Valid sample size `n`.
- Whether the value is client-measured, Ollama-reported, system-sampled, or derived.
- Availability or approximation flags.
- Any exclusions applied.

Every performance chart must show:

- Central tendency and dispersion, or the raw distribution.
- Sample size.
- A note that cold and warm results are not pooled.

Every quality chart must show:

- Deterministic and judge scores separately.
- Judge model tag.
- Missing judge coverage.
- Quality coverage, including failed items that were not scored.

Every hardware chart must show:

- Baseline and inference windows.
- System-wide versus Ollama-specific values.
- WDDM caveats where applicable.

---

## 13. Limitations

- Client metrics include localhost overhead.
- Ollama metrics exclude client overhead but may exclude scheduler queueing.
- Hardware sampling is periodic, not continuous.
- System-wide metrics cannot be attributed solely to the model.
- Ollama RSS is approximate.
- WDDM limits per-process GPU attribution.
- P95 is unstable with small samples.
- Lexical quality metrics do not understand paraphrase.
- Judge scores are subjective and potentially biased.
- The benchmark is a sample of behavior, not a complete capability assessment.
- No metric in this project justifies a claim of universal model superiority.
