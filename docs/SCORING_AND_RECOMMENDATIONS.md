# Scoring and Recommendations

**Status:** Phase 08 implemented  
**Methodology version:** `1.0.0`  
**Schema version:** `1.1.0`

This document is the canonical definition of the Phase 08 analytics, scoring, workload profiles, confidence rules, and recommendation behavior. Where an older design note disagrees with this document or the code, this document and `src/benchmark/analytics/` take precedence.

## 1. Evidence Source and Filters

All analytics are computed from stored experiment results:

- `responses` for status and response text.
- `performance_metrics` for TTFT, total latency, and token throughput.
- `evaluations` for deterministic, judge, and composite scores.
- `hardware_measurements` for CPU, RAM, GPU, system VRAM, and Ollama model VRAM.
- `runs` for inference profile and cold-start state.
- `models` for tags and display names.

The default analysis basis is:

- Stored responses only, never hard-coded metrics.
- Warm-up responses excluded.
- `smoke` runs excluded.
- Cold-start runs excluded.
- Missing measurements remain missing; they are not replaced with zero.
- An explicit `--run-profile` filter overrides the smoke exclusion.
- `--include-smoke` and `--include-cold-start` must be requested explicitly.

Stored, measured values are the only inputs. The analytics layer does not invent missing hardware, evaluation, or timing values.

## 2. Descriptive Statistics

For each model and each metric, the engine reports:

- count of finite values
- mean
- median
- minimum
- maximum
- population standard deviation (`ddof=0`)
- P95 with linear interpolation

Latency, TTFT, output tokens per second, and prompt tokens per second use successful responses only. Failed, timed-out, empty, and cancelled responses are never treated as fast measurements.

### 2.1 Outcome Rates

Let:

- `M` = measured responses
- `C` = cancelled responses
- `A = M - C` = attempted responses
- `S` = successful responses
- `F = A - S` = failed responses, including timeouts
- `T` = timed-out responses

Then:

```text
success_rate    = S / A
failure_rate    = F / A
timeout_rate    = T / A
cancellation_rate = C / M
```

Timeouts are included in the failure rate and are also reported separately. Cancellations are disclosed but excluded from the model's success/failure denominator.

### 2.2 Consistency

Response consistency is the mean pairwise text similarity across successful repetitions of the same question. Text is whitespace-normalized before comparison. A question measured once returns no consistency value. The value is `null` rather than an artificial 100.

## 3. Quality

Quality uses stored evaluation scores on the 0-100 scale. Deterministic, judge, and composite scores are kept separate.

A response is considered evaluated when at least one of its deterministic, judge, or composite scores exists. Judge coverage is the fraction of evaluated responses that have a judge score.

### 3.1 Category Scores

Scores are first aggregated by the seven dataset categories. For each category:

```text
category_score = mean of available response scores in that category
```

Failed responses with no evaluation are not assigned a quality score.

### 3.2 Primary Quality Score

The primary quality score is the equal-category mean:

```text
quality = mean(category composite scores with data)
```

This prevents a category with more prompts from dominating the result.

### 3.3 Sensitivity Scores

Two sensitivity values are reported:

```text
item_weighted_quality =
    mean(composite score of every evaluated response)

failure_as_zero_quality =
    sum(composite scores of scored successful responses)
    / (scored successful responses + failed responses)
```

The failure-as-zero value counts failed responses as zero and excludes successful but unevaluated responses from its denominator. It is explicitly a sensitivity check, not the primary score.

### 3.4 Coverage

```text
quality_coverage = evaluated responses / measured responses
judge_coverage   = evaluated responses with judge score / evaluated responses
```

Coverage and missing judge scores are always reported as limitations.

## 4. Resource Usage

Resource metrics are averaged over responses with a stored hardware summary:

- CPU average percent
- RAM average used MiB
- GPU utilization average percent
- system VRAM average used MiB
- Ollama model VRAM average MiB

Quality-per-resource values are computed only when both quality and the resource value are present and positive:

```text
quality_per_cpu_point       = quality / cpu_avg_percent
quality_per_ram_gib         = quality / ram_avg_gib
quality_per_vram_gib        = quality / system_vram_avg_gib
quality_per_model_vram_gib  = quality / model_vram_avg_gib
```

Unavailable measurements remain `null` and carry explicit availability flags.

## 5. Normalization

Relative dimensions use min-max scaling across the models being compared in the same experiment and profile.

Higher is better:

```text
score = 100 * (value - min) / (max - min)
```

Lower is better:

```text
score = 100 * (max - value) / (max - min)
```

Rules:

- If `max == min`, every model receives 50 and the dimension is marked `no_discrimination`.
- Missing values remain missing.
- Absolute quality and reliability are not min-max normalized; they already use a 0-100 scale.
- A relative dimension is dropped for every model if any compared model lacks the required measurement. This preserves comparability.
- Reliability requires a value for every compared model. A model with no attempted requests has no reliability score.
- The weighted average renormalizes over the dimensions that remain available.
- Quality is required for a composite score. A model with no quality score is excluded from recommendations.

## 6. Efficiency

Efficiency combines resource use after lower-is-better normalization:

```text
efficiency = mean(normalized CPU, normalized RAM, normalized VRAM)
```

Rules:

- Only resource components measured for every compared model are included.
- Ollama model VRAM is preferred when available for every model; otherwise system VRAM is used when available for every model.
- CPU and RAM are system-wide and include operating-system and application overhead.
- If no resource component is comparable across models, efficiency is unavailable.
- The result records which components were used.

## 7. Workload Profiles

Five profiles are configured in `configs/benchmark.yaml`. Weights are percentages that must total 100. Dimension weights are validated at load time.

| Profile | Quality | Latency | Throughput | Efficiency | Reliability |
|---|---:|---:|---:|---:|---:|
| `general_purpose` | 40 | 20 | 15 | 10 | 15 |
| `coding` | 45 | 15 | 15 | 10 | 15 |
| `reasoning` | 50 | 10 | 10 | 10 | 20 |
| `json_generation` | 50 | 10 | 20 | 5 | 15 |
| `resource_constrained` | 30 | 20 | 15 | 25 | 10 |

Optional category weights for profile-specific quality:

| Profile | Category weights |
|---|---|
| `general_purpose` | none; equal-category quality |
| `coding` | coding 0.70, instruction_following 0.20, logical_reasoning 0.10 |
| `reasoning` | logical_reasoning 0.45, mathematical_reasoning 0.35, general_knowledge 0.20 |
| `json_generation` | structured_json 0.80, instruction_following 0.20 |
| `resource_constrained` | none; equal-category quality |

If a required category is missing, the remaining category weights are renormalized and category coverage is disclosed. Configured category weights must total 1.0.

The configured default is `default_scoring_profile`, normally `general_purpose`.

## 8. Composite Score

The composite is the weighted mean of available dimensions:

```text
composite = sum(weight[d] * score[d]) / sum(weight[d])
```

Only available dimensions contribute to the denominator. Quality is mandatory; if quality is unavailable, the composite is `null`.

Latency uses P95 total latency. Throughput uses mean output tokens per second. Reliability uses the absolute success rate on a 0-100 scale.

## 9. Recommendations

A recommendation is comparative and workload-specific. It is not a claim that a model is universally best.

### 9.1 Eligibility

- At least two models must have a comparable composite score.
- Each eligible model must have evaluated responses and a profile-specific quality score.
- Smoke-only experiments cannot produce a recommendation.
- A model without evaluations is excluded and the reason is stored.
- If fewer than two models are comparable, the result is unavailable and explains why.

### 9.2 Failure Gate

A model with a failure rate above 20% cannot rank first:

```text
failure_rate > 0.20  ->  not eligible for first place
```

Gate-passing models are ranked before models above the gate. If every comparable model exceeds the gate, no recommendation is produced. Timeout rates above 10% produce an explicit warning.

### 9.3 Ties

Composite scores within 2.0 points are practical ties. Each recommendation stores:

- `tied = true`
- `tied_with = [other model tags within 2.0 points]`

Ranks remain unique (`1`, `2`, ...) because the database has a unique constraint on `(experiment_id, workload, rank)`. Near-ties are disclosed rather than assigned equal ranks.

### 9.4 Confidence

Confidence is an evidence label, not statistical significance.

| Label | Conditions |
|---|---|
| `high` | At least 3 compared models, at least 20 evaluated responses, all 5 dimensions, judge coverage >= 50%, quality coverage >= 80%, failure rate <= 10%, and no tie |
| `medium` | At least 2 compared models, at least 5 evaluated responses, at least 4 dimensions, and quality coverage >= 50% |
| `low` | All other cases, including small samples, missing evidence, or ties |

Confidence and limitations are stored with every recommendation.

### 9.5 Explanation

Each recommendation contains:

- workload, rank, model tag, and composite score
- confidence
- strength and weakness statements relative to the other measured models
- raw metrics and all five sub-scores
- missing dimensions
- evaluated-response and quality-coverage evidence
- failure and timeout rates
- rationale
- limitations
- tie information

## 10. Historical Comparison

Two or more experiments can be compared on the same scoring profile and filter basis. Per-model deltas are reported for:

- quality score
- P95 latency
- mean TTFT
- mean output tokens per second
- success, failure, and timeout rates
- CPU, RAM, GPU utilization, system VRAM, and model VRAM

Missing values are preserved. Percent change is omitted when the baseline is zero. Profile or filter mismatches are disclosed as limitations. A model present in only one experiment is reported as a limitation rather than imputed.

## 11. CLI

Compute analytics for one profile or all configured profiles:

```powershell
.\.venv\Scripts\python.exe -m benchmark analyze EXPERIMENT_ID
.\.venv\Scripts\python.exe -m benchmark analyze EXPERIMENT_ID --all-profiles --json
.\.venv\Scripts\python.exe -m benchmark analyze EXPERIMENT_ID --profile coding --persist
.\.venv\Scripts\python.exe -m benchmark analyze EXPERIMENT_ID --run-profile quality --include-cold-start
```

Compare experiments:

```powershell
.\.venv\Scripts\python.exe -m benchmark compare BASELINE_ID CANDIDATE_ID --json
.\.venv\Scripts\python.exe -m benchmark compare BASELINE_ID CANDIDATE_ID --profile reasoning
```

## 12. Persistence

Schema version `1.1.0` adds nullable `recommendations.details_json`. Existing databases migrate additively on open. The migration does not rewrite or delete historical rows.

The recommendation table stores the core ranking fields plus `details_json`, which contains:

- all sub-scores and metrics
- missing dimensions
- evidence counts
- quality and judge coverage
- failure and timeout rates
- tie information
- confidence and limitations
- methodology version

## 13. Manual Verification Example

The analytics test fixtures include four responses:

| Response | Category | Status | Composite |
|---|---|---|---|
| 1 | general_knowledge | success | 90 |
| 2 | general_knowledge | success | 70 |
| 3 | general_knowledge | error | none |
| 4 | coding | success | 50 |

Hand calculation:

```text
evaluated = 3
quality coverage = 3 / 4 = 0.75
general_knowledge category = (90 + 70) / 2 = 80
coding category = 50
primary quality = (80 + 50) / 2 = 65
item-weighted quality = (90 + 70 + 50) / 3 = 70
failure-as-zero quality = (90 + 70 + 50) / (3 + 1) = 52.5
```

The tests assert these values directly. They are synthetic verification data, not model benchmark results.

## 14. Limitations

- Min-max scores are relative to the compared models and are not absolute measurements.
- Three local models and a small number of repetitions cannot support statistical-significance claims.
- System-wide CPU, RAM, and GPU values include other processes.
- Model-specific VRAM is available only when Ollama reports it.
- Judge coverage can be zero when no judge was configured; deterministic and composite quality can still be reported.
- Historical comparisons are descriptive. They do not establish causation.
- Weights are configuration choices; changing them can change rankings.
