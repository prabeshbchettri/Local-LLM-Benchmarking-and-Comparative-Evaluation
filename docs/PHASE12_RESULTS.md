# Phase 12 Results - Final Experiments and Documentation

Date: 2026-11-14  
Status: Complete, with the reduced workload explicitly approved by the user.  
Database: `storage/phase12/phase12.db` (gitignored)  
Reports: `data/exports/phase12-main/`, `data/exports/phase12-coldstart/`, `data/exports/phase12-consistency/`

This document records what was measured, how it was measured, what the results were, and what the results do not support. Every value below comes from stored rows or generated reports; no result was inferred or fabricated.

---

## 1. Scope and deviation from the planned workload

The original plan targeted 900 measured responses: 3 models x 100 items x 3 full repetitions, plus separate cold-start and consistency probes. The user approved a reduced scope of **300 measured responses** for the full benchmark plus separate subsets. The completed workload was:

| Workload | Experiment ID | Models | Items | Repeats | Warm-up | Measured | Evaluation records |
|---|---|---|---:|---:|---:|---:|---:|
| Full quality benchmark | `20261114T074040Z-9edf76` | 3 | 100 | 1 | 1/model | 300 | 300 |
| Cold-start subset | `20261114T095758Z-c3ae3a` | 3 | 5 | 3 | 0 | 45 | 45 |
| Consistency subset | `20261114T102018Z-e114a1` | 3 | 10 | 3 | 0 | 90 | 90 |
| **Total** | | | | | | **435** | **435** |

The full run contains 303 response rows: 300 measured responses plus 3 warm-up responses. Warm-ups are stored but excluded from analytics.

Not executed: the 9-prompt x 5-repeat performance probe, the planned 3 full repetitions, a multi-hour soak test, and a multi-repetition full-dataset reliability run. These omissions are limitations, not hidden results.

The main run's first Qwen segment used the pre-fix streaming client and produced one 30-minute timeout. The remainder of the main run and both subsets used the fixed client. The fix is described in section 8.

---

## 2. Environment

| Item | Value |
|---|---|
| Operating system | Windows 11 (`Windows-11-10.0.26300-SP0`), PowerShell 5.1 |
| CPU | AMD Ryzen 5 5600H, 6 physical cores / 12 logical threads |
| RAM | 14,877,257,728 bytes (16 GB class) |
| GPU | NVIDIA GeForce GTX 1650, 4,096 MiB VRAM |
| GPU driver | 555.99 |
| Python | 3.13.5 (`.venv`) |
| Ollama | 0.40.16 at `http://127.0.0.1:11434` |
| Free disk at final health check | 160.4 GiB |
| Dataset | `benchmark-v1.0.0`, 100 items, SHA-256 `ceafe917235213c710da9e3b6a1ceecb86ba519b214581f683191c51358e46f1`, VALID |
| Configuration hash | `34c5793a1e4075824beb5db8f207dd02ca641bd9d1705f49b53ff6f2e1b039ac` |
| Benchmark temperature | 0.0 |
| Context length / max output | 4096 / 512 tokens |
| Cooldown between models | 60 seconds |

### Verified models

| Role | Tag | Digest | Size | Parameters | Quantization |
|---|---|---|---:|---|---|
| Candidate | `llama3.1:8b` | `46e0c10c039e019119339687c3c1757cc81b9da49709a3b3924863ba87ca666e` | 4,920,753,328 B | 8.0B | Q4_K_M |
| Candidate | `qwen2.5:7b-instruct` | `845dbda0ea48ed749caafd9e6037047aa19acfcfd82e704d7ca97d631a0b697e` | 4,683,087,332 B | 7.6B | Q4_K_M |
| Candidate | `mistral:7b-instruct-v0.3-q4_K_M` | `6577803aa9a036369e481d648a2baebb381ebc6e897f2bb9a766a2aa7bfbc1cf` | 4,372,824,384 B | 7.2B | Q4_K_M |
| Judge | `qwen2.5:3b` | `357c53fb659c5076de1d65ccb0b397446227b71a42be9d1603d46168015c9e4b` | 1,929,912,432 B | 3.1B | Q4_K_M |

`benchmark models verify` reported all four configured models as verified. `benchmark health` passed with no warnings and confirmed all four models installed.

---

## 3. Main quality benchmark

Experiment: `20261114T074040Z-9edf76` (`phase12-full-100x1`)  
Run profile: `quality`; 100 items per model; one measured repetition; one warm-up per model.  
Started 2026-11-14T07:40:40Z, completed 2026-11-14T09:57:35Z (2 h 16 min 55 s). The elapsed time includes the 30-minute Qwen timeout.

### 3.1 Performance, reliability, and resources

| Model | Success | Mean latency | Median | P95 | Mean TTFT | Output tok/s | CPU avg | RAM avg | System VRAM avg | Model VRAM avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `llama3.1:8b` | 100.0% | 19.31 s | 13.82 s | 59.16 s | 1.40 s | 8.00 | 63.5% | 13,392 MiB | 2,476 MiB | 2,255 MiB |
| `qwen2.5:7b-instruct` | 99.0% | 15.21 s | 8.96 s | 51.95 s | 1.21 s | 9.46 | 54.4% | 11,842 MiB | 2,418 MiB | 2,206 MiB |
| `mistral:7b-instruct-v0.3-q4_K_M` | 100.0% | 12.55 s | 9.00 s | 37.24 s | 1.20 s | 9.76 | 55.8% | 12,307 MiB | 2,445 MiB | 2,223 MiB |

Peak hardware values observed during the main run:

| Model | CPU peak | RAM peak | GPU utilization peak | VRAM peak | GPU temperature peak |
|---|---:|---:|---:|---:|---:|
| `llama3.1:8b` | 100.0% | 13.53 GiB | 97.0% | 2.42 GiB | 59 C |
| `qwen2.5:7b-instruct` | 100.0% | 13.83 GiB | 98.0% | 2.37 GiB | 62 C |
| `mistral:7b-instruct-v0.3-q4_K_M` | 87.3% | 12.37 GiB | 98.0% | 2.39 GiB | 63 C |

Qwen's one failure was a timeout on `lr-007`. It is excluded from latency and throughput statistics and counted as a failure in the 99.0% success rate.

### 3.2 Quality

| Model | Deterministic score | Judge score | Quality score | Item-weighted | Failure-as-zero | Evaluated | Judge coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| `llama3.1:8b` | 46.58 | 84.03 | 49.79 | 49.69 | 49.69 | 100/100 | 9/100 |
| `qwen2.5:7b-instruct` | 50.02 | 88.19 | 53.47 | 54.02 | 53.48 | 99/100 | 9/99 |
| `mistral:7b-instruct-v0.3-q4_K_M` | 45.75 | 91.67 | 49.44 | 49.22 | 49.22 | 100/100 | 9/100 |

The quality scores are close: Qwen leads by 4.0 points over Mistral and 3.7 points over Llama. The judge covers only the nine summarization responses per model. Deterministic exact-match checks passed 0 of 27 for every model, so the quality scores are conservative and should not be read as fine-grained quality differences. See section 7.

### 3.3 Category results

Values are the stored category composite scores (0-100). For summarization, the value combines deterministic overlap and the judge; other categories use deterministic evaluators only.

| Category | Llama | Qwen | Mistral | Interpretation |
|---|---:|---:|---:|---|
| General knowledge | 13.3 | 13.3 | 13.3 | Exact-match evaluator passed 2/15 for every model; not discriminative. |
| Logical reasoning | 0.0 | 0.0 | 0.0 | Exact-match evaluator passed 0/15 for every model; not discriminative. |
| Mathematical reasoning | 93.3 | 93.3 | 93.3 | Numeric evaluator passed 13/14 for every model. |
| Coding | 48.3 | **60.0** | 40.0 | Qwen passed 9/15 code tests; Llama 7/15; Mistral 6/15. |
| Instruction following | **68.3** | 64.4 | 65.6 | All models passed 6/14 constraint sets; partial scores differ. |
| Structured JSON | 73.3 | **90.0** | 80.0 | Qwen passed 13/15 schemas; Mistral 11/15; Llama 10/15. |
| Summarization | 51.9 | 53.2 | **53.8** | Judge + reference overlap; differences are small and judge-limited. |

The identical general-knowledge, logical-reasoning, and mathematical-reasoning scores are an evaluator limitation, not evidence that the models are equally capable. The coding, instruction-following, and JSON checks do discriminate.

### 3.4 Recommendations

The recommendation engine uses the configured `general_purpose` weights: quality 40%, latency 20%, throughput 15%, efficiency 10%, reliability 15%. A model with a failure rate above 20% cannot rank first.

| Workload | Rank 1 | Rank 2 | Rank 3 |
|---|---|---|---|
| General purpose | Mistral 77.13 | Qwen 65.29 | Llama 34.92 |
| Coding | Mistral 70.85 | Qwen 66.96 | Llama 36.38 |
| Reasoning | Mistral 65.02 | Qwen 59.07 | Llama 37.67 |
| JSON generation | Mistral 87.23 | Qwen 82.21 | Llama 51.17 |
| Resource constrained | Mistral 78.22 | Qwen 69.99 | Llama 24.94 |

Mistral ranks first in every configured profile because its latency, throughput, and efficiency advantages outweigh small quality differences. This does not mean Mistral wins every quality category: Qwen leads the deterministic coding and structured-JSON checks, and Llama leads instruction following. The recommendation is a weighted engineering choice, not a universal quality claim. Confidence is `medium` for every result.

---

## 4. Cold-start subset

Experiment: `20261114T095758Z-c3ae3a` (`phase12-coldstart-5x3`)  
Items: `gk-002`, `lr-001`, `mr-001`, `code-001`, `json-001`; 3 repetitions; model unloaded before each measured request; no warm-up.  
Measured: 45 responses; 45 successful; 45 evaluations. Duration: 22 min 06 s.

| Model | Success | Mean latency | Median | P95 | Mean TTFT | Output tok/s | CPU avg | RAM avg | System VRAM avg | Deterministic score |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `llama3.1:8b` | 100.0% | 31.59 s | 28.46 s | 46.35 s | 14.87 s | 8.92 | 40.6% | 10,517 MiB | 1,878 MiB | 60.0 |
| `qwen2.5:7b-instruct` | 100.0% | 23.04 s | 21.80 s | 32.40 s | 14.85 s | 10.30 | 32.1% | 9,990 MiB | 1,772 MiB | 60.0 |
| `mistral:7b-instruct-v0.3-q4_K_M` | 100.0% | 17.61 s | 17.21 s | 24.46 s | 11.34 s | 10.79 | 32.9% | 9,747 MiB | 1,835 MiB | 40.0 |

Cold-start overhead relative to the same five items in the warm main run:

| Model | Warm mean | Cold mean | Ratio | Warm TTFT | Cold TTFT | TTFT overhead |
|---|---:|---:|---:|---:|---:|---:|
| `llama3.1:8b` | 19.62 s | 31.59 s | 1.61x | 1.32 s | 14.87 s | +13.55 s |
| `qwen2.5:7b-instruct` | 10.12 s | 23.04 s | 2.28x | 1.34 s | 14.85 s | +13.52 s |
| `mistral:7b-instruct-v0.3-q4_K_M` | 7.12 s | 17.61 s | 2.47x | 1.10 s | 11.34 s | +10.24 s |

The cold-start deterministic scores cover only five items and no judge calls; they are not comparable to the 100-item quality scores. Cold-start general-purpose ranking: Mistral 73.67, Qwen 71.78, Llama 39.00.

---

## 5. Consistency subset

Experiment: `20261114T102018Z-e114a1` (`phase12-consistency-10x3`)  
Items: the ten `consistency-v1.0.0` probe items; 3 repetitions; warm model; no warm-up.  
Measured: 90 responses; 90 successful; 90 evaluations. Duration: 26 min 23 s.

| Model | Success | Mean latency | Median | P95 | Mean TTFT | Output tok/s | Response consistency | Quality score | Composite (general) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `llama3.1:8b` | 100.0% | 13.21 s | 10.70 s | 39.05 s | 0.61 s | 8.50 | **0.944** | 50.3 | 36.12 |
| `qwen2.5:7b-instruct` | 100.0% | 13.14 s | 9.35 s | 38.04 s | 0.56 s | 9.07 | 0.894 | 50.2 | 56.27 |
| `mistral:7b-instruct-v0.3-q4_K_M` | 100.0% | 11.69 s | 9.22 s | 33.93 s | 0.56 s | 9.31 | 0.780 | 44.2 | 75.56 |

Response consistency is the mean pairwise text similarity across repeated identical prompts. The run used the global benchmark temperature of **0.0**, not the 0.7 value recorded in the consistency probe file, because the experiment CLI does not load per-probe generation overrides. Therefore this measurement reflects determinism and runtime stability at greedy decoding, not robustness under sampled output. Mistral produced the most textual variation across identical prompts; Llama produced the least. Consistency is not a quality score, and Mistral still ranks first on the general-purpose composite because of speed and efficiency.

---

## 6. Hardware observations

- All three models were partially offloaded: system VRAM stayed near 2.4 GiB on a 4 GiB GTX 1650, while CPU averages were 54-64% in the main run and peaks reached 100%.
- GPU utilization averages were about 25% while peaks reached 97-98%, consistent with short GPU bursts and CPU-side work.
- Main-run RAM averages were 11.8-13.4 GiB; peaks were 12.4-13.8 GiB on a 16 GB machine.
- GPU temperatures stayed at 59-63 C peak. No thermal throttling was observed in the stored samples, but the machine was not under a multi-hour soak.
- Hardware values are system-wide. They include the operating system and other processes and are not exclusively attributable to a model. Model VRAM is Ollama's own accounting.

---

## 7. Evaluation and judge validation

### 7.1 Deterministic evaluators

| Evaluation type | Llama | Qwen | Mistral |
|---|---:|---:|---:|
| Code tests | 7/15 | **9/15** | 6/15 |
| Exact match | 0/27 | 0/27 | 0/27 |
| Instruction constraints | 6/14 | 6/14 | 6/14 |
| JSON schema | 10/15 | **13/15** | 11/15 |
| Numeric | 13/14 | 13/14 | 13/14 |
| Reference overlap | 0/9 | 0/9 | 0/9 |
| Refusal check | **5/6** | 4/6 | 4/6 |

The exact-match and reference-overlap evaluators are deliberately strict. They penalize correct paraphrases. For example, Llama's `sum-005` response was a correct two-sentence summary of the water cycle, but key-fact lexical matching found 0 of 5 facts and the deterministic overlap score was 14.94/100. This is why quality scores are low in absolute terms and why category-level general-knowledge and logical-reasoning scores should not be used to rank the models.

### 7.2 Judge sample

- The judge ran on 27 responses: 9 summarization responses per model.
- Judge coverage is 9% of measured responses per model.
- Judge mean normalized score: 87.96; deterministic overlap mean: 20.91; Pearson correlation: 0.172; mean gap: +67.05 points.
- Manual spot-check of ten judged responses found:
  - One clear false negative: Llama `sum-005` received 18.75/100 for a correct two-sentence summary. The judge's relevance rationale claimed the response did not answer the request, but it did.
  - One borderline low completeness score: Qwen `sum-001` received completeness 3 despite covering the main facts.
  - The other eight cases were broadly plausible; no clear false positive was found.
- The judge shares the Qwen family with the Qwen candidate. It is not neutral, and the 3B judge is small.

The judge scores are useful as a second signal, not as ground truth. The quality ranking (Qwen 53.47, Llama 49.79, Mistral 49.44) is close and judge-limited. The performance ranking is much more stable.

---

## 8. Defect found and fixed

The main Qwen run recorded a timeout on `lr-007` after **1,839 seconds** even though `request_timeout_seconds` is 300. The cause was that Ollama sent blank keep-alive lines while the request was stalled. `httpx` resets its per-read timeout when any line arrives, and the engine only checked its wall-clock deadline when a parsed chunk was yielded. Blank lines were skipped by the NDJSON parser, so the deadline was never enforced.

Fix in `src/benchmark/execution/ollama_client.py`:

- Added `_deadline_lines`, which checks the wall-clock deadline before every line, including blank keep-alive lines.
- `stream_chat` now wraps `response.iter_lines()` with that guard.
- Added regression test `test_blank_keepalive_lines_respect_total_deadline` in `tests/unit/test_ollama_client.py`.

The timeout is still a real failure and remains in the main experiment's reliability count. After the fix, the cold-start and consistency subsets produced zero timeouts.

---

## 9. Reproducibility

Environment and model checks:

```powershell
.\.venv\Scripts\python.exe -m benchmark health
.\.venv\Scripts\python.exe -m benchmark models verify
.\.venv\Scripts\python.exe scripts\validate_dataset.py
```

Main quality run:

```powershell
.\.venv\Scripts\python.exe scripts\run_experiment.py `
  --database storage\phase12\phase12.db `
  --dataset data\datasets\benchmark_v1.json `
  --models llama3.1:8b qwen2.5:7b-instruct mistral:7b-instruct-v0.3-q4_K_M `
  --profile quality --repeats 1 --warmup 1 --cooldown 60 `
  --name phase12-full-100x1
```

Cold-start run:

```powershell
.\.venv\Scripts\python.exe scripts\run_experiment.py `
  --database storage\phase12\phase12.db `
  --dataset data\datasets\benchmark_v1.json `
  --models llama3.1:8b qwen2.5:7b-instruct mistral:7b-instruct-v0.3-q4_K_M `
  --profile cold_start --cold-start --warmup 0 --repeats 3 --cooldown 60 `
  --items gk-002 lr-001 mr-001 code-001 json-001 `
  --name phase12-coldstart-5x3
```

Consistency run:

```powershell
.\.venv\Scripts\python.exe scripts\run_experiment.py `
  --database storage\phase12\phase12.db `
  --dataset data\datasets\benchmark_v1.json `
  --models llama3.1:8b qwen2.5:7b-instruct mistral:7b-instruct-v0.3-q4_K_M `
  --profile consistency --warmup 0 --repeats 3 --cooldown 60 `
  --items gk-002 gk-009 lr-005 lr-010 mr-006 code-009 if-006 json-011 sum-004 sum-010 `
  --name phase12-consistency-10x3
```

Analytics and reports:

```powershell
.\.venv\Scripts\python.exe -m benchmark analyze 20261114T074040Z-9edf76 --database storage\phase12\phase12.db --all-profiles
.\.venv\Scripts\python.exe scripts\generate_report.py 20261114T074040Z-9edf76 --database storage\phase12\phase12.db --output-dir data\exports\phase12-main
```

The `analyze` and `compare` commands now accept `--database`, removing the earlier need to copy configuration files or databases.

---

## 10. Artifacts

| Artifact | Path |
|---|---|
| Main database | `storage/phase12/phase12.db` |
| Main report bundle | `data/exports/phase12-main/` (6 CSV, JSON, HTML, PDF, manifest; PDF 9 pages) |
| Cold-start report bundle | `data/exports/phase12-coldstart/` (6 CSV, JSON, HTML, PDF, manifest; PDF 9 pages) |
| Consistency report bundle | `data/exports/phase12-consistency/` (6 CSV, JSON, HTML, PDF, manifest; PDF 9 pages) |
| Test report | `docs/TESTING_REPORT.md` |
| This report | `docs/PHASE12_RESULTS.md` |

All report bundles include `manifest.json` with byte sizes and SHA-256 hashes.

### Validation summary

| Check | Result |
|---|---|
| `python -m compileall -q src tests scripts` | PASS |
| `python -m pytest -q` | 330 passed, 3 skipped in 83.12 s |
| Live Ollama tests | 3 passed in 19.25 s |
| `benchmark health` | PASS; all four configured models installed |
| `benchmark models verify` | PASS; all four verified |
| Dataset validation | VALID; 100 items, probes valid |
| SQLite integrity / foreign keys | `ok` / no rows |
| Database totals | 3 experiments, 9 runs, 438 responses, 435 evaluations, 0 errors |
| Report bundles | 10 files each; 9 artifacts plus manifest; PDFs 9 pages |
| Dashboard | `/_stcore/health` 200 `ok`; root page 200 |

---

## 11. Limitations and interpretation

1. **Reduced workload.** The full benchmark has one repetition per item, not three. Run-to-run variance is not estimated.
2. **Single machine.** Results apply to this Windows laptop, GTX 1650, 16 GB RAM, and Ollama 0.40.16. They do not generalize to other hardware or Ollama versions.
3. **Quality uncertainty.** Exact-match and reference-overlap evaluators penalize correct paraphrase. The judge covers only nine summarization responses per model and produced at least one clear false negative. Quality differences are small and low-confidence.
4. **Judge bias.** The judge is a 3B Qwen model and shares a family with the Qwen candidate. It is not a neutral human panel.
5. **Consistency temperature.** The consistency run used temperature 0.0, not the probe's 0.7. It measures determinism/runtime stability, not sampled-output robustness.
6. **Small subsets.** Cold-start uses 5 items x 3 repeats; consistency uses 10 items x 3 repeats. These are indicative, not statistically robust.
7. **No performance probe.** The 9-prompt x 5-repeat performance probe was not executed. Latency statistics come from the 100-item quality pass and the two subsets.
8. **No long soak.** No multi-hour memory/thermal soak was performed.
9. **System-wide hardware.** CPU, RAM, GPU, and VRAM values include other processes and are not exclusively attributable to a model.
10. **Timeout history.** The main Qwen run's 30-minute timeout occurred before the keep-alive deadline fix. The timeout is counted as a failure, not hidden.
11. **No lint/type/coverage tooling.** Validation used byte-compilation, the test suite, live Ollama tests, health checks, dataset validation, and database integrity checks.
12. **Sandbox.** Generated code runs in a restricted subprocess, not a security boundary against adversarial code.

**Bottom line:** Mistral 7B Instruct v0.3 is the best balanced choice on this machine for general-purpose, resource-constrained, and throughput-sensitive use. Qwen 2.5 7B Instruct is the strongest candidate on the discriminating deterministic coding and structured-JSON checks and is a reasonable quality-first alternative. Llama 3.1 8B Instruct is the slowest and least efficient here, despite good instruction-following and consistency. These are engineering recommendations with medium confidence, not universal model rankings.
