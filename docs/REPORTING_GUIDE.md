# Reporting Guide

Phase 10 turns a completed experiment in SQLite into reproducible CSV, JSON, HTML, and PDF reports. Every number is computed from stored rows by the analytics engine; no report value is hardcoded, and missing measurements are exported as missing rather than zero.

## Commands

```powershell
# All four formats into data/exports/<experiment_id>/
.\.venv\Scripts\python.exe -m benchmark report EXPERIMENT_ID

# Pick formats and an output directory
.\.venv\Scripts\python.exe -m benchmark report EXPERIMENT_ID --format csv --format json
.\.venv\Scripts\python.exe -m benchmark report EXPERIMENT_ID --output-dir data\exports\review

# Machine-readable summary for scripts and CI
.\.venv\Scripts\python.exe -m benchmark report EXPERIMENT_ID --json

# Standalone entry point (no package install required)
.\.venv\Scripts\python.exe scripts\generate_report.py EXPERIMENT_ID
```

`scripts\generate_report.py` and `benchmark report` call the same `generate_reports` service.

| Option | Meaning |
| --- | --- |
| `EXPERIMENT_ID` | Stored experiment to report. |
| `--database PATH` | Database override; defaults to `storage/benchmark.db`. |
| `--output-dir PATH` | Output directory; defaults to `data/exports/<experiment_id>`. |
| `--format {csv,json,html,pdf}` | Repeatable; defaults to all four. Duplicates and case are normalized. |
| `--profile NAME` | Scoring profile; defaults to the configured default profile. |
| `--run-profile NAME` | Include only stored runs with this inference profile. |
| `--include-cold-start` | Include cold-start runs; excluded by default. |
| `--include-smoke` | Include smoke runs; excluded by default. |
| `--allow-incomplete` | Export a non-completed experiment, flagged preliminary. |
| `--json` | Print the generation summary as JSON. |
| `--config-dir PATH` | Alternate configuration directory. |

Exit codes: `0` on success, `1` for a reporting or configuration error, `130` for an interrupt. Errors go to stderr with an `ERROR:` prefix.

## Refusing incomplete experiments

Only experiments whose stored status is `completed` are exported by default. A pending, running, or failed experiment is refused with a message explaining that `--allow-incomplete` is required. When `--allow-incomplete` is used, the bundle is marked preliminary and every export carries that flag:

- `experiment_summary.csv` has `preliminary=True`.
- JSON has `"preliminary": true`.
- HTML and PDF render a `PRELIMINARY DATA` banner.
- `manifest.json` records `"preliminary": true`.

Preliminary exports are for inspection only and must not be presented as final benchmark results.

## Output layout

```
data/exports/<experiment_id>/
├── experiment_summary.csv
├── model_comparison.csv
├── performance_metrics.csv
├── hardware_metrics.csv
├── evaluation_results.csv
├── responses.csv
├── <experiment_id>.json
├── <experiment_id>.html
├── <experiment_id>.pdf
└── manifest.json
```

Selecting a subset of formats writes only those files plus `manifest.json`.

## CSV exports

All CSVs are UTF-8, use `\n` line endings, and preserve full Unicode response text. Missing values are empty cells, never `0`. Units are encoded in header names (`_seconds`, `_mib`, `_pct`, `_c_`, `0_100`, `0_1`) so headers are stable even when a table has no rows.

| File | Contents |
| --- | --- |
| `experiment_summary.csv` | One row: experiment identity, status, preliminary flag, dataset name/version/hash/schema/item count, config hash, timestamps, counts, profile, and reported model tags. |
| `model_comparison.csv` | One row per model: response counts and success/failure/timeout/cancellation percentages, latency mean/median/p95, TTFT, token rates, quality/deterministic/judge/reliability/composite scores, coverage percentages, resource averages, and response consistency. |
| `performance_metrics.csv` | One row per reported response: model, question, category, repetition, run profile, cold-start flag, status, latency breakdown, token counts and rates, stream chunks. |
| `hardware_metrics.csv` | One row per response with a hardware summary: sample counts, CPU/RAM/GPU/VRAM averages and peaks, GPU temperature, Ollama process usage, model VRAM, GPU identity, availability flags, and sampler errors. Rows are absent when no hardware summary was stored. |
| `evaluation_results.csv` | One row per reported response: evaluation type and status, deterministic score/max/pass, judge status/model/raw/normalized/missing, composite score, check and error counts. |
| `responses.csv` | One row per reported response: prompt, system prompt, full response text, timestamps, latency, TTFT, token counts, composite score, and structured error fields. |

## JSON export

`<experiment_id>.json` is versioned by `REPORT_SCHEMA_VERSION` (`1.0.0`) and contains:

- `schema_version`, `generated_at`, `preliminary`
- `experiment` identity, status, timestamps, counts, notes
- `dataset` name, version, content SHA-256, schema version, item count, source path, random seed
- `configuration` config hash, stored config, host, software, report filters
- `model_metadata` model tags, digests, size, family, parameters, quantization, context length, capabilities, license, Ollama version, and raw model info
- `runs`, `responses`, `performance_metrics`, `hardware_measurements`, `evaluation_results`, `evaluation_breakdown`
- `analytics` (score profile, per-model statistics, quality, resources, limitations)
- `recommendations` (workload decisions, evidence confidence, methodology version)
- `errors` and deduplicated `limitations`

Missing numeric measurements are `null`. Byte values are accompanied by MiB convenience fields where useful.

## HTML and PDF

Both contain the same ten sections, in order:

1. Executive summary
2. Experimental environment
3. Methodology
4. Model metadata
5. Performance results
6. Quality results
7. Hardware analysis
8. Reliability analysis
9. Recommendations
10. Limitations

Two appendices appear when applicable: structured errors and truncated response excerpts (full text remains in the CSV and JSON exports). The HTML report is self-contained with inline SVG charts and no external assets. All stored text is HTML-escaped; only generated chart SVG is marked safe. The PDF is generated with ReportLab; Unicode body text uses an available system font and falls back to Helvetica with a recorded warning.

## Manifest and integrity

`manifest.json` records:

- schema version, experiment id and status, preliminary flag, profile
- generation timestamp, report filters (`run_profile`, `include_cold_start`, `include_smoke`)
- PDF page count and PDF warnings
- every file's name, format, byte size, and SHA-256

Verify an export with:

```powershell
Get-FileHash data\exports\EXPERIMENT_ID\model_comparison.csv -Algorithm SHA256
```

The digest must match the corresponding manifest entry. Reports are deterministic for the same database rows, configuration, filters, and generation inputs; `generated_at` is the one intentionally changing field.

## Dashboard

The dashboard includes a **Reports** page. Select an experiment, choose the formats, and select **Prepare export files** to build the downloads in memory through the same service. Completed experiments show run, response, evaluation, dataset, and hash counts; non-completed experiments are refused by the panel instead of being presented as final results. Preparing exports writes nothing to the database.

## Missing data and limitations

- A response row exists only when a response was stored; failed and timed-out responses are included with their status and error fields.
- Hardware rows exist only when a hardware summary was stored.
- Evaluation columns are blank/null when no evaluation exists; judge columns are blank/null when the judge was not eligible or failed.
- Recommendations and analytics limitations propagate to the JSON, HTML, PDF, and manifest warnings. They are never replaced with an optimistic fallback.

## Tests

`tests/reporting/` covers CSV headers, units, row counts, Unicode, missing-value blanks, JSON structure and nulls, HTML sections and escaping, PDF readability and section presence, manifest hashes, format selection, preliminary handling, refusal of incomplete/missing experiments, and both the CLI and standalone script. The dashboard Reports page is covered by `tests/dashboard/test_dashboard_app.py`.

Run only the reporting tests with:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\reporting -q
```

## Related documentation

- `docs/DEVELOPMENT_ROADMAP.md` - phase scope and acceptance criteria
- `docs/SCORING_AND_RECOMMENDATIONS.md` - score definitions and recommendation rules
- `docs/EVALUATION_METHODOLOGY.md` - deterministic checks and judge behavior
- `docs/DATABASE_SCHEMA.md` - stored tables and columns
