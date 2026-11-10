# Dataset Guide — benchmark-v1.0.0

> **Document purpose:** Describe the files, schema, distributions, hashing rules, validation workflow, and change-control rules for the Phase 03 benchmark dataset and probes.

| Field | Value |
|---|---|
| Dataset | `benchmark-v1.0.0` |
| Dataset version | `1.0.0` |
| Item schema version | `1.0.0` |
| Random seed | `42` |
| Status | Phase 03 implemented (pending approval) |
| Date | 2026-11-10 |

---

## 1. Files

| File | Purpose |
|---|---|
| `data/datasets/benchmark_v1.json` | Canonical 100-item dataset; source of truth |
| `data/datasets/benchmark_v1.0.0.jsonl` | Deterministic JSONL compatibility export |
| `data/datasets/benchmark_v1.0.0.manifest.json` | Sidecar metadata for the JSONL export |
| `data/datasets/probes/performance_v1.0.0.json` | Nine fixed performance prompts |
| `data/datasets/probes/consistency_v1.0.0.json` | Ten stratified consistency items |
| `data/datasets/schemas/benchmark_item.schema.json` | Generated Draft 2020-12 item JSON Schema |

The JSON dataset is authoritative. The JSONL file is generated from it and validated through the same Pydantic models.

### 1.1 Recorded hashes

| Artifact | SHA-256 |
|---|---|
| Dataset content | `ceafe917235213c710da9e3b6a1ceecb86ba519b214581f683191c51358e46f1` |
| Performance probe content | `ab2a2e522bb521458b43acb25545765fdb5988c7a7dea2a86155b6bc9d579099` |
| Consistency probe content | `0fc14378424790c81fddc5fc192560425d1f8c9a931da72ff5805a0973c2108c` |

---

## 2. Distributions

### 2.1 Categories and difficulty

| Category | ID prefix | Items | Easy | Medium | Hard | Evaluation types |
|---|---|---:|---:|---:|---:|---|
| General knowledge | `gk` | 15 | 5 | 7 | 3 | 13 exact match, 2 refusal |
| Logical reasoning | `lr` | 15 | 4 | 7 | 4 | 14 exact match, 1 refusal |
| Mathematical reasoning | `mr` | 15 | 5 | 7 | 3 | 14 numeric, 1 refusal |
| Coding | `code` | 15 | 4 | 7 | 4 | 15 code tests |
| Instruction following | `if` | 15 | 5 | 7 | 3 | 14 instruction constraints, 1 refusal |
| Structured JSON | `json` | 15 | 5 | 7 | 3 | 15 JSON Schema |
| Summarization | `sum` | 10 | 2 | 5 | 3 | 9 reference overlap, 1 refusal |
| **Total** | | **100** | **30** | **47** | **23** | |

Evaluation-type totals: `exact_match` 27, `numeric` 14, `code_tests` 15, `instruction_constraints` 14, `json_schema` 15, `reference_overlap` 9, `refusal_check` 6, `composite` 0.

### 2.2 Unanswerable items

| ID | Category | Scenario |
|---|---|---|
| `gk-008` | General knowledge | Population of the fictional city of Eldoria in Valoria |
| `gk-014` | General knowledge | Exact historical population of the fictional village of Veyra |
| `lr-014` | Logical reasoning | Insufficient premises to decide whether the pet is a cat |
| `mr-014` | Mathematical reasoning | Rectangle area given without enough information to determine length |
| `if-013` | Instruction following | Exact population of the fictional city of Noria in exactly five words |
| `sum-009` | Summarization | Findings from a nonexistent 2026 trial of the fictional drug Neuroclear |

All unanswerable items use `refusal_check`, set `is_unanswerable: true`, include a `reference_answer`, and list `forbidden_claims`.

---

## 3. Item schema

The generated JSON Schema is at `data/datasets/schemas/benchmark_item.schema.json`. The Pydantic model in `src/benchmark/dataset/models.py` is authoritative.

Core fields:

- `id`: stable identifier with a category prefix, e.g. `gk-001`.
- `category`: one of the seven categories.
- `difficulty`: `easy`, `medium`, or `hard`.
- `prompt`: exact user prompt; minimum 20 characters.
- `system_instruction`: optional system message.
- `expected_answer`: canonical answer for exact-match and numeric items.
- `evaluation_type`: deterministic evaluator identifier.
- `reference_answer`: reference used by code, JSON, overlap, refusal, and judge evaluation.
- `test_cases`: authored unit tests for coding items.
- `json_schema`: Draft 2020-12 schema for structured JSON items.
- `required_constraints`: deterministic instruction constraints.
- `maximum_output_tokens`: per-item generation cap, default 512.
- `tags`, `is_unanswerable`, `judge_eligible`, `max_score`, `source`, `notes`.
- `accepted_answers`, `key_facts`, `forbidden_claims`.
- `function_name`, `function_signature`.
- `numeric_tolerance`, `expected_unit`.

Method-specific requirements are enforced by `DatasetItem`:

- `exact_match` requires `expected_answer`.
- `numeric` requires a numeric `expected_answer`; optional `numeric_tolerance` and `expected_unit`.
- `code_tests` requires `function_name`, `function_signature`, `test_cases`, and `reference_answer`.
- `json_schema` requires `json_schema` and a `reference_answer` that satisfies it.
- `instruction_constraints` requires `required_constraints`.
- `reference_overlap` requires `reference_answer` and `key_facts`.
- `refusal_check` requires `reference_answer`, `forbidden_claims`, and `is_unanswerable: true`.
- `composite` requires `reference_answer`.

The model derives evaluator configuration through `DatasetItem.evaluation_config()`. There is no stored `evaluation_config` field.

### 3.1 Instruction constraints

Supported constraint types:

- `word_count`, `sentence_count`, `bullet_count`, `line_count`
- `required_phrase`, `forbidden_phrase`, `occurrence_count`
- `starts_with`, `ends_with`
- `lowercase`
- `each_line_starts_with`, `each_line_ends_with`, `each_line_contains`
- `contains_all`, `ordered_terms`, `exact_text`

Constraint text preserves leading and trailing whitespace. For example, `each_line_starts_with` can require `"- "` rather than only `"-"`.

---

## 4. Hashing and ordering

- The dataset hash covers only item payloads, not container metadata such as `created_at`.
- Items are sorted by `id` before hashing.
- Each item is serialized with sorted keys, compact separators, UTF-8 text, and no ASCII escaping.
- Top-level `null` fields are omitted with `exclude_none=True`.
- A code test's `expected` value is always retained, including `null`, so `None` remains hash-covered.
- The hash is the lowercase SHA-256 hex digest of the canonical JSON array of item payloads.
- The dataset `random_seed` is fixed at `42`; `BenchmarkDataset.ordered_items()` returns a deterministic seeded shuffle.
- The JSONL export is sorted by `id` and uses the same canonical item payloads.

Any item content change changes the hash and requires regenerating the JSONL export, manifest, and documentation references.

---

## 5. Loading and validation

Python API:

```python
from pathlib import Path

from benchmark.dataset import load_dataset, load_consistency_probe, load_performance_probe

dataset = load_dataset(Path("data/datasets/benchmark_v1.json"))
performance = load_performance_probe(Path("data/datasets/probes/performance_v1.0.0.json"))
consistency = load_consistency_probe(Path("data/datasets/probes/consistency_v1.0.0.json"))
```

`load_dataset()` supports `.json` and `.jsonl`. JSONL loading reads the sidecar `*.manifest.json` file, filters manifest-only fields such as `item_count` and `format`, and validates the combined payload.

Validation CLI:

```powershell
.\.venv\Scripts\python.exe scripts\validate_dataset.py
.\.venv\Scripts\python.exe scripts\validate_dataset.py --json
.\.venv\Scripts\python.exe scripts\validate_dataset.py --no-probes
.\.venv\Scripts\python.exe scripts\validate_dataset.py --dataset data\datasets\benchmark_v1.0.0.jsonl
```

The CLI validates:

- dataset counts, IDs, categories, difficulties, and unanswerable placement
- dataset and probe hashes
- item-defined JSON Schemas
- reference JSON documents against their schemas
- consistency-probe IDs, categories, and prompts against the dataset

A valid run prints `Status: VALID` and exits `0`; an invalid run prints errors and exits `1`.

---

## 6. JSONL export and manifest

- Each non-empty line is one compact JSON item.
- Items are sorted by `id`.
- The sidecar manifest stores `name`, `version`, `schema_version`, `created_at`, `random_seed`, `description`, `content_sha256`, `item_count`, and `format`.
- The loader ignores manifest-only fields that are not part of `BenchmarkDataset`.
- JSONL and JSON loaders produce the same validated dataset and content hash.

---

## 7. Probes

### 7.1 Performance probe

- Nine prompts: `P1`–`P3` short, `P4`–`P6` medium, `P7`–`P9` long.
- Five repeats per prompt.
- Target output lengths: 60 words for short prompts, 120 for medium, 200 for long.
- Maximum output tokens: 512.
- Prompt lengths: 18–19 words short, 101–107 words medium, 419–456 words long.
- Used only for latency, throughput, and output-length measurement; never for quality scoring.

### 7.2 Consistency probe

- Ten dataset items covering all seven categories.
- Selected IDs: `gk-002`, `gk-009`, `lr-005`, `lr-010`, `mr-006`, `code-009`, `if-006`, `json-011`, `sum-004`, `sum-010`.
- Three repeats per item.
- Sampling: temperature 0.7, top-p 0.9, top-k 40, seed 42, repeat penalty 1.1, context 4096, maximum output 512.
- Prompts must match the dataset exactly.

---

## 8. Authoring and review rules

- All items are original and authored for this project.
- Prompts are self-contained and do not require browsing, tools, or external files.
- Coding prompts state the function name and signature; tests cover normal, edge, and boundary cases.
- JSON prompts state the required structure; every reference document validates against its schema.
- Summarization prompts include the source text and length requirement.
- Answer leakage is avoided.
- Unanswerable prompts require an explicit refusal and prohibit fabricated claims.
- Manual spot-checks covered 62 of 100 items across all seven categories.
- Review corrections: removed a duplicate accepted answer in `gk-009`; preserved the required `"- "` prefix in `if-002` by keeping significant whitespace in constraint text.

---

## 9. Change control

- A content change requires a new dataset version and a new hash.
- Do not edit `benchmark_v1.json` without regenerating the JSONL export, manifest, generated schema, and documentation hash references.
- Do not change probes without a new probe version and hash.
- Any change to prompts, answers, constraints, schemas, or evaluation configuration invalidates prior results.
