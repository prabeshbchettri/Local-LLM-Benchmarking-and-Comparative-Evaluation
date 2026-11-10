"""
File: test_dataset_loader.py

Description:
Unit tests for dataset and probe loading, validation, hashing, and JSONL export.

Responsibilities:
- Verify the canonical dataset and both probes load and validate.
- Verify content hashes, duplicate IDs, and item-defined schemas are enforced.
- Verify JSONL round trips and generated schema artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmark.dataset.loader import (
    export_jsonl,
    load_consistency_probe,
    load_dataset,
    load_performance_probe,
    verify_probe_consistency,
)
from benchmark.dataset.models import Category, EvaluationType
from benchmark.dataset.validation import validate_dataset
from benchmark.core.exceptions import DatasetValidationError

pytestmark = pytest.mark.unit


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def test_loads_canonical_dataset(dataset_path: Path) -> None:
    dataset = load_dataset(dataset_path)
    assert dataset.name == "benchmark-v1.0.0"
    assert dataset.version == "1.0.0"
    assert len(dataset.items) == 100
    assert len(dataset.content_sha256) == 64
    assert dataset.category_distribution() == {
        "general_knowledge": 15,
        "logical_reasoning": 15,
        "mathematical_reasoning": 15,
        "coding": 15,
        "instruction_following": 15,
        "structured_json": 15,
        "summarization": 10,
    }
    assert dataset.difficulty_distribution() == {"easy": 30, "medium": 47, "hard": 23}
    assert dataset.evaluation_type_distribution()["composite"] == 0
    assert sum(dataset.unanswerable_distribution().values()) == 6


def test_validation_summary_includes_probes(
    dataset_path: Path,
    performance_probe_path: Path,
    consistency_probe_path: Path,
) -> None:
    summary = validate_dataset(
        dataset_path,
        performance_probe_path=performance_probe_path,
        consistency_probe_path=consistency_probe_path,
    )
    assert summary.valid
    assert summary.total_items == 100
    assert summary.performance_prompts == 9
    assert summary.consistency_items == 10
    assert summary.errors == []


def test_validation_summary_reports_missing_dataset(tmp_path: Path) -> None:
    summary = validate_dataset(tmp_path / "missing.json")
    assert not summary.valid
    assert summary.errors


def test_hash_mismatch_is_rejected(dataset_path: Path, tmp_path: Path) -> None:
    payload = _read_json(dataset_path)
    payload["items"][0]["expected_answer"] = "Lyon"
    modified = tmp_path / "benchmark_v1.json"
    _write_json(modified, payload)
    with pytest.raises(DatasetValidationError, match="hash mismatch"):
        load_dataset(modified)


def test_duplicate_ids_are_rejected(dataset_path: Path, tmp_path: Path) -> None:
    payload = _read_json(dataset_path)
    payload["items"][1]["id"] = payload["items"][0]["id"]
    modified = tmp_path / "benchmark_v1.json"
    _write_json(modified, payload)
    with pytest.raises(DatasetValidationError, match="unique"):
        load_dataset(modified, verify_hash=False)


def test_invalid_item_json_schema_is_rejected(dataset_path: Path, tmp_path: Path) -> None:
    payload = _read_json(dataset_path)
    json_item = next(item for item in payload["items"] if item["id"] == "json-001")
    json_item["json_schema"] = {"type": 123}
    modified = tmp_path / "benchmark_v1.json"
    _write_json(modified, payload)
    with pytest.raises(DatasetValidationError, match="invalid JSON Schema"):
        load_dataset(modified, verify_hash=False)


def test_invalid_reference_json_is_rejected(dataset_path: Path, tmp_path: Path) -> None:
    payload = _read_json(dataset_path)
    json_item = next(item for item in payload["items"] if item["id"] == "json-001")
    json_item["reference_answer"] = "not-json"
    modified = tmp_path / "benchmark_v1.json"
    _write_json(modified, payload)
    with pytest.raises(DatasetValidationError, match="not valid JSON"):
        load_dataset(modified, verify_hash=False)


def test_jsonl_export_round_trip(dataset_path: Path, tmp_path: Path) -> None:
    dataset = load_dataset(dataset_path)
    jsonl_path, manifest_path = export_jsonl(dataset, tmp_path / "benchmark_v1.0.0.jsonl")
    reloaded = load_dataset(jsonl_path)
    assert reloaded.content_sha256 == dataset.content_sha256
    assert [item.id for item in reloaded.items] == sorted(
        item.id for item in dataset.items
    )
    manifest = _read_json(manifest_path)
    assert manifest["item_count"] == 100
    assert manifest["format"] == "jsonl"
    assert manifest["content_sha256"] == dataset.content_sha256


def test_loads_performance_probe(performance_probe_path: Path) -> None:
    probe = load_performance_probe(performance_probe_path)
    assert len(probe.prompts) == 9
    assert probe.repeats == 5
    assert {prompt.group.value for prompt in probe.prompts} == {"short", "medium", "long"}
    assert all(
        sum(1 for prompt in probe.prompts if prompt.group.value == group) == 3
        for group in ("short", "medium", "long")
    )


def test_loads_consistency_probe_and_verifies_references(
    dataset_path: Path,
    consistency_probe_path: Path,
) -> None:
    dataset = load_dataset(dataset_path)
    probe = load_consistency_probe(consistency_probe_path)
    verify_probe_consistency(dataset, probe)
    assert len(probe.items) == 10
    assert {item.category for item in probe.items} == set(Category)
    assert probe.repeats == 3


def test_performance_probe_hash_mismatch_is_rejected(
    performance_probe_path: Path,
    tmp_path: Path,
) -> None:
    payload = _read_json(performance_probe_path)
    payload["prompts"][0]["prompt"] = "A different prompt that is long enough to be valid."
    modified = tmp_path / "performance_v1.0.0.json"
    _write_json(modified, payload)
    with pytest.raises(DatasetValidationError, match="hash mismatch"):
        load_performance_probe(modified)


def test_consistency_prompt_mismatch_is_rejected(
    dataset_path: Path,
    consistency_probe_path: Path,
    tmp_path: Path,
) -> None:
    dataset = load_dataset(dataset_path)
    payload = _read_json(consistency_probe_path)
    payload["items"][0]["prompt"] = "A mismatched prompt that is long enough to be valid."
    modified = tmp_path / "consistency_v1.0.0.json"
    _write_json(modified, payload)
    probe = load_consistency_probe(modified, verify_hash=False)
    with pytest.raises(DatasetValidationError, match="prompt mismatch"):
        verify_probe_consistency(dataset, probe)


def test_generated_item_schema_artifact(dataset_dir: Path) -> None:
    schema_path = dataset_dir / "schemas" / "benchmark_item.schema.json"
    schema = _read_json(schema_path)
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["$id"].endswith("benchmark_item.schema.json")
    assert len(schema["properties"]) == 25
    assert "evaluation_type" in schema["properties"]


def test_ordered_items_are_deterministic(dataset_path: Path) -> None:
    dataset = load_dataset(dataset_path)
    first = [item.id for item in dataset.ordered_items()]
    second = [item.id for item in dataset.ordered_items()]
    assert first == second
    assert sorted(first) == sorted(item.id for item in dataset.items)


def test_every_item_has_a_primary_evaluation_method(dataset_path: Path) -> None:
    dataset = load_dataset(dataset_path)
    assert all(
        item.evaluation_type is not EvaluationType.COMPOSITE for item in dataset.items
    )
    assert all(item.evaluation_type for item in dataset.items)
