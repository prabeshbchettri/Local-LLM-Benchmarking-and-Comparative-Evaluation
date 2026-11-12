"""
File: loader.py

Description:
Loading, hash verification, JSON Schema validation, and JSONL export for dataset files.

Responsibilities:
- Load canonical JSON datasets and line-delimited JSON compatibility exports.
- Load and verify performance and consistency probes.
- Validate item-defined JSON Schemas and reference JSON documents.
- Export deterministic JSONL files, manifests, and item JSON Schema artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError
from pydantic import ValidationError

from benchmark.core.exceptions import DatasetError, DatasetValidationError
from benchmark.dataset.models import (
    BenchmarkDataset,
    ConsistencyProbe,
    DatasetItem,
    PerformanceProbe,
    compute_consistency_probe_hash,
    compute_dataset_content_hash,
    compute_performance_probe_hash,
    item_payload,
)


def _read_text(path: Path) -> str:
    if not path.is_file():
        raise DatasetError(f"Dataset file not found: {path}")
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise DatasetError(f"Could not read dataset file {path}: {exc}") from exc


def _read_json_document(path: Path) -> Any:
    text = _read_text(path)
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise DatasetValidationError(
            f"Invalid JSON in {path}: line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc


def _manifest_path(path: Path) -> Path:
    return path.with_suffix(".manifest.json")


def _read_jsonl_document(path: Path, manifest_path: Path | None = None) -> dict[str, Any]:
    text = _read_text(path)
    items: list[Any] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            items.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise DatasetValidationError(
                f"Invalid JSONL in {path} at line {line_number}: {exc.msg}"
            ) from exc
    resolved_manifest = manifest_path or _manifest_path(path)
    manifest = _read_json_document(resolved_manifest)
    if not isinstance(manifest, dict):
        raise DatasetValidationError(
            f"Dataset manifest must be an object: {resolved_manifest}"
        )
    payload = {
        field: manifest[field]
        for field in BenchmarkDataset.model_fields
        if field != "items" and field in manifest
    }
    payload["items"] = items
    return payload


def _validate_item_definitions(dataset: BenchmarkDataset) -> None:
    for item in dataset.items:
        if item.json_schema is None:
            continue
        try:
            Draft202012Validator.check_schema(item.json_schema)
        except SchemaError as exc:
            raise DatasetValidationError(
                f"Item {item.id} has an invalid JSON Schema: {exc.message}"
            ) from exc
        validator = Draft202012Validator(item.json_schema)
        if item.reference_answer is None:
            continue
        try:
            reference = json.loads(item.reference_answer)
        except json.JSONDecodeError as exc:
            raise DatasetValidationError(
                f"Item {item.id} reference_answer is not valid JSON: {exc.msg}"
            ) from exc
        errors = sorted(validator.iter_errors(reference), key=lambda error: list(error.path))
        if errors:
            detail = errors[0]
            location = "/".join(str(part) for part in detail.path) or "<root>"
            raise DatasetValidationError(
                f"Item {item.id} reference_answer does not satisfy its JSON Schema at "
                f"{location}: {detail.message}"
            )


def load_dataset(
    path: Path,
    *,
    verify_hash: bool = True,
    validate_schemas: bool = True,
) -> BenchmarkDataset:
    """Load and validate a canonical JSON or JSONL dataset file."""
    resolved = Path(path).expanduser().resolve()
    if resolved.suffix.lower() == ".json":
        payload = _read_json_document(resolved)
    elif resolved.suffix.lower() == ".jsonl":
        payload = _read_jsonl_document(resolved)
    else:
        raise DatasetValidationError(
            f"Unsupported dataset format {resolved.suffix!r}; expected .json or .jsonl"
        )
    if not isinstance(payload, dict):
        raise DatasetValidationError(f"Dataset document must be an object: {resolved}")
    try:
        dataset = BenchmarkDataset.model_validate(payload)
    except ValidationError as exc:
        raise DatasetValidationError(
            f"Dataset validation failed for {resolved}: {exc}"
        ) from exc
    if verify_hash:
        computed = compute_dataset_content_hash(dataset.items)
        if computed != dataset.content_sha256:
            raise DatasetValidationError(
                f"Dataset hash mismatch for {resolved}: expected "
                f"{dataset.content_sha256}, computed {computed}"
            )
    if validate_schemas:
        _validate_item_definitions(dataset)
    return dataset


def load_performance_probe(path: Path, *, verify_hash: bool = True) -> PerformanceProbe:
    """Load and validate a performance probe file."""
    resolved = Path(path).expanduser().resolve()
    payload = _read_json_document(resolved)
    if not isinstance(payload, dict):
        raise DatasetValidationError(f"Performance probe must be an object: {resolved}")
    try:
        probe = PerformanceProbe.model_validate(payload)
    except ValidationError as exc:
        raise DatasetValidationError(
            f"Performance probe validation failed for {resolved}: {exc}"
        ) from exc
    if verify_hash:
        computed = compute_performance_probe_hash(probe.prompts)
        if computed != probe.content_sha256:
            raise DatasetValidationError(
                f"Performance probe hash mismatch for {resolved}: expected "
                f"{probe.content_sha256}, computed {computed}"
            )
    return probe


def load_consistency_probe(path: Path, *, verify_hash: bool = True) -> ConsistencyProbe:
    """Load and validate a consistency probe file."""
    resolved = Path(path).expanduser().resolve()
    payload = _read_json_document(resolved)
    if not isinstance(payload, dict):
        raise DatasetValidationError(f"Consistency probe must be an object: {resolved}")
    try:
        probe = ConsistencyProbe.model_validate(payload)
    except ValidationError as exc:
        raise DatasetValidationError(
            f"Consistency probe validation failed for {resolved}: {exc}"
        ) from exc
    if verify_hash:
        computed = compute_consistency_probe_hash(probe.items)
        if computed != probe.content_sha256:
            raise DatasetValidationError(
                f"Consistency probe hash mismatch for {resolved}: expected "
                f"{probe.content_sha256}, computed {computed}"
            )
    return probe


def ordered_items(dataset: BenchmarkDataset, seed: int | None = None) -> list[DatasetItem]:
    """Return the dataset in deterministic seeded order."""
    return dataset.ordered_items(seed)


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8", newline="\n")
        temporary.replace(path)
    except OSError as exc:
        raise DatasetError(f"Could not write {path}: {exc}") from exc
    finally:
        if temporary.exists():
            temporary.unlink(missing_ok=True)


def _dataset_manifest(dataset: BenchmarkDataset) -> dict[str, Any]:
    return {
        "name": dataset.name,
        "version": dataset.version,
        "schema_version": dataset.schema_version,
        "created_at": dataset.created_at.isoformat().replace("+00:00", "Z"),
        "random_seed": dataset.random_seed,
        "description": dataset.description,
        "content_sha256": dataset.content_sha256,
        "item_count": len(dataset.items),
        "format": "jsonl",
    }


def export_jsonl(
    dataset: BenchmarkDataset,
    path: Path,
    manifest_path: Path | None = None,
) -> tuple[Path, Path]:
    """Write a deterministic JSONL export and its version manifest."""
    resolved = Path(path).expanduser().resolve()
    manifest = Path(manifest_path).expanduser().resolve() if manifest_path else _manifest_path(resolved)
    lines = [
        json.dumps(
            item_payload(item),
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        for item in sorted(dataset.items, key=lambda item: item.id)
    ]
    _write_text_atomic(resolved, "\n".join(lines) + "\n")
    manifest_text = json.dumps(
        _dataset_manifest(dataset),
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
    )
    _write_text_atomic(manifest, manifest_text + "\n")
    return resolved, manifest


def build_item_json_schema() -> dict[str, Any]:
    """Build the JSON Schema for one dataset item from the Pydantic model."""
    schema = DatasetItem.model_json_schema()
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://local-llm-benchmark.local/schemas/benchmark_item.schema.json",
        "title": "Benchmark dataset item",
        "description": "One validated item in benchmark-v1.0.0.",
        **schema,
    }
    return schema


def write_item_json_schema(path: Path) -> Path:
    """Write the generated item JSON Schema to disk."""
    resolved = Path(path).expanduser().resolve()
    text = json.dumps(build_item_json_schema(), indent=2, ensure_ascii=False)
    _write_text_atomic(resolved, text + "\n")
    return resolved


def verify_probe_consistency(dataset: BenchmarkDataset, probe: ConsistencyProbe) -> None:
    """Verify that consistency probe items exist and match the dataset prompts."""
    items_by_id = {item.id: item for item in dataset.items}
    for probe_item in probe.items:
        dataset_item = items_by_id.get(probe_item.item_id)
        if dataset_item is None:
            raise DatasetValidationError(
                f"Consistency probe references unknown item {probe_item.item_id}"
            )
        if probe_item.category is not dataset_item.category:
            raise DatasetValidationError(
                f"Consistency probe category mismatch for {probe_item.item_id}"
            )
        if probe_item.prompt != dataset_item.prompt:
            raise DatasetValidationError(
                f"Consistency probe prompt mismatch for {probe_item.item_id}"
            )
