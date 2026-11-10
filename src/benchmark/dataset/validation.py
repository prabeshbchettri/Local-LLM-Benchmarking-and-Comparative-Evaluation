"""
File: validation.py

Description:
Dataset and probe validation summaries for command-line and health-check use.

Responsibilities:
- Load and validate the quality dataset and both probes.
- Report category, difficulty, evaluation, and unanswerable distributions.
- Verify dataset hashes and consistency-probe references.
- Render a concise human-readable validation summary.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field

from benchmark.core.exceptions import DatasetError
from benchmark.dataset.loader import (
    load_consistency_probe,
    load_dataset,
    load_performance_probe,
    verify_probe_consistency,
)
from benchmark.dataset.models import compute_dataset_content_hash
from benchmark.models.base import BaseSchema


class DatasetValidationSummary(BaseSchema):
    """Validation results and distributions for the dataset and probes."""

    path: str
    valid: bool = False
    name: str | None = None
    version: str | None = None
    schema_version: str | None = None
    content_sha256: str | None = None
    computed_content_hash: str | None = None
    total_items: int = 0
    category_distribution: dict[str, int] = Field(default_factory=dict)
    difficulty_distribution: dict[str, int] = Field(default_factory=dict)
    evaluation_type_distribution: dict[str, int] = Field(default_factory=dict)
    unanswerable_count: int = 0
    unanswerable_distribution: dict[str, int] = Field(default_factory=dict)
    performance_prompts: int = 0
    consistency_items: int = 0
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def validate_dataset(
    path: Path,
    *,
    performance_probe_path: Path | None = None,
    consistency_probe_path: Path | None = None,
    verify_hash: bool = True,
) -> DatasetValidationSummary:
    """Validate a dataset and optional probe files."""
    summary = DatasetValidationSummary(path=str(Path(path).expanduser().resolve()))
    try:
        dataset = load_dataset(path, verify_hash=verify_hash)
    except DatasetError as exc:
        summary.errors.append(str(exc))
        return summary

    summary.name = dataset.name
    summary.version = dataset.version
    summary.schema_version = dataset.schema_version
    summary.content_sha256 = dataset.content_sha256
    summary.computed_content_hash = compute_dataset_content_hash(dataset.items)
    summary.total_items = len(dataset.items)
    summary.category_distribution = dataset.category_distribution()
    summary.difficulty_distribution = dataset.difficulty_distribution()
    summary.evaluation_type_distribution = dataset.evaluation_type_distribution()
    summary.unanswerable_distribution = dataset.unanswerable_distribution()
    summary.unanswerable_count = sum(summary.unanswerable_distribution.values())

    if summary.computed_content_hash != summary.content_sha256:
        summary.errors.append(
            "content hash mismatch: stored "
            f"{summary.content_sha256}, computed {summary.computed_content_hash}"
        )

    consistency_probe = None
    if performance_probe_path is not None:
        try:
            performance_probe = load_performance_probe(
                performance_probe_path,
                verify_hash=verify_hash,
            )
            summary.performance_prompts = len(performance_probe.prompts)
        except DatasetError as exc:
            summary.errors.append(str(exc))

    if consistency_probe_path is not None:
        try:
            consistency_probe = load_consistency_probe(
                consistency_probe_path,
                verify_hash=verify_hash,
            )
            summary.consistency_items = len(consistency_probe.items)
        except DatasetError as exc:
            summary.errors.append(str(exc))

    if consistency_probe is not None:
        try:
            verify_probe_consistency(dataset, consistency_probe)
        except DatasetError as exc:
            summary.errors.append(str(exc))

    summary.valid = not summary.errors
    return summary


def _format_distribution(distribution: dict[str, int]) -> str:
    return ", ".join(f"{name}={count}" for name, count in distribution.items())


def render_validation_summary(summary: DatasetValidationSummary) -> str:
    """Render a validation summary as plain text."""
    lines = [f"Dataset path: {summary.path}"]
    if summary.name:
        lines.append(
            f"Dataset: {summary.name} v{summary.version} (schema {summary.schema_version})"
        )
    if summary.content_sha256:
        lines.append(f"Stored SHA-256: {summary.content_sha256}")
    if summary.computed_content_hash:
        lines.append(f"Computed SHA-256: {summary.computed_content_hash}")
    lines.append(f"Items: {summary.total_items}")
    if summary.category_distribution:
        lines.append(f"Categories: {_format_distribution(summary.category_distribution)}")
    if summary.difficulty_distribution:
        lines.append(
            f"Difficulties: {_format_distribution(summary.difficulty_distribution)}"
        )
    if summary.evaluation_type_distribution:
        lines.append(
            "Evaluation types: "
            f"{_format_distribution(summary.evaluation_type_distribution)}"
        )
    if summary.unanswerable_distribution:
        lines.append(
            "Unanswerable: "
            f"{summary.unanswerable_count} "
            f"({_format_distribution(summary.unanswerable_distribution)})"
        )
    lines.append(f"Performance prompts: {summary.performance_prompts}")
    lines.append(f"Consistency items: {summary.consistency_items}")
    for warning in summary.warnings:
        lines.append(f"WARNING: {warning}")
    for error in summary.errors:
        lines.append(f"ERROR: {error}")
    lines.append(f"Status: {'VALID' if summary.valid else 'INVALID'}")
    return "\n".join(lines)
