"""
File: __init__.py

Description:
Dataset loading, validation, hashing, and probe models.

Responsibilities:
- Export the public dataset API.
"""

from benchmark.dataset.loader import (
    build_item_json_schema,
    export_jsonl,
    load_consistency_probe,
    load_dataset,
    load_performance_probe,
    ordered_items,
    verify_probe_consistency,
    write_item_json_schema,
)
from benchmark.dataset.models import (
    BenchmarkDataset,
    Category,
    CodeTestCase,
    ConsistencyProbe,
    ConsistencyProbeItem,
    DatasetItem,
    Difficulty,
    EvaluationType,
    InstructionConstraint,
    PerformanceProbe,
    PerformancePrompt,
    ProbeGroup,
    compute_consistency_probe_hash,
    compute_dataset_content_hash,
    compute_performance_probe_hash,
)
from benchmark.dataset.validation import (
    DatasetValidationSummary,
    render_validation_summary,
    validate_dataset,
)

__all__ = [
    "BenchmarkDataset",
    "Category",
    "CodeTestCase",
    "ConsistencyProbe",
    "ConsistencyProbeItem",
    "DatasetItem",
    "DatasetValidationSummary",
    "Difficulty",
    "EvaluationType",
    "InstructionConstraint",
    "PerformanceProbe",
    "PerformancePrompt",
    "ProbeGroup",
    "build_item_json_schema",
    "compute_consistency_probe_hash",
    "compute_dataset_content_hash",
    "compute_performance_probe_hash",
    "export_jsonl",
    "load_consistency_probe",
    "load_dataset",
    "load_performance_probe",
    "ordered_items",
    "render_validation_summary",
    "validate_dataset",
    "verify_probe_consistency",
    "write_item_json_schema",
]
