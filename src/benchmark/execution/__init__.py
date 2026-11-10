"""
File: __init__.py

Description:
Execution services for Ollama access, model metadata, and lifecycle management.

Responsibilities:
- Export the public execution API.
"""

from benchmark.execution.metadata import (
    CompatibilityAssessment,
    LoadedModelSnapshot,
    ModelInspection,
    ModelMetadata,
    ModelRole,
    ModelSnapshot,
    ModelSource,
    ModelStatus,
    ModelVerification,
    assess_compatibility,
    build_loaded_snapshot,
    build_missing_metadata,
    build_model_metadata,
    build_verification,
    estimate_kv_cache_bytes,
    format_bytes,
    format_pull_event,
    read_snapshot,
    write_snapshot,
)
from benchmark.execution.ollama_client import OllamaClient, PullResult
from benchmark.execution.registry import ModelManager, RegisteredModel, normalize_tag

__all__ = [
    "CompatibilityAssessment",
    "LoadedModelSnapshot",
    "ModelInspection",
    "ModelManager",
    "ModelMetadata",
    "ModelRole",
    "ModelSnapshot",
    "ModelSource",
    "ModelStatus",
    "ModelVerification",
    "OllamaClient",
    "PullResult",
    "RegisteredModel",
    "assess_compatibility",
    "build_loaded_snapshot",
    "build_missing_metadata",
    "build_model_metadata",
    "build_verification",
    "estimate_kv_cache_bytes",
    "format_bytes",
    "format_pull_event",
    "normalize_tag",
    "read_snapshot",
    "write_snapshot",
]
