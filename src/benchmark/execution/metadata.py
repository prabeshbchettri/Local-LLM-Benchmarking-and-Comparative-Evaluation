"""
File: metadata.py

Description:
Model metadata, verification, compatibility, and snapshot models.

Responsibilities:
- Convert Ollama API responses into stable model metadata.
- Compare installed digests and quantization with configuration.
- Estimate memory fit against the project's target hardware.
- Write metadata snapshots as atomic JSON files.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from benchmark.core.config import ModelSpec
from benchmark.models.base import BaseSchema
from benchmark.models.ollama import (
    OllamaLoadedModel,
    OllamaModelSummary,
    OllamaPullEvent,
    OllamaShowResponse,
)

GIB = 1024**3
TARGET_VRAM_BYTES = 4 * GIB
TARGET_RAM_BYTES = 16 * GIB
PARAMETER_PATTERN = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*([KMBT])\s*$", re.IGNORECASE)
PARAMETER_MULTIPLIERS = {
    "K": 1_000,
    "M": 1_000_000,
    "B": 1_000_000_000,
    "T": 1_000_000_000_000,
}


class ModelRole(StrEnum):
    """Role of a model in the benchmark roster."""

    CANDIDATE = "candidate"
    JUDGE = "judge"
    OTHER = "other"


class ModelSource(StrEnum):
    """How a model entered the local Ollama store."""

    INSTALLED = "installed"
    PULLED = "pulled"
    MISSING = "missing"


class ModelMetadata(BaseSchema):
    """Reproducible metadata for one model artifact."""

    tag: str
    name: str
    role: ModelRole
    source: ModelSource = ModelSource.INSTALLED
    installed: bool = True
    digest: str | None = None
    size_bytes: int | None = None
    family: str | None = None
    parameter_count: int | None = None
    parameter_size: str | None = None
    quantization: str | None = None
    context_length: int | None = None
    embedding_length: int | None = None
    capabilities: list[str] = Field(default_factory=list)
    license: str | None = None
    modified_at: str | None = None
    ollama_version: str
    model_info: dict[str, Any] = Field(default_factory=dict)


class LoadedModelSnapshot(BaseSchema):
    """A model currently loaded in Ollama memory."""

    tag: str
    name: str
    digest: str | None = None
    size_bytes: int | None = None
    size_vram_bytes: int | None = None
    context_length: int | None = None
    expires_at: str | None = None
    family: str | None = None
    quantization: str | None = None


class ModelVerification(BaseSchema):
    """Digest and quantization verification for one configured model."""

    tag: str
    name: str
    role: ModelRole
    installed: bool
    digest: str | None = None
    expected_digest: str | None = None
    digest_match: bool | None = None
    quantization: str | None = None
    expected_quantization: str | None = None
    quantization_match: bool | None = None
    problems: list[str] = Field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Return whether the model is installed and has no verification problems."""
        return self.installed and not self.problems


class CompatibilityAssessment(BaseSchema):
    """Heuristic memory-fit assessment for the target laptop."""

    fit: Literal[
        "full_gpu_possible",
        "partial_cpu_offload",
        "insufficient_memory",
        "unknown",
    ]
    target_vram_bytes: int
    target_ram_bytes: int
    estimated_kv_cache_bytes: int | None = None
    estimated_total_bytes: int | None = None
    notes: list[str] = Field(default_factory=list)


class ModelInspection(BaseSchema):
    """Combined metadata, verification, and compatibility for one model."""

    metadata: ModelMetadata
    verification: ModelVerification
    compatibility: CompatibilityAssessment


class ModelStatus(BaseSchema):
    """Compact inventory entry used by the CLI."""

    tag: str
    name: str
    role: ModelRole
    installed: bool
    digest: str | None = None
    size_bytes: int | None = None
    parameter_size: str | None = None
    quantization: str | None = None
    context_length: int | None = None


class ModelSnapshot(BaseSchema):
    """Point-in-time snapshot of the local model store and configured roster."""

    schema_version: str = "1.0.0"
    created_at: datetime
    ollama_version: str
    ollama_base_url: str
    config_hash: str
    config_dir: str
    project_root: str
    models: list[ModelMetadata] = Field(default_factory=list)
    verifications: list[ModelVerification] = Field(default_factory=list)
    loaded_models: list[LoadedModelSnapshot] = Field(default_factory=list)


def _int_value(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _string_value(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def parse_parameter_count(value: str | None) -> int | None:
    """Parse a human-readable parameter size such as 8.0B."""
    if not value:
        return None
    match = PARAMETER_PATTERN.fullmatch(value)
    if match is None:
        return None
    number = float(match.group(1))
    multiplier = PARAMETER_MULTIPLIERS[match.group(2).upper()]
    return int(number * multiplier)


def _architecture(info: dict[str, Any], family: str | None) -> str:
    return _string_value(info.get("general.architecture")) or family or ""


def _info_value(info: dict[str, Any], architecture: str, key: str) -> Any:
    if architecture:
        value = info.get(f"{architecture}.{key}")
        if value is not None:
            return value
    return info.get(f"general.{key}")


def build_model_metadata(
    spec: ModelSpec,
    summary: OllamaModelSummary,
    show: OllamaShowResponse,
    ollama_version: str,
    role: ModelRole,
    source: ModelSource = ModelSource.INSTALLED,
) -> ModelMetadata:
    """Build metadata from /api/tags and /api/show responses."""
    info = show.model_info
    family = summary.details.family or show.details.family or None
    architecture = _architecture(info, family)
    parameter_count = _int_value(info.get("general.parameter_count"))
    if parameter_count is None:
        parameter_count = parse_parameter_count(
            summary.details.parameter_size or show.details.parameter_size
        )
    context_length = _int_value(_info_value(info, architecture, "context_length"))
    embedding_length = _int_value(_info_value(info, architecture, "embedding_length"))
    return ModelMetadata(
        tag=spec.tag,
        name=spec.name,
        role=role,
        source=source,
        installed=True,
        digest=summary.digest or None,
        size_bytes=summary.size or None,
        family=family,
        parameter_count=parameter_count,
        parameter_size=summary.details.parameter_size or show.details.parameter_size or None,
        quantization=summary.details.quantization_level
        or show.details.quantization_level
        or None,
        context_length=context_length,
        embedding_length=embedding_length,
        capabilities=show.capabilities,
        license=_string_value(show.license) or _string_value(info.get("general.license")),
        modified_at=summary.modified_at or show.modified_at,
        ollama_version=ollama_version,
        model_info=info,
    )


def build_missing_metadata(
    spec: ModelSpec,
    ollama_version: str,
    role: ModelRole,
) -> ModelMetadata:
    """Build a metadata entry for a configured model that is not installed."""
    return ModelMetadata(
        tag=spec.tag,
        name=spec.name,
        role=role,
        source=ModelSource.MISSING,
        installed=False,
        ollama_version=ollama_version,
    )


def build_verification(metadata: ModelMetadata, spec: ModelSpec) -> ModelVerification:
    """Compare metadata with expected digest and quantization."""
    problems: list[str] = []
    digest_match: bool | None = None
    quantization_match: bool | None = None
    if not metadata.installed:
        problems.append("model is not installed")
    elif spec.expected_digest is not None:
        if metadata.digest is None:
            problems.append("installed digest is unavailable")
        else:
            installed = metadata.digest.lower().removeprefix("sha256:")
            expected = spec.expected_digest.lower().removeprefix("sha256:")
            digest_match = installed == expected
            if not digest_match:
                problems.append(
                    f"digest mismatch: expected {spec.expected_digest}, found {metadata.digest}"
                )
    if metadata.installed and spec.expected_quantization is not None:
        if metadata.quantization is None:
            problems.append("installed quantization is unavailable")
        else:
            quantization_match = (
                metadata.quantization.lower() == spec.expected_quantization.lower()
            )
            if not quantization_match:
                problems.append(
                    "quantization mismatch: expected "
                    f"{spec.expected_quantization}, found {metadata.quantization}"
                )
    return ModelVerification(
        tag=metadata.tag,
        name=metadata.name,
        role=metadata.role,
        installed=metadata.installed,
        digest=metadata.digest,
        expected_digest=spec.expected_digest,
        digest_match=digest_match,
        quantization=metadata.quantization,
        expected_quantization=spec.expected_quantization,
        quantization_match=quantization_match,
        problems=problems,
    )


def build_loaded_snapshot(model: OllamaLoadedModel) -> LoadedModelSnapshot:
    """Convert an /api/ps entry into a snapshot entry."""
    return LoadedModelSnapshot(
        tag=model.tag,
        name=model.name,
        digest=model.digest or None,
        size_bytes=model.size or None,
        size_vram_bytes=model.size_vram if model.size_vram else None,
        context_length=model.context_length,
        expires_at=model.expires_at,
        family=model.details.family or None,
        quantization=model.details.quantization_level or None,
    )


def _kv_head_count(value: Any) -> int | None:
    if isinstance(value, list):
        values = [item for item in (_int_value(entry) for entry in value) if item is not None]
        return max(values) if values else None
    return _int_value(value)


def estimate_kv_cache_bytes(metadata: ModelMetadata, num_ctx: int) -> int | None:
    """Estimate KV-cache size from model_info using an f16 assumption."""
    if not metadata.model_info or num_ctx <= 0:
        return None
    info = metadata.model_info
    architecture = _architecture(info, metadata.family)
    block_count = _int_value(_info_value(info, architecture, "block_count"))
    head_count = _int_value(_info_value(info, architecture, "attention.head_count"))
    kv_heads = _kv_head_count(
        _info_value(info, architecture, "attention.head_count_kv")
    )
    head_dim = _int_value(_info_value(info, architecture, "attention.key_length"))
    embedding_length = metadata.embedding_length or _int_value(
        _info_value(info, architecture, "embedding_length")
    )
    if head_dim is None and embedding_length is not None and head_count:
        head_dim = embedding_length // head_count
    if not block_count or not kv_heads or not head_dim:
        return None
    bytes_per_element = 2
    tensors = 2
    overhead_factor = 1.05
    return int(
        num_ctx
        * block_count
        * kv_heads
        * head_dim
        * bytes_per_element
        * tensors
        * overhead_factor
    )


def assess_compatibility(
    metadata: ModelMetadata,
    num_ctx: int,
    target_vram_bytes: int = TARGET_VRAM_BYTES,
    target_ram_bytes: int = TARGET_RAM_BYTES,
) -> CompatibilityAssessment:
    """Assess whether a model can fit on the target hardware using a heuristic."""
    notes = [
        "Heuristic estimate for the project's 4 GiB VRAM / 16 GiB RAM target.",
        "Actual layer placement must be measured with /api/ps after loading.",
    ]
    if not metadata.installed or metadata.size_bytes is None:
        return CompatibilityAssessment(
            fit="unknown",
            target_vram_bytes=target_vram_bytes,
            target_ram_bytes=target_ram_bytes,
            notes=[*notes, "Model is not installed, so memory fit cannot be assessed."],
        )
    kv_cache = estimate_kv_cache_bytes(metadata, num_ctx)
    total = metadata.size_bytes + (kv_cache or 0)
    if kv_cache is None:
        notes.append("KV cache could not be estimated; total omits its memory cost.")
    if total <= int(target_vram_bytes * 0.90):
        fit: Literal[
            "full_gpu_possible",
            "partial_cpu_offload",
            "insufficient_memory",
            "unknown",
        ] = "full_gpu_possible"
    elif total <= target_vram_bytes + target_ram_bytes:
        fit = "partial_cpu_offload"
    else:
        fit = "insufficient_memory"
    return CompatibilityAssessment(
        fit=fit,
        target_vram_bytes=target_vram_bytes,
        target_ram_bytes=target_ram_bytes,
        estimated_kv_cache_bytes=kv_cache,
        estimated_total_bytes=total,
        notes=notes,
    )


def format_pull_event(event: OllamaPullEvent) -> str:
    """Format one pull progress event for CLI display."""
    if event.total and event.completed is not None:
        percent = min(100.0, event.completed / event.total * 100.0)
        return f"{event.status}: {percent:.1f}%"
    return event.status or "pulling"


def format_bytes(value: int | None) -> str:
    """Format a byte count for CLI display."""
    if value is None:
        return "unknown"
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    size = float(value)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{value} B"


def write_snapshot(snapshot: ModelSnapshot, path: Path) -> Path:
    """Write a metadata snapshot atomically and return the resolved path."""
    target = path.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    payload = json.dumps(
        snapshot.model_dump(mode="json"),
        indent=2,
        ensure_ascii=False,
    )
    temporary.write_text(payload + "\n", encoding="utf-8")
    temporary.replace(target)
    return target


def read_snapshot(path: Path) -> ModelSnapshot:
    """Read and validate a metadata snapshot."""
    return ModelSnapshot.model_validate_json(path.read_text(encoding="utf-8"))


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)
