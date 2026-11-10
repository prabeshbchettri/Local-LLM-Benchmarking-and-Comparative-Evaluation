"""
File: test_metadata.py

Description:
Unit tests for model metadata, verification, compatibility, and snapshots.

Responsibilities:
- Verify metadata extraction from Ollama responses.
- Verify digest, quantization, and missing-model checks.
- Verify heuristic compatibility and snapshot round trips.
"""

from __future__ import annotations

import pytest

from benchmark.core.config import ModelSpec
from benchmark.execution.metadata import (
    ModelMetadata,
    ModelRole,
    ModelSnapshot,
    ModelSource,
    assess_compatibility,
    build_loaded_snapshot,
    build_missing_metadata,
    build_model_metadata,
    build_verification,
    estimate_kv_cache_bytes,
    format_bytes,
    parse_parameter_count,
    read_snapshot,
    utc_now,
    write_snapshot,
)
from benchmark.models.ollama import (
    OllamaLoadedModel,
    OllamaModelDetailsSummary,
    OllamaModelSummary,
    OllamaShowResponse,
)

pytestmark = pytest.mark.unit

DIGEST = "a" * 64


def _spec() -> ModelSpec:
    return ModelSpec(
        name="Llama 3.1 8B Instruct",
        tag="llama3.1:8b",
        expected_digest=DIGEST,
        expected_quantization="Q4_K_M",
    )


def _summary(digest: str = DIGEST) -> OllamaModelSummary:
    return OllamaModelSummary.model_validate(
        {
            "name": "llama3.1:8b",
            "model": "llama3.1:8b",
            "modified_at": "2026-11-10T00:00:00Z",
            "size": 4_920_753_328,
            "digest": digest,
            "details": {
                "family": "llama",
                "parameter_size": "8.0B",
                "quantization_level": "Q4_K_M",
            },
        }
    )


def _show() -> OllamaShowResponse:
    return OllamaShowResponse.model_validate(
        {
            "details": {
                "family": "llama",
                "parameter_size": "8.0B",
                "quantization_level": "Q4_K_M",
            },
            "model_info": {
                "general.architecture": "llama",
                "general.parameter_count": 8_030_261_312,
                "llama.context_length": 131_072,
                "llama.embedding_length": 4096,
                "llama.block_count": 32,
                "llama.attention.head_count": 32,
                "llama.attention.head_count_kv": 8,
            },
            "capabilities": ["completion", "tools"],
            "license": "Llama 3.1 Community License",
        }
    )


def _metadata(digest: str = DIGEST, quantization: str = "Q4_K_M") -> ModelMetadata:
    summary = _summary(digest)
    summary.details.quantization_level = quantization
    return build_model_metadata(
        _spec(),
        summary,
        _show(),
        "0.40.15",
        ModelRole.CANDIDATE,
    )


def test_parse_parameter_count() -> None:
    assert parse_parameter_count("8.0B") == 8_000_000_000
    assert parse_parameter_count("7.6B") == 7_600_000_000
    assert parse_parameter_count("not-a-size") is None


def test_build_model_metadata() -> None:
    metadata = _metadata()
    assert metadata.tag == "llama3.1:8b"
    assert metadata.digest == DIGEST
    assert metadata.family == "llama"
    assert metadata.parameter_count == 8_030_261_312
    assert metadata.context_length == 131_072
    assert metadata.embedding_length == 4096
    assert metadata.quantization == "Q4_K_M"
    assert metadata.capabilities == ["completion", "tools"]
    assert metadata.license == "Llama 3.1 Community License"


def test_verification_passes_for_matching_metadata() -> None:
    verification = build_verification(_metadata(), _spec())
    assert verification.ok
    assert verification.digest_match is True
    assert verification.quantization_match is True
    assert verification.problems == []


def test_verification_reports_digest_and_quantization_mismatches() -> None:
    verification = build_verification(
        _metadata(digest="b" * 64, quantization="Q5_K_M"),
        _spec(),
    )
    assert not verification.ok
    assert verification.digest_match is False
    assert verification.quantization_match is False
    assert len(verification.problems) == 2


def test_missing_metadata_has_one_problem() -> None:
    metadata = build_missing_metadata(_spec(), "0.40.15", ModelRole.CANDIDATE)
    verification = build_verification(metadata, _spec())
    assert metadata.source == ModelSource.MISSING
    assert not metadata.installed
    assert verification.problems == ["model is not installed"]


def test_estimate_kv_cache() -> None:
    metadata = _metadata()
    estimate = estimate_kv_cache_bytes(metadata, 4096)
    assert estimate is not None
    assert estimate > 0
    assert estimate_kv_cache_bytes(metadata, 0) is None


def test_compatibility_assessment() -> None:
    metadata = _metadata()
    metadata.size_bytes = 1 * 1024**3
    full = assess_compatibility(metadata, 4096)
    assert full.fit == "full_gpu_possible"
    assert full.estimated_total_bytes is not None

    metadata.size_bytes = 5 * 1024**3
    partial = assess_compatibility(metadata, 4096)
    assert partial.fit == "partial_cpu_offload"

    metadata.size_bytes = 30 * 1024**3
    insufficient = assess_compatibility(metadata, 4096)
    assert insufficient.fit == "insufficient_memory"

    metadata.installed = False
    metadata.size_bytes = None
    unknown = assess_compatibility(metadata, 4096)
    assert unknown.fit == "unknown"


def test_format_bytes() -> None:
    assert format_bytes(None) == "unknown"
    assert format_bytes(512) == "512 B"
    assert format_bytes(1536) == "1.50 KiB"
    assert format_bytes(4 * 1024**3) == "4.00 GiB"


def test_loaded_model_snapshot() -> None:
    loaded = OllamaLoadedModel.model_validate(
        {
            "name": "qwen2.5:3b",
            "model": "qwen2.5:3b",
            "size": 2_163_568_803,
            "size_vram": 2_163_568_803,
            "context_length": 4096,
            "details": {
                "family": "qwen2",
                "quantization_level": "Q4_K_M",
            },
        }
    )
    snapshot = build_loaded_snapshot(loaded)
    assert snapshot.tag == "qwen2.5:3b"
    assert snapshot.size_vram_bytes == 2_163_568_803
    assert snapshot.context_length == 4096


def test_snapshot_round_trip(tmp_path) -> None:
    metadata = _metadata()
    snapshot = ModelSnapshot(
        created_at=utc_now(),
        ollama_version="0.40.15",
        ollama_base_url="http://127.0.0.1:11434",
        config_hash="c" * 64,
        config_dir="configs",
        project_root=".",
        models=[metadata],
        verifications=[build_verification(metadata, _spec())],
    )
    path = write_snapshot(snapshot, tmp_path / "model_metadata.json")
    loaded = read_snapshot(path)
    assert loaded.schema_version == "1.0.0"
    assert loaded.models[0].tag == "llama3.1:8b"
    assert loaded.verifications[0].ok
