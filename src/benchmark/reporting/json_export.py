"""
File: json_export.py

Description:
Structured JSON export of one experiment report.

Why this file exists:
Downstream tooling needs the complete, machine-readable evidence: metadata,
configuration, responses, metrics, hardware, evaluations, and recommendations.

Responsibilities:
- Serialize the bundle into a documented, versioned JSON structure.
- Keep missing measurements as null rather than zero.
- Preserve Unicode text exactly.
- Include the preliminary flag and every limitation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from benchmark.reporting.bundle import REPORT_SCHEMA_VERSION, ReportBundle

BYTES_PER_MIB = 1024 * 1024


def _mib(value: int | None) -> float | None:
    return None if value is None else round(value / BYTES_PER_MIB, 4)


def _response_payload(bundle: ReportBundle) -> list[dict[str, Any]]:
    return [
        {
            "model_tag": row.model_tag,
            "model_name": row.model_name,
            "question_id": row.question_id,
            "category": row.category,
            "run_profile": row.run_profile,
            "repetition": row.repetition,
            "cold_start": row.cold_start,
            "status": row.response_status,
            "finish_reason": row.finish_reason,
            "prompt": row.prompt,
            "system_prompt": row.system_prompt,
            "response_text": row.response_text,
            "started_at": row.started_at.isoformat(),
            "completed_at": row.completed_at.isoformat(),
            "stream_chunk_count": row.stream_chunk_count,
            "error_type": row.error_type,
            "error_message": row.error_message,
        }
        for row in bundle.inspections
    ]


def _performance_payload(bundle: ReportBundle) -> list[dict[str, Any]]:
    return [
        {
            "model_tag": row.model_tag,
            "question_id": row.question_id,
            "repetition": row.repetition,
            "run_profile": row.run_profile,
            "cold_start": row.cold_start,
            "status": row.response_status,
            "total_latency_seconds": row.latency_seconds,
            "ttft_seconds": row.ttft_seconds,
            "load_duration_seconds": row.load_duration_seconds,
            "prompt_eval_duration_seconds": row.prompt_eval_duration_seconds,
            "eval_duration_seconds": row.eval_duration_seconds,
            "prompt_eval_count": row.prompt_eval_count,
            "eval_count": row.eval_count,
            "output_tokens_per_second": row.output_tokens_per_second,
            "prompt_tokens_per_second": row.prompt_tokens_per_second,
        }
        for row in bundle.inspections
    ]


def _hardware_payload(bundle: ReportBundle) -> list[dict[str, Any]]:
    return [
        {
            "model_tag": row.model_tag,
            "question_id": row.question_id,
            "repetition": row.repetition,
            "cold_start": row.cold_start,
            "sample_count": row.sample_count,
            "baseline_sample_count": row.baseline_sample_count,
            "inference_sample_count": row.inference_sample_count,
            "cpu_avg_percent": row.cpu_avg_percent,
            "cpu_peak_percent": row.cpu_peak_percent,
            "ram_avg_used_bytes": row.ram_avg_used_bytes,
            "ram_peak_used_bytes": row.ram_peak_used_bytes,
            "ram_avg_mib": _mib(row.ram_avg_used_bytes),
            "ram_peak_mib": _mib(row.ram_peak_used_bytes),
            "gpu_utilization_avg_percent": row.gpu_utilization_avg_percent,
            "gpu_utilization_peak_percent": row.gpu_utilization_peak_percent,
            "vram_avg_used_bytes": row.vram_avg_used_bytes,
            "vram_peak_used_bytes": row.vram_peak_used_bytes,
            "vram_avg_mib": _mib(row.vram_avg_used_bytes),
            "vram_peak_mib": _mib(row.vram_peak_used_bytes),
            "gpu_temperature_avg_c": row.gpu_temperature_avg_c,
            "gpu_temperature_peak_c": row.gpu_temperature_peak_c,
            "ollama_rss_avg_bytes": row.ollama_rss_avg_bytes,
            "ollama_rss_peak_bytes": row.ollama_rss_peak_bytes,
            "ollama_model_size_vram_bytes": row.ollama_model_size_vram_bytes,
            "gpu_name": row.gpu_name,
            "gpu_available": row.gpu_available,
            "gpu_utilization_available": row.gpu_utilization_available,
            "gpu_memory_available": row.gpu_memory_available,
            "gpu_temperature_available": row.gpu_temperature_available,
            "model_vram_available": row.model_vram_available,
            "gpu_error": row.gpu_error,
            "sampler_error": row.sampler_error,
            "attribution_note": row.attribution_note,
        }
        for row in bundle.hardware_rows
    ]


def _evaluation_payload(bundle: ReportBundle) -> list[dict[str, Any]]:
    return [
        {
            "model_tag": row.model_tag,
            "question_id": row.question_id,
            "category": row.category,
            "repetition": row.repetition,
            "evaluation_type": row.evaluation_type,
            "status": row.evaluation_status,
            "deterministic_score": row.deterministic_score,
            "deterministic_max_score": row.deterministic_max_score,
            "deterministic_normalized_score_0_100": row.deterministic_normalized_score,
            "passed": row.passed,
            "judge_status": row.judge_status,
            "judge_model_tag": row.judge_model_tag,
            "judge_raw_score": row.judge_raw_score,
            "judge_normalized_score_0_100": row.judge_normalized_score,
            "judge_missing": row.judge_missing,
            "composite_score_0_100": row.composite_score,
            "checks": row.checks,
            "judge_dimensions": row.judge_dimensions,
            "errors": row.evaluation_errors,
        }
        for row in bundle.inspections
    ]


def json_payload(bundle: ReportBundle) -> dict[str, Any]:
    """Build the structured JSON report payload."""
    experiment = bundle.experiment
    detail = bundle.detail
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "generated_at": bundle.generated_at.isoformat(),
        "preliminary": bundle.preliminary,
        "experiment": {
            "experiment_id": experiment.experiment_id,
            "name": experiment.name,
            "status": experiment.status,
            "created_at": experiment.created_at.isoformat(),
            "started_at": experiment.started_at.isoformat() if experiment.started_at else None,
            "completed_at": (
                experiment.completed_at.isoformat() if experiment.completed_at else None
            ),
            "run_count": experiment.run_count,
            "response_count": experiment.response_count,
            "evaluation_count": experiment.evaluation_count,
            "notes": detail.notes,
        },
        "dataset": {
            "name": experiment.dataset_name,
            "version": experiment.dataset_version,
            "content_sha256": experiment.dataset_hash,
            "schema_version": detail.dataset_schema_version,
            "item_count": detail.dataset_item_count,
            "source_path": detail.dataset_source_path,
            "random_seed": detail.dataset_random_seed,
        },
        "configuration": {
            "config_hash": experiment.config_hash,
            "config": detail.config,
            "host": detail.host,
            "software": detail.software,
            "database_path": bundle.database_path,
            "report_filters": {
                "profile": bundle.profile,
                "run_profile": bundle.run_profile,
                "include_cold_start": bundle.include_cold_start,
                "include_smoke": bundle.include_smoke,
            },
        },
        "model_metadata": [model.model_dump(mode="json") for model in detail.models],
        "runs": [run.model_dump(mode="json") for run in detail.runs],
        "responses": _response_payload(bundle),
        "performance_metrics": _performance_payload(bundle),
        "hardware_measurements": _hardware_payload(bundle),
        "evaluation_results": _evaluation_payload(bundle),
        "evaluation_breakdown": [
            row.model_dump(mode="json") for row in bundle.evaluation_breakdown
        ],
        "analytics": bundle.analysis.model_dump(mode="json"),
        "recommendations": [
            result.model_dump(mode="json") for result in bundle.recommendations
        ],
        "errors": [
            {
                **error,
                "occurred_at": (
                    error["occurred_at"].isoformat()
                    if hasattr(error.get("occurred_at"), "isoformat")
                    else error.get("occurred_at")
                ),
            }
            for error in bundle.errors
        ],
        "limitations": list(
            dict.fromkeys(item for item in bundle.limitations if item)
        ),
    }


def json_bytes(bundle: ReportBundle, *, indent: int = 2) -> bytes:
    """Return the structured JSON export as UTF-8 bytes."""
    return json.dumps(
        json_payload(bundle),
        ensure_ascii=False,
        indent=indent,
        sort_keys=False,
    ).encode("utf-8")


def write_json_export(bundle: ReportBundle, path: Path) -> Path:
    """Write the structured JSON export to a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json_bytes(bundle))
    return path


__all__ = ["json_bytes", "json_payload", "write_json_export"]
