"""
File: csv_export.py

Description:
CSV exports for model comparison, performance, hardware, evaluation, responses,
and experiment summaries.

Why this file exists:
Spreadsheet workflows need flat, stable columns with units in the header and empty
cells for missing measurements. Missing values must never become zero.

Responsibilities:
- Define one explicit column set per export so headers exist even with no rows.
- Convert bytes to MiB and fractions to percentages with units in the header.
- Preserve full Unicode response text.
- Write payloads to disk or return them as bytes for the dashboard.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from benchmark.reporting.bundle import ReportBundle

BYTES_PER_MIB = 1024 * 1024

CSV_FILENAMES: dict[str, str] = {
    "experiment_summary": "experiment_summary.csv",
    "model_comparison": "model_comparison.csv",
    "performance_metrics": "performance_metrics.csv",
    "hardware_metrics": "hardware_metrics.csv",
    "evaluation_results": "evaluation_results.csv",
    "responses": "responses.csv",
}

EXPERIMENT_SUMMARY_COLUMNS = [
    "experiment_id",
    "name",
    "status",
    "preliminary",
    "dataset_name",
    "dataset_version",
    "dataset_hash",
    "dataset_schema_version",
    "dataset_item_count",
    "config_hash",
    "created_at",
    "started_at",
    "completed_at",
    "runs",
    "measured_responses",
    "evaluated_responses",
    "profile",
    "models",
]

MODEL_COMPARISON_COLUMNS = [
    "model_tag",
    "model_name",
    "measured_responses",
    "successful_responses",
    "failed_responses",
    "success_rate_pct",
    "failure_rate_pct",
    "timeout_rate_pct",
    "cancellation_rate_pct",
    "latency_mean_seconds",
    "latency_median_seconds",
    "latency_p95_seconds",
    "ttft_mean_seconds",
    "output_tokens_per_second_mean",
    "prompt_tokens_per_second_mean",
    "quality_score_0_100",
    "deterministic_score_0_100",
    "judge_score_0_100",
    "reliability_score_0_100",
    "composite_score_0_100",
    "quality_coverage_pct",
    "judge_coverage_pct",
    "cpu_avg_percent",
    "ram_avg_mib",
    "gpu_avg_percent",
    "vram_avg_mib",
    "model_vram_avg_mib",
    "response_consistency_0_1",
]

PERFORMANCE_METRICS_COLUMNS = [
    "model_tag",
    "question_id",
    "category",
    "repetition",
    "run_profile",
    "cold_start",
    "response_status",
    "total_latency_seconds",
    "ttft_seconds",
    "load_duration_seconds",
    "prompt_eval_duration_seconds",
    "eval_duration_seconds",
    "prompt_eval_count",
    "eval_count",
    "output_tokens_per_second",
    "prompt_tokens_per_second",
    "stream_chunk_count",
]

HARDWARE_METRICS_COLUMNS = [
    "model_tag",
    "question_id",
    "repetition",
    "cold_start",
    "sample_count",
    "baseline_sample_count",
    "inference_sample_count",
    "cpu_avg_percent",
    "cpu_peak_percent",
    "ram_avg_mib",
    "ram_peak_mib",
    "gpu_utilization_avg_percent",
    "gpu_utilization_peak_percent",
    "vram_avg_mib",
    "vram_peak_mib",
    "gpu_temperature_avg_c",
    "gpu_temperature_peak_c",
    "ollama_rss_avg_mib",
    "ollama_rss_peak_mib",
    "ollama_model_vram_mib",
    "gpu_name",
    "gpu_available",
    "gpu_utilization_available",
    "gpu_memory_available",
    "gpu_temperature_available",
    "model_vram_available",
    "gpu_error",
    "sampler_error",
    "attribution_note",
]

EVALUATION_RESULTS_COLUMNS = [
    "model_tag",
    "question_id",
    "category",
    "repetition",
    "evaluation_type",
    "evaluation_status",
    "deterministic_score_0_100",
    "deterministic_max_score",
    "passed",
    "judge_status",
    "judge_model_tag",
    "judge_raw_score",
    "judge_normalized_score_0_100",
    "judge_missing",
    "composite_score_0_100",
    "check_count",
    "error_count",
]

RESPONSES_COLUMNS = [
    "model_tag",
    "model_name",
    "question_id",
    "category",
    "run_profile",
    "repetition",
    "cold_start",
    "response_status",
    "finish_reason",
    "prompt",
    "system_prompt",
    "response_text",
    "started_at",
    "completed_at",
    "total_latency_seconds",
    "ttft_seconds",
    "eval_count",
    "output_tokens_per_second",
    "composite_score_0_100",
    "error_type",
    "error_message",
]


def _fraction(value: float | None) -> float | None:
    return None if value is None else value * 100.0


def _mib(value: int | None) -> float | None:
    return None if value is None else value / BYTES_PER_MIB


def _neutralize_formula(value: Any) -> Any:
    """Prefix spreadsheet formula triggers in text cells to prevent execution."""
    if isinstance(value, str) and value[:1] in {"=", "+", "-", "@"}:
        return "'" + value
    return value


def _frame(rows: list[dict[str, Any]], columns: list[str]) -> pd.DataFrame:
    """Build a DataFrame with the exact declared columns."""
    return pd.DataFrame(rows, columns=columns).map(_neutralize_formula)


def csv_frames(bundle: ReportBundle) -> dict[str, pd.DataFrame]:
    """Build every CSV table for one experiment."""
    experiment = bundle.experiment
    analysis = bundle.analysis

    summary = _frame(
        [
            {
                "experiment_id": experiment.experiment_id,
                "name": experiment.name,
                "status": experiment.status,
                "preliminary": bundle.preliminary,
                "dataset_name": experiment.dataset_name,
                "dataset_version": experiment.dataset_version,
                "dataset_hash": experiment.dataset_hash,
                "dataset_schema_version": bundle.detail.dataset_schema_version,
                "dataset_item_count": bundle.detail.dataset_item_count,
                "config_hash": experiment.config_hash,
                "created_at": experiment.created_at.isoformat(),
                "started_at": (
                    experiment.started_at.isoformat() if experiment.started_at else None
                ),
                "completed_at": (
                    experiment.completed_at.isoformat()
                    if experiment.completed_at
                    else None
                ),
                "runs": len(bundle.detail.runs),
                "measured_responses": analysis.measured_responses,
                "evaluated_responses": analysis.evaluated_responses,
                "profile": analysis.profile,
                "models": ", ".join(model.model_tag for model in analysis.models),
            }
        ],
        EXPERIMENT_SUMMARY_COLUMNS,
    )

    model_rows = [
        {
            "model_tag": model.model_tag,
            "model_name": model.model_name,
            "measured_responses": model.statistics.measured_responses,
            "successful_responses": model.statistics.successful_responses,
            "failed_responses": model.statistics.failed_responses,
            "success_rate_pct": _fraction(model.statistics.success_rate),
            "failure_rate_pct": _fraction(model.statistics.failure_rate),
            "timeout_rate_pct": _fraction(model.statistics.timeout_rate),
            "cancellation_rate_pct": _fraction(model.statistics.cancellation_rate),
            "latency_mean_seconds": model.statistics.latency.mean,
            "latency_median_seconds": model.statistics.latency.median,
            "latency_p95_seconds": model.statistics.latency.p95,
            "ttft_mean_seconds": model.statistics.ttft.mean,
            "output_tokens_per_second_mean": model.statistics.output_tokens_per_second.mean,
            "prompt_tokens_per_second_mean": model.statistics.prompt_tokens_per_second.mean,
            "quality_score_0_100": model.quality_score,
            "deterministic_score_0_100": model.deterministic_score,
            "judge_score_0_100": model.judge_score,
            "reliability_score_0_100": model.reliability_score,
            "composite_score_0_100": model.composite_score,
            "quality_coverage_pct": _fraction(model.quality.quality_coverage),
            "judge_coverage_pct": _fraction(model.quality.judge_coverage),
            "cpu_avg_percent": model.resources.cpu_avg_percent,
            "ram_avg_mib": model.resources.ram_avg_mib,
            "gpu_avg_percent": model.resources.gpu_utilization_avg_percent,
            "vram_avg_mib": model.resources.vram_avg_mib,
            "model_vram_avg_mib": model.resources.model_vram_avg_mib,
            "response_consistency_0_1": model.statistics.response_consistency,
        }
        for model in analysis.models
    ]

    performance_rows = [
        {
            "model_tag": row.model_tag,
            "question_id": row.question_id,
            "category": row.category,
            "repetition": row.repetition,
            "run_profile": row.run_profile,
            "cold_start": row.cold_start,
            "response_status": row.response_status,
            "total_latency_seconds": row.latency_seconds,
            "ttft_seconds": row.ttft_seconds,
            "load_duration_seconds": row.load_duration_seconds,
            "prompt_eval_duration_seconds": row.prompt_eval_duration_seconds,
            "eval_duration_seconds": row.eval_duration_seconds,
            "prompt_eval_count": row.prompt_eval_count,
            "eval_count": row.eval_count,
            "output_tokens_per_second": row.output_tokens_per_second,
            "prompt_tokens_per_second": row.prompt_tokens_per_second,
            "stream_chunk_count": row.stream_chunk_count,
        }
        for row in bundle.inspections
    ]

    hardware_rows = [
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
            "ram_avg_mib": _mib(row.ram_avg_used_bytes),
            "ram_peak_mib": _mib(row.ram_peak_used_bytes),
            "gpu_utilization_avg_percent": row.gpu_utilization_avg_percent,
            "gpu_utilization_peak_percent": row.gpu_utilization_peak_percent,
            "vram_avg_mib": _mib(row.vram_avg_used_bytes),
            "vram_peak_mib": _mib(row.vram_peak_used_bytes),
            "gpu_temperature_avg_c": row.gpu_temperature_avg_c,
            "gpu_temperature_peak_c": row.gpu_temperature_peak_c,
            "ollama_rss_avg_mib": _mib(row.ollama_rss_avg_bytes),
            "ollama_rss_peak_mib": _mib(row.ollama_rss_peak_bytes),
            "ollama_model_vram_mib": _mib(row.ollama_model_size_vram_bytes),
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

    evaluation_rows = [
        {
            "model_tag": row.model_tag,
            "question_id": row.question_id,
            "category": row.category,
            "repetition": row.repetition,
            "evaluation_type": row.evaluation_type,
            "evaluation_status": row.evaluation_status,
            "deterministic_score_0_100": row.deterministic_normalized_score,
            "deterministic_max_score": row.deterministic_max_score,
            "passed": row.passed,
            "judge_status": row.judge_status,
            "judge_model_tag": row.judge_model_tag,
            "judge_raw_score": row.judge_raw_score,
            "judge_normalized_score_0_100": row.judge_normalized_score,
            "judge_missing": row.judge_missing,
            "composite_score_0_100": row.composite_score,
            "check_count": len(row.checks),
            "error_count": len(row.evaluation_errors),
        }
        for row in bundle.inspections
    ]

    response_rows = [
        {
            "model_tag": row.model_tag,
            "model_name": row.model_name,
            "question_id": row.question_id,
            "category": row.category,
            "run_profile": row.run_profile,
            "repetition": row.repetition,
            "cold_start": row.cold_start,
            "response_status": row.response_status,
            "finish_reason": row.finish_reason,
            "prompt": row.prompt,
            "system_prompt": row.system_prompt,
            "response_text": row.response_text,
            "started_at": row.started_at.isoformat(),
            "completed_at": row.completed_at.isoformat(),
            "total_latency_seconds": row.latency_seconds,
            "ttft_seconds": row.ttft_seconds,
            "eval_count": row.eval_count,
            "output_tokens_per_second": row.output_tokens_per_second,
            "composite_score_0_100": row.composite_score,
            "error_type": row.error_type,
            "error_message": row.error_message,
        }
        for row in bundle.inspections
    ]

    return {
        "experiment_summary": summary,
        "model_comparison": _frame(model_rows, MODEL_COMPARISON_COLUMNS),
        "performance_metrics": _frame(performance_rows, PERFORMANCE_METRICS_COLUMNS),
        "hardware_metrics": _frame(hardware_rows, HARDWARE_METRICS_COLUMNS),
        "evaluation_results": _frame(evaluation_rows, EVALUATION_RESULTS_COLUMNS),
        "responses": _frame(response_rows, RESPONSES_COLUMNS),
    }


def csv_payloads(bundle: ReportBundle) -> dict[str, bytes]:
    """Return every CSV export as UTF-8 bytes keyed by file name."""
    payloads: dict[str, bytes] = {}
    for key, frame in csv_frames(bundle).items():
        payloads[CSV_FILENAMES[key]] = frame.to_csv(
            index=False,
            na_rep="",
            lineterminator="\n",
        ).encode("utf-8")
    return payloads


def write_csv_exports(bundle: ReportBundle, directory: Path) -> list[Path]:
    """Write every CSV export into a directory and return the paths."""
    directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for name, data in csv_payloads(bundle).items():
        path = directory / name
        path.write_bytes(data)
        paths.append(path)
    return paths


__all__ = [
    "CSV_FILENAMES",
    "csv_frames",
    "csv_payloads",
    "write_csv_exports",
]
