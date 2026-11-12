"""
File: comparison.py

Description:
Historical metric comparisons between stored experiments.

Why this file exists:
A model can improve on one metric while regressing on another. Comparing raw values
and percent changes makes those trade-offs explicit instead of hiding them in a
single new score.

Responsibilities:
- Compare quality, P95 latency, throughput, reliability, CPU, and RAM per model.
- Report absolute and percent changes, preserving missing measurements.
- Use Pandas to align model rows across experiments.
"""

from __future__ import annotations

import pandas as pd

from benchmark.analytics.models import (
    ExperimentAnalytics,
    ExperimentComparison,
    MetricDelta,
    ModelComparison,
)

METRICS: dict[str, str] = {
    "quality_score": "quality_score",
    "p95_latency_seconds": "latency_p95",
    "mean_ttft_seconds": "ttft_mean",
    "mean_output_tokens_per_second": "throughput_mean",
    "success_rate": "success_rate",
    "failure_rate": "failure_rate",
    "timeout_rate": "timeout_rate",
    "cpu_avg_percent": "cpu_avg_percent",
    "ram_avg_mib": "ram_avg_mib",
    "gpu_utilization_avg_percent": "gpu_utilization_avg_percent",
    "vram_avg_mib": "vram_avg_mib",
    "model_vram_avg_mib": "model_vram_avg_mib",
}
LOWER_IS_BETTER = {
    "p95_latency_seconds",
    "mean_ttft_seconds",
    "failure_rate",
    "timeout_rate",
    "cpu_avg_percent",
    "ram_avg_mib",
    "gpu_utilization_avg_percent",
    "vram_avg_mib",
    "model_vram_avg_mib",
}


def _model_values(analytics: ExperimentAnalytics) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for model in analytics.models:
        rows.append(
            {
                "model_tag": model.model_tag,
                "quality_score": model.quality_score,
                "p95_latency_seconds": model.statistics.latency.p95,
                "mean_ttft_seconds": model.statistics.ttft.mean,
                "mean_output_tokens_per_second": (
                    model.statistics.output_tokens_per_second.mean
                ),
                "success_rate": model.statistics.success_rate,
                "failure_rate": model.statistics.failure_rate,
                "timeout_rate": model.statistics.timeout_rate,
                "cpu_avg_percent": model.resources.cpu_avg_percent,
                "ram_avg_mib": model.resources.ram_avg_mib,
                "gpu_utilization_avg_percent": (
                    model.resources.gpu_utilization_avg_percent
                ),
                "vram_avg_mib": model.resources.vram_avg_mib,
                "model_vram_avg_mib": model.resources.model_vram_avg_mib,
            }
        )
    return rows


def _delta(metric: str, baseline: float | None, candidate: float | None) -> MetricDelta:
    if baseline is None or candidate is None:
        return MetricDelta(
            metric=metric,
            baseline=baseline,
            candidate=candidate,
            available=False,
        )
    delta = candidate - baseline
    percent = None
    if baseline != 0:
        percent = delta / abs(baseline) * 100.0
    return MetricDelta(
        metric=metric,
        baseline=baseline,
        candidate=candidate,
        delta=float(delta),
        percent_change=float(percent) if percent is not None else None,
        available=True,
    )


def compare_analytics(
    baseline: ExperimentAnalytics,
    candidate: ExperimentAnalytics,
) -> ExperimentComparison:
    """Compare two analyses per model using Pandas-aligned rows."""
    baseline_frame = pd.DataFrame(_model_values(baseline)).set_index("model_tag")
    candidate_frame = pd.DataFrame(_model_values(candidate)).set_index("model_tag")
    common = sorted(set(baseline_frame.index) & set(candidate_frame.index))
    limitations: list[str] = []
    baseline_only = sorted(set(baseline_frame.index) - set(candidate_frame.index))
    candidate_only = sorted(set(candidate_frame.index) - set(baseline_frame.index))
    if baseline_only:
        limitations.append(
            "models only in the baseline experiment: " + ", ".join(baseline_only)
        )
    if candidate_only:
        limitations.append(
            "models only in the candidate experiment: " + ", ".join(candidate_only)
        )
    if not common:
        limitations.append("the two experiments share no model tags")
    if baseline.profile != candidate.profile:
        limitations.append(
            f"profiles differ: baseline {baseline.profile}, "
            f"candidate {candidate.profile}"
        )
    if baseline.run_profile != candidate.run_profile:
        limitations.append(
            "run-profile filters differ: baseline "
            f"{baseline.run_profile or 'all'}, candidate "
            f"{candidate.run_profile or 'all'}"
        )

    comparisons: list[ModelComparison] = []
    for tag in common:
        baseline_row = baseline_frame.loc[tag]
        candidate_row = candidate_frame.loc[tag]
        deltas: list[MetricDelta] = []
        model_limitations: list[str] = []
        for metric in METRICS:
            baseline_value = baseline_row.get(metric)
            candidate_value = candidate_row.get(metric)
            delta = _delta(
                metric,
                float(baseline_value) if pd.notna(baseline_value) else None,
                float(candidate_value) if pd.notna(candidate_value) else None,
            )
            deltas.append(delta)
            if not delta.available:
                model_limitations.append(
                    f"{metric} is missing in at least one experiment"
                )
        comparisons.append(
            ModelComparison(
                model_tag=tag,
                metrics=deltas,
                limitations=model_limitations,
            )
        )
    return ExperimentComparison(
        baseline_experiment_id=baseline.experiment_id,
        candidate_experiment_id=candidate.experiment_id,
        baseline_profile=baseline.profile,
        candidate_profile=candidate.profile,
        models=comparisons,
        limitations=limitations,
    )
