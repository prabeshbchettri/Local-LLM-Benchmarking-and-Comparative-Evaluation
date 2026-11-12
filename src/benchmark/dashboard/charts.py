"""
File: charts.py

Description:
Plotly chart builders shared by the dashboard pages.

Why this file exists:
Charts must plot the exact values produced by the analytics engine and repository
reads. Keeping figure construction here gives every page the same palette, sizing,
and hover behavior, and lets tests compare trace values with engine output.

Responsibilities:
- Build latency, TTFT, throughput, prompt-speed, and cold/warm charts.
- Build quality, category, evaluation-rate, and judge-score charts.
- Build hardware average/peak and per-response timeline charts.
- Build historical percent-change and dimension-comparison charts.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from benchmark.analytics.models import (
    DIMENSIONS,
    CategoryQuality,
    ExperimentAnalytics,
    ExperimentComparison,
    ModelAnalytics,
)
from benchmark.database.repositories import AnalyticsRow, EvaluationBreakdownView, HardwareUsageRow

PALETTE = ["#2563eb", "#0ea5e9", "#14b8a6", "#f59e0b", "#ef4444", "#8b5cf6"]
DETERMINISTIC_COLOR = "#2563eb"
JUDGE_COLOR = "#f59e0b"
COMPOSITE_COLOR = "#14b8a6"
WARM_COLOR = "#2563eb"
COLD_COLOR = "#f59e0b"
DIMENSION_LABELS = {
    "quality": "Quality",
    "latency": "Latency",
    "throughput": "Throughput",
    "efficiency": "Efficiency",
    "reliability": "Reliability",
}
METRIC_LABELS = {
    "quality_score": "Quality",
    "p95_latency_seconds": "P95 latency",
    "mean_ttft_seconds": "Mean TTFT",
    "mean_output_tokens_per_second": "Throughput",
    "success_rate": "Success rate",
    "failure_rate": "Failure rate",
    "timeout_rate": "Timeout rate",
    "cpu_avg_percent": "CPU avg",
    "ram_avg_mib": "RAM avg",
    "gpu_utilization_avg_percent": "GPU avg",
    "vram_avg_mib": "VRAM avg",
    "model_vram_avg_mib": "Model VRAM",
}
EVALUATION_LABELS = {
    "code_tests": "Coding tests",
    "json_schema": "JSON validity",
    "instruction_constraints": "Instruction following",
    "exact_match": "Exact match",
    "numeric": "Numeric",
    "reference_overlap": "Reference overlap",
    "refusal_check": "Refusal check",
    "composite": "Composite",
}


def _apply(
    figure: go.Figure,
    *,
    height: int = 360,
    y_title: str | None = None,
    x_title: str | None = None,
) -> go.Figure:
    figure.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=10, r=10, t=20, b=10),
        font=dict(size=13),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        hoverlabel=dict(font_size=12),
    )
    if y_title:
        figure.update_yaxes(title_text=y_title)
    if x_title:
        figure.update_xaxes(title_text=x_title)
    return figure


def model_color_map(tags: Sequence[str]) -> dict[str, str]:
    """Return a stable color per model tag."""
    ordered = list(dict.fromkeys(tags))
    return {tag: PALETTE[index % len(PALETTE)] for index, tag in enumerate(ordered)}


def _selected_models(
    analysis: ExperimentAnalytics, models: Sequence[str] | None
) -> list[ModelAnalytics]:
    allowed = set(models) if models else None
    return [
        model
        for model in analysis.models
        if allowed is None or model.model_tag in allowed
    ]


def latency_distribution(
    rows: Sequence[AnalyticsRow],
    models: Sequence[str] | None = None,
) -> go.Figure | None:
    """Box plot of successful per-response total latency."""
    allowed = set(models) if models else None
    values = [
        {
            "model": row.model_tag,
            "latency": row.latency_seconds,
        }
        for row in rows
        if row.response_status == "success"
        and row.latency_seconds is not None
        and (allowed is None or row.model_tag in allowed)
    ]
    if not values:
        return None
    frame = pd.DataFrame(values)
    colors = model_color_map(frame["model"].unique())
    figure = px.box(
        frame,
        x="model",
        y="latency",
        color="model",
        color_discrete_map=colors,
        points="outliers",
        labels={"model": "Model", "latency": "Total latency (s)"},
    )
    figure.update_layout(showlegend=False)
    return _apply(figure, y_title="Total latency (s)")


def ttft_comparison(
    analysis: ExperimentAnalytics,
    models: Sequence[str] | None = None,
) -> go.Figure | None:
    """Grouped bars of mean and P95 time to first token."""
    selected = _selected_models(analysis, models)
    if not selected:
        return None
    means = [model.statistics.ttft.mean for model in selected]
    p95 = [model.statistics.ttft.p95 for model in selected]
    if all(value is None for value in means + p95):
        return None
    figure = go.Figure()
    figure.add_bar(
        name="Mean TTFT",
        x=[model.model_tag for model in selected],
        y=means,
        marker_color=DETERMINISTIC_COLOR,
    )
    figure.add_bar(
        name="P95 TTFT",
        x=[model.model_tag for model in selected],
        y=p95,
        marker_color="#93c5fd",
    )
    figure.update_layout(barmode="group")
    return _apply(figure, y_title="Time to first token (s)")


def throughput_comparison(
    analysis: ExperimentAnalytics,
    models: Sequence[str] | None = None,
) -> go.Figure | None:
    """Grouped bars of mean and P95 output tokens per second."""
    selected = _selected_models(analysis, models)
    if not selected:
        return None
    means = [
        model.statistics.output_tokens_per_second.mean for model in selected
    ]
    p95 = [model.statistics.output_tokens_per_second.p95 for model in selected]
    if all(value is None for value in means + p95):
        return None
    figure = go.Figure()
    figure.add_bar(
        name="Mean",
        x=[model.model_tag for model in selected],
        y=means,
        marker_color=DETERMINISTIC_COLOR,
    )
    figure.add_bar(
        name="P95",
        x=[model.model_tag for model in selected],
        y=p95,
        marker_color="#93c5fd",
    )
    figure.update_layout(barmode="group")
    return _apply(figure, y_title="Output tokens per second")


def prompt_speed_comparison(
    analysis: ExperimentAnalytics,
    models: Sequence[str] | None = None,
) -> go.Figure | None:
    """Bars of mean prompt processing speed."""
    selected = _selected_models(analysis, models)
    values = [
        model.statistics.prompt_tokens_per_second.mean for model in selected
    ]
    if all(value is None for value in values):
        return None
    figure = go.Figure(
        go.Bar(
            x=[model.model_tag for model in selected],
            y=values,
            marker_color="#14b8a6",
            hovertemplate="%{x}<br>%{y:.1f} prompt tok/s<extra></extra>",
        )
    )
    return _apply(figure, y_title="Prompt tokens per second")


def cold_warm_comparison(
    rows: Sequence[AnalyticsRow],
    metric: str,
    label: str,
    models: Sequence[str] | None = None,
) -> go.Figure | None:
    """Grouped bars comparing cold-start and warm successful responses."""
    allowed = set(models) if models else None
    values: list[dict[str, Any]] = []
    for row in rows:
        if row.response_status != "success" or (allowed and row.model_tag not in allowed):
            continue
        value = getattr(row, metric)
        if value is None:
            continue
        values.append(
            {
                "model": row.model_tag,
                "state": "Cold start" if row.cold_start else "Warm",
                "value": value,
            }
        )
    if not values:
        return None
    frame = pd.DataFrame(values)
    grouped = (
        frame.groupby(["model", "state"], as_index=False)["value"].mean()
    )
    figure = px.bar(
        grouped,
        x="model",
        y="value",
        color="state",
        barmode="group",
        color_discrete_map={"Warm": WARM_COLOR, "Cold start": COLD_COLOR},
        labels={"model": "Model", "value": label, "state": ""},
        category_orders={"state": ["Warm", "Cold start"]},
    )
    return _apply(figure, y_title=label)


def composite_ranking(
    analysis: ExperimentAnalytics,
    models: Sequence[str] | None = None,
) -> go.Figure | None:
    """Horizontal bars of workload composite scores."""
    selected = [
        model
        for model in _selected_models(analysis, models)
        if model.composite_score is not None
    ]
    if not selected:
        return None
    selected.sort(key=lambda model: model.composite_score or 0.0)
    figure = go.Figure(
        go.Bar(
            x=[model.composite_score for model in selected],
            y=[model.model_tag for model in selected],
            orientation="h",
            marker_color=DETERMINISTIC_COLOR,
            text=[f"{model.composite_score:.1f}" for model in selected],
            textposition="outside",
            hovertemplate="%{y}<br>Composite %{x:.2f}/100<extra></extra>",
        )
    )
    figure.update_xaxes(range=[0, 105])
    return _apply(figure, x_title=f"Composite score ({analysis.profile})")


def dimension_radar(
    analysis: ExperimentAnalytics,
    models: Sequence[str] | None = None,
) -> go.Figure | None:
    """Radar chart of normalized scoring dimensions for complete models."""
    selected = [
        model
        for model in _selected_models(analysis, models)
        if all(
            model.dimensions.get(name) is not None
            and model.dimensions[name].normalized_score is not None
            for name in DIMENSIONS
        )
    ]
    if len(selected) < 1:
        return None
    colors = model_color_map([model.model_tag for model in selected])
    figure = go.Figure()
    closed = list(DIMENSIONS) + [DIMENSIONS[0]]
    for model in selected:
        values = [
            model.dimensions[name].normalized_score for name in closed
        ]
        figure.add_trace(
            go.Scatterpolar(
                r=values,
                theta=[DIMENSION_LABELS[name] for name in closed],
                name=model.model_tag,
                line=dict(color=colors[model.model_tag], width=2),
                fill="toself",
                opacity=0.18,
                hovertemplate="%{theta}: %{r:.1f}<extra>%{fullData.name}</extra>",
            )
        )
    figure.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=True,
    )
    return _apply(figure, height=420)


def quality_overall(
    analysis: ExperimentAnalytics,
    models: Sequence[str] | None = None,
) -> go.Figure | None:
    """Grouped bars separating deterministic and judge scores."""
    selected = _selected_models(analysis, models)
    deterministic = [model.deterministic_score for model in selected]
    judge = [model.judge_score for model in selected]
    if all(value is None for value in deterministic + judge):
        return None
    figure = go.Figure()
    figure.add_bar(
        name="Deterministic",
        x=[model.model_tag for model in selected],
        y=deterministic,
        marker_color=DETERMINISTIC_COLOR,
    )
    figure.add_bar(
        name="LLM judge",
        x=[model.model_tag for model in selected],
        y=judge,
        marker_color=JUDGE_COLOR,
    )
    figure.update_layout(barmode="group")
    figure.update_yaxes(range=[0, 105])
    return _apply(figure, y_title="Normalized score (0-100)")


def quality_categories(
    categories: Sequence[CategoryQuality],
) -> go.Figure | None:
    """Grouped bars of per-category deterministic, judge, and composite scores."""
    if not categories:
        return None
    if all(
        category.composite_score is None
        and category.deterministic_score is None
        and category.judge_score is None
        for category in categories
    ):
        return None
    count = len(categories)
    figure = go.Figure()
    figure.add_bar(
        name="Deterministic",
        x=[category.category.replace("_", " ") for category in categories],
        y=[category.deterministic_score for category in categories],
        marker_color=DETERMINISTIC_COLOR,
    )
    figure.add_bar(
        name="LLM judge",
        x=[category.category.replace("_", " ") for category in categories],
        y=[category.judge_score for category in categories],
        marker_color=JUDGE_COLOR,
    )
    figure.add_bar(
        name="Composite",
        x=[category.category.replace("_", " ") for category in categories],
        y=[category.composite_score for category in categories],
        marker_color=COMPOSITE_COLOR,
    )
    figure.update_layout(barmode="group")
    figure.update_yaxes(range=[0, 105])
    _apply(figure, y_title="Score (0-100)")
    if count > 4:
        figure.update_xaxes(tickangle=-25)
    return figure


def evaluation_type_rates(
    rows: Sequence[EvaluationBreakdownView],
    models: Sequence[str] | None = None,
    evaluation_types: Sequence[str] | None = None,
) -> go.Figure | None:
    """Grouped bars of pass rate by deterministic evaluation method."""
    allowed_models = set(models) if models else None
    allowed_types = set(evaluation_types) if evaluation_types else None
    values: list[dict[str, Any]] = []
    for row in rows:
        if allowed_models and row.model_tag not in allowed_models:
            continue
        if allowed_types and row.evaluation_type not in allowed_types:
            continue
        if row.evaluated_responses <= 0:
            continue
        values.append(
            {
                "model": row.model_tag,
                "evaluation": EVALUATION_LABELS.get(
                    row.evaluation_type or "unknown",
                    (row.evaluation_type or "unknown").replace("_", " ").title(),
                ),
                "pass_rate": row.passed_responses / row.evaluated_responses,
                "evaluated": row.evaluated_responses,
            }
        )
    if not values:
        return None
    frame = pd.DataFrame(values)
    colors = model_color_map(frame["model"].unique())
    figure = px.bar(
        frame,
        x="evaluation",
        y="pass_rate",
        color="model",
        barmode="group",
        color_discrete_map=colors,
        labels={"evaluation": "", "pass_rate": "Pass rate", "model": "Model"},
        hover_data={"evaluated": True, "pass_rate": ":.1%"},
    )
    figure.update_yaxes(tickformat=".0%", range=[0, 1.05])
    return _apply(figure, y_title="Pass rate")


def judge_scores(
    analysis: ExperimentAnalytics,
    models: Sequence[str] | None = None,
) -> go.Figure | None:
    """Bars of mean judge scores with coverage shown on hover."""
    selected = _selected_models(analysis, models)
    values = [model.judge_score for model in selected]
    if all(value is None for value in values):
        return None
    figure = go.Figure(
        go.Bar(
            x=[model.model_tag for model in selected],
            y=values,
            marker_color=JUDGE_COLOR,
            customdata=[model.quality.judge_coverage for model in selected],
            hovertemplate=(
                "%{x}<br>Judge score %{y:.1f}<br>"
                "Judge coverage %{customdata:.0%}<extra></extra>"
            ),
        )
    )
    figure.update_yaxes(range=[0, 105])
    return _apply(figure, y_title="Mean judge score (0-100)")


def hardware_avg_peak(
    summary_rows: Sequence[Mapping[str, object]],
    average_key: str,
    peak_key: str,
    label: str,
    unit: str,
    *,
    divisor: float = 1.0,
) -> go.Figure | None:
    """Grouped bars of average and peak hardware usage per model."""
    if not summary_rows:
        return None
    frame = pd.DataFrame(summary_rows)
    if average_key not in frame or peak_key not in frame:
        return None
    averages = [
        None if pd.isna(value) else float(value) / divisor
        for value in frame[average_key]
    ]
    peaks = [
        None if pd.isna(value) else float(value) / divisor
        for value in frame[peak_key]
    ]
    if all(value is None for value in averages + peaks):
        return None
    figure = go.Figure()
    figure.add_bar(
        name="Average",
        x=frame["model_tag"],
        y=averages,
        marker_color=DETERMINISTIC_COLOR,
    )
    figure.add_bar(
        name="Peak",
        x=frame["model_tag"],
        y=peaks,
        marker_color="#f97316",
    )
    figure.update_layout(barmode="group")
    return _apply(figure, y_title=f"{label} ({unit})")


def hardware_series(
    rows: Sequence[HardwareUsageRow],
    value_key: str,
    label: str,
    unit: str,
    models: Sequence[str] | None = None,
    *,
    divisor: float = 1.0,
) -> go.Figure | None:
    """Line chart of per-response hardware values ordered by question."""
    allowed = set(models) if models else None
    filtered = [
        row
        for row in rows
        if (allowed is None or row.model_tag in allowed)
        and getattr(row, value_key) is not None
    ]
    if not filtered:
        return None
    figure = go.Figure()
    colors = model_color_map([row.model_tag for row in filtered])
    tags = sorted({row.model_tag for row in filtered})
    for tag in tags:
        items = [
            row
            for row in filtered
            if row.model_tag == tag
        ]
        items.sort(key=lambda row: (row.question_id, row.repetition))
        figure.add_trace(
            go.Scatter(
                x=[
                    f"{row.question_id} · r{row.repetition}"
                    for row in items
                ],
                y=[getattr(row, value_key) / divisor for row in items],
                mode="lines+markers",
                name=tag,
                line=dict(color=colors[tag], width=2),
                marker=dict(size=6),
                hovertemplate="%{x}<br>%{y:.2f} " + unit + "<extra>" + tag + "</extra>",
            )
        )
    return _apply(figure, y_title=f"{label} ({unit})", height=380)


def comparison_changes(
    comparison: ExperimentComparison,
    metrics: Sequence[str] | None = None,
) -> go.Figure | None:
    """Grouped bars of percent change per metric and model."""
    selected = set(metrics) if metrics else None
    values: list[dict[str, Any]] = []
    for model in comparison.models:
        for delta in model.metrics:
            if delta.percent_change is None:
                continue
            if selected and delta.metric not in selected:
                continue
            values.append(
                {
                    "model": model.model_tag,
                    "metric": METRIC_LABELS.get(
                        delta.metric,
                        delta.metric.replace("_", " ").title(),
                    ),
                    "change": delta.percent_change,
                }
            )
    if not values:
        return None
    frame = pd.DataFrame(values)
    colors = model_color_map(frame["model"].unique())
    figure = px.bar(
        frame,
        x="metric",
        y="change",
        color="model",
        barmode="group",
        color_discrete_map=colors,
        labels={"metric": "", "change": "Change vs baseline (%)", "model": "Model"},
    )
    figure.add_hline(y=0, line_width=1, line_color="#94a3b8")
    figure.update_xaxes(tickangle=-25)
    return _apply(figure, y_title="Change vs baseline (%)")


__all__ = [
    "EVALUATION_LABELS",
    "METRIC_LABELS",
    "cold_warm_comparison",
    "comparison_changes",
    "composite_ranking",
    "dimension_radar",
    "evaluation_type_rates",
    "hardware_avg_peak",
    "hardware_series",
    "judge_scores",
    "latency_distribution",
    "model_color_map",
    "prompt_speed_comparison",
    "quality_categories",
    "quality_overall",
    "throughput_comparison",
    "ttft_comparison",
]
