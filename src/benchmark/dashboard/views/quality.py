"""
Quality evaluation page.

Responsibilities:
- Show category-wise accuracy and coding, JSON, and instruction-following pass rates.
- Show LLM judge scores and coverage.
- Separate deterministic and subjective evaluation.
"""

from __future__ import annotations

import streamlit as st

from benchmark.core.exceptions import BenchmarkError
from benchmark.dashboard import charts, data, ui


def _score_text(value: float | None, digits: int = 1) -> str:
    """Format a score or an explicit dash for a missing measurement."""
    return "—" if value is None else f"{value:.{digits}f}"


def render(context: data.DashboardContext) -> None:
    """Render the quality evaluation page."""
    ui.page_header(
        "Quality Evaluation",
        "Deterministic checks and blind LLM judge scores stored with each response.",
    )
    view = ui.select_experiment(context, "quality_experiment")
    if view is None:
        return
    controls = st.columns([2, 2])
    with controls[0]:
        profile = ui.select_profile(context, "quality_profile")
    try:
        with st.spinner("Computing quality statistics…"):
            analysis = data.analysis(context, view.experiment_id, profile=profile)
    except BenchmarkError as exc:
        st.warning(f"No evaluated responses are available: {exc}")
        return
    with controls[1]:
        models = ui.select_models(
            [model.model_tag for model in analysis.models],
            "quality_models",
        )
    selected_models = [
        model for model in analysis.models if model.model_tag in models
    ]
    try:
        with st.spinner("Loading evaluation methods…"):
            breakdown = data.evaluation_breakdown(context, view.experiment_id)
    except BenchmarkError:
        breakdown = []

    judge_total = sum(
        model.quality.judge_evaluated_responses for model in selected_models
    )
    evaluated_total = sum(
        model.quality.evaluated_responses for model in selected_models
    )
    judge_coverage = judge_total / evaluated_total if evaluated_total else None
    best = max(
        (
            model
            for model in selected_models
            if model.composite_score is not None
        ),
        key=lambda model: model.composite_score or 0.0,
        default=None,
    )
    ui.metric_row(
        [
            (
                "Evaluated responses",
                ui.format_count(evaluated_total),
                "Responses with a stored deterministic or composite score.",
            ),
            (
                "Quality coverage",
                ui.format_percent(
                    evaluated_total
                    / max(
                        sum(
                            model.statistics.measured_responses
                            for model in selected_models
                        ),
                        1,
                    )
                ),
                "Evaluated responses divided by measured responses.",
            ),
            (
                "Judge coverage",
                ui.format_percent(judge_coverage),
                "Evaluated responses that also have a judge score.",
            ),
            (
                "Best composite",
                ui.format_score(best.composite_score if best else None),
                best.model_tag if best else "No composite score is available.",
            ),
        ]
    )

    st.write("")
    ui.section(
        "Category-wise accuracy",
        "The primary quality score gives every measured category equal influence.",
    )
    category_model = st.selectbox(
        "Category breakdown model",
        options=[model.model_tag for model in selected_models],
        key="quality_category_model",
    )
    chosen = next(
        model for model in selected_models if model.model_tag == category_model
    )
    if chosen.quality.categories:
        category_rows = ui.clean_frame(
            [
                {
                    "Category": category.category.replace("_", " ").title(),
                    "Measured": category.measured_responses,
                    "Evaluated": category.evaluated_responses,
                    "Deterministic": _score_text(category.deterministic_score),
                    "LLM judge": _score_text(category.judge_score),
                    "Composite": _score_text(category.composite_score),
                    "Coverage": category.quality_coverage * 100.0,
                }
                for category in chosen.quality.categories
            ]
        )
        st.dataframe(
            category_rows,
            width="stretch",
            hide_index=True,
            column_config={
                "Coverage": st.column_config.ProgressColumn(
                    min_value=0.0, max_value=100.0, format="%.0f%%"
                ),
            },
        )
        chart = charts.quality_categories(chosen.quality.categories)
        if chart is not None:
            st.plotly_chart(
                chart,
                width="stretch",
                config={"displayModeBar": False},
            )
    else:
        st.info("No category scores are stored for this model.")

    st.write("")
    ui.section(
        "Deterministic versus subjective",
        "Deterministic checks and blind judge scores are kept separate and never imputed.",
    )
    left, right = st.columns(2)
    with left:
        chart = charts.quality_overall(analysis, models)
        if chart is None:
            st.info("No overall deterministic or judge scores are available.")
        else:
            st.plotly_chart(
                chart,
                width="stretch",
                config={"displayModeBar": False},
            )
    with right:
        judge_chart = charts.judge_scores(analysis, models)
        if judge_chart is None:
            st.info("No judge scores are stored for the selected models.")
        else:
            st.plotly_chart(
                judge_chart,
                width="stretch",
                config={"displayModeBar": False},
            )

    st.write("")
    ui.section(
        "Coding, JSON, and instruction-following pass rates",
        "Pass rates come from stored deterministic evaluation rows.",
    )
    requested_types = ["code_tests", "json_schema", "instruction_constraints"]
    rate_chart = charts.evaluation_type_rates(
        breakdown,
        models,
        requested_types,
    )
    if rate_chart is None:
        st.info(
            "No coding, JSON-validity, or instruction-following evaluations are "
            "stored for the selected models."
        )
    else:
        st.plotly_chart(
            rate_chart,
            width="stretch",
            config={"displayModeBar": False},
        )
    if breakdown:
        breakdown_frame = ui.clean_frame(
            [
                {
                    "Model": row.model_tag,
                    "Method": charts.EVALUATION_LABELS.get(
                        row.evaluation_type or "unknown",
                        (row.evaluation_type or "unknown").replace("_", " ").title(),
                    ),
                    "Evaluated": row.evaluated_responses,
                    "Passed": row.passed_responses,
                    "Pass rate": (
                        row.passed_responses / row.evaluated_responses * 100.0
                        if row.evaluated_responses
                        else None
                    ),
                    "Deterministic": _score_text(row.deterministic_score),
                    "LLM judge": _score_text(row.judge_score),
                    "Composite": _score_text(row.composite_score),
                }
                for row in breakdown
                if row.model_tag in models
            ]
        )
        st.dataframe(
            breakdown_frame,
            width="stretch",
            hide_index=True,
            column_config={
                "Pass rate": st.column_config.NumberColumn(format="%.1f%%"),
            },
        )

    st.write("")
    ui.section("Judge reliability and coverage")
    st.dataframe(
        ui.clean_frame(
            [
                {
                    "Model": model.model_tag,
                    "Evaluated": model.quality.evaluated_responses,
                    "Judge scored": model.quality.judge_evaluated_responses,
                    "Judge coverage": model.quality.judge_coverage * 100.0,
                    "Deterministic": _score_text(model.deterministic_score),
                    "LLM judge": _score_text(model.judge_score),
                    "Composite": _score_text(model.quality_score),
                }
                for model in selected_models
            ]
        ),
        width="stretch",
        hide_index=True,
        column_config={
            "Judge coverage": st.column_config.ProgressColumn(
                min_value=0.0, max_value=100.0, format="%.0f%%"
            ),
        },
    )
    if judge_coverage is not None and judge_coverage < 0.5:
        st.warning(
            "Most evaluated responses have no judge score; the subjective view is "
            "descriptive, not a ranking basis."
        )

    ui.limitations(
        analysis.limitations
        + [
            limitation
            for model in selected_models
            for limitation in model.quality.limitations
        ]
    )


__all__ = ["render"]
