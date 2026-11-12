"""
Model comparison page.

Responsibilities:
- Compare selected models side by side on latency, TTFT, throughput, quality,
  success rate, RAM, and VRAM.
- Use the analytics engine for all values and disclose missing measurements.
"""

from __future__ import annotations

import streamlit as st

from benchmark.core.exceptions import BenchmarkError
from benchmark.dashboard import charts, data, ui


def render(context: data.DashboardContext) -> None:
    """Render the model comparison page."""
    ui.page_header(
        "Model Comparison",
        "Side-by-side metrics for models measured in the same experiment.",
    )
    view = ui.select_experiment(
        context,
        "comparison_experiment",
        help_text="Comparisons are most meaningful within one experiment.",
    )
    if view is None:
        return
    profile = ui.select_profile(context, "comparison_profile")
    try:
        with st.spinner("Loading comparable model metrics…"):
            analysis = data.analysis(context, view.experiment_id, profile=profile)
    except BenchmarkError as exc:
        st.warning(f"No comparable measurements for this experiment: {exc}")
        return

    tags = [model.model_tag for model in analysis.models]
    selected = ui.select_models(tags, "comparison_models")
    models = [model for model in analysis.models if model.model_tag in selected]
    if not models:
        ui.empty_state(
            "No models selected",
            "Choose at least one model to compare.",
        )
        return

    best = max(
        (
            model
            for model in models
            if model.composite_score is not None
        ),
        key=lambda model: model.composite_score or 0.0,
        default=None,
    )
    measured_total = sum(
        model.statistics.measured_responses for model in models
    )
    evaluated_total = sum(
        model.quality.evaluated_responses for model in models
    )
    coverage = evaluated_total / measured_total if measured_total else None
    ui.metric_row(
        [
            ("Models compared", ui.format_count(len(models)), None),
            (
                "Measured responses",
                ui.format_count(measured_total),
                "Measured responses for the selected models.",
            ),
            (
                "Best composite",
                ui.format_score(best.composite_score if best else None),
                best.model_tag if best else "No model has a composite score.",
            ),
            (
                "Evaluated coverage",
                ui.format_percent(coverage),
                "Evaluated responses divided by measured responses.",
            ),
        ]
    )
    st.caption(
        f"Profile `{analysis.profile}` · {analysis.profile_description}"
    )

    st.write("")
    rows = []
    for model in models:
        statistics = model.statistics
        resources = model.resources
        vram = (
            resources.model_vram_avg_mib
            if resources.model_vram_available
            else resources.vram_avg_mib
        )
        rows.append(
            {
                "Model": model.model_tag,
                "Avg latency (s)": statistics.latency.mean,
                "Median latency (s)": statistics.latency.median,
                "P95 latency (s)": statistics.latency.p95,
                "Mean TTFT (s)": statistics.ttft.mean,
                "Output tok/s": statistics.output_tokens_per_second.mean,
                "Quality": model.quality_score,
                "Success rate": (
                    statistics.success_rate * 100.0
                    if statistics.success_rate is not None
                    else None
                ),
                "RAM avg (MiB)": resources.ram_avg_mib,
                "VRAM avg (MiB)": vram,
            }
        )
    st.dataframe(
        ui.clean_frame(rows),
        width="stretch",
        hide_index=True,
        column_config={
            "Avg latency (s)": st.column_config.NumberColumn(format="%.2f"),
            "Median latency (s)": st.column_config.NumberColumn(format="%.2f"),
            "P95 latency (s)": st.column_config.NumberColumn(format="%.2f"),
            "Mean TTFT (s)": st.column_config.NumberColumn(format="%.2f"),
            "Output tok/s": st.column_config.NumberColumn(format="%.1f"),
            "Quality": st.column_config.NumberColumn(format="%.1f"),
            "Success rate": st.column_config.ProgressColumn(
                min_value=0.0, max_value=100.0, format="%.0f%%"
            ),
            "RAM avg (MiB)": st.column_config.NumberColumn(format="%.0f"),
            "VRAM avg (MiB)": st.column_config.NumberColumn(format="%.0f"),
        },
    )
    st.caption(
        "VRAM uses model-specific Ollama VRAM when available and system VRAM "
        "otherwise. Hardware values are system-wide and include other processes."
    )

    st.write("")
    radar = charts.dimension_radar(analysis, selected)
    if radar is not None:
        ui.section(
            "Normalized scoring dimensions",
            "Relative dimensions are normalized across the models in this experiment.",
        )
        st.plotly_chart(
            radar,
            width="stretch",
            config={"displayModeBar": False},
        )
    else:
        st.info(
            "A dimension radar needs at least one model with all five measured "
            "dimensions."
        )

    missing = [
        f"{model.model_tag}: {', '.join(model.limitations)}"
        for model in models
        if model.limitations
    ]
    ui.limitations(analysis.limitations + missing)


__all__ = ["render"]
