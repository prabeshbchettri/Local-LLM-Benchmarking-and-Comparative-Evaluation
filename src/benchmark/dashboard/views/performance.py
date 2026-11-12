"""
Performance analysis page.

Responsibilities:
- Show latency distributions, TTFT, output throughput, and prompt processing speed.
- Compare cold-start and warm-start performance from the same stored rows.
"""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from benchmark.core.exceptions import BenchmarkError
from benchmark.dashboard import charts, data, ui


def _show(
    column: st.delta_generator.DeltaGenerator,
    figure: go.Figure | None,
    title: str,
    empty_message: str,
) -> None:
    with column:
        st.markdown(f"**{title}**")
        if figure is None:
            st.info(empty_message)
        else:
            st.plotly_chart(
                figure,
                width="stretch",
                config={"displayModeBar": False},
            )


def render(context: data.DashboardContext) -> None:
    """Render the performance analysis page."""
    ui.page_header(
        "Performance Analysis",
        "Latency, time to first token, throughput, and cold-start behavior.",
    )
    view = ui.select_experiment(
        context,
        "performance_experiment",
    )
    if view is None:
        return
    controls = st.columns([2, 2, 1])
    with controls[0]:
        profile = ui.select_profile(context, "performance_profile")
    with controls[1]:
        include_cold = st.toggle(
            "Include cold-start runs in comparisons",
            value=False,
            key="performance_cold",
            help=(
                "Cold-start requests unload the model before every call. They are "
                "excluded from default analytics and can be compared separately."
            ),
        )
    try:
        with st.spinner("Computing performance statistics…"):
            analysis = data.analysis(
                context,
                view.experiment_id,
                profile=profile,
                include_cold_start=include_cold,
            )
    except BenchmarkError as exc:
        st.warning(f"No measured responses match these filters: {exc}")
        return
    models = ui.select_models(
        [model.model_tag for model in analysis.models],
        "performance_models",
    )

    try:
        with st.spinner("Loading stored response timings…"):
            rows = data.measured_rows(
                context,
                view.experiment_id,
                include_cold_start=include_cold,
            )
            rows_with_cold = (
                rows
                if include_cold
                else data.measured_rows(
                    context,
                    view.experiment_id,
                    include_cold_start=True,
                )
            )
    except BenchmarkError as exc:
        st.warning(f"Stored response rows are unavailable: {exc}")
        return

    ui.metric_row(
        [
            (
                "Measured responses",
                ui.format_count(analysis.measured_responses),
                "Successful and failed measured responses in this view.",
            ),
            (
                "P95 latency",
                ui.format_seconds(
                    max(
                        (
                            model.statistics.latency.p95
                            for model in analysis.models
                            if model.statistics.latency.p95 is not None
                        ),
                        default=None,
                    )
                ),
                "Highest per-model P95 latency in the selected view.",
            ),
            (
                "Mean output rate",
                ui.format_rate(
                    max(
                        (
                            model.statistics.output_tokens_per_second.mean
                            for model in analysis.models
                            if model.statistics.output_tokens_per_second.mean
                            is not None
                        ),
                        default=None,
                    )
                ),
                "Highest per-model mean output rate.",
            ),
            (
                "Cold-start data",
                "Yes"
                if any(row.cold_start for row in rows_with_cold)
                else "No",
                "Whether stored cold-start responses exist for this experiment.",
            ),
        ]
    )

    st.write("")
    left, right = st.columns(2)
    _show(
        left,
        charts.latency_distribution(rows, models),
        "Latency distribution",
        "No successful responses with measured latency match this view.",
    )
    _show(
        right,
        charts.ttft_comparison(analysis, models),
        "Time to first token",
        "TTFT was not measured for the selected models.",
    )
    left, right = st.columns(2)
    _show(
        left,
        charts.throughput_comparison(analysis, models),
        "Output tokens per second",
        "Output throughput was not measured for the selected models.",
    )
    _show(
        right,
        charts.prompt_speed_comparison(analysis, models),
        "Prompt processing speed",
        "Prompt processing speed was not measured for the selected models.",
    )

    st.write("")
    ui.section(
        "Cold start versus warm start",
        "Successful requests only. Cold-start runs unload the model before each call.",
    )
    if include_cold:
        st.caption(
            "Cold-start runs are currently included in the charts above; these "
            "comparisons use the same stored rows."
        )
    if not any(row.cold_start for row in rows_with_cold):
        st.info(
            "No cold-start responses are stored for this experiment. Run an "
            "experiment with cold-start enabled to populate this comparison."
        )
    else:
        left, right = st.columns(2)
        _show(
            left,
            charts.cold_warm_comparison(
                rows_with_cold, "latency_seconds", "Total latency (s)", models
            ),
            "Cold vs warm total latency",
            "No successful cold or warm responses match this view.",
        )
        _show(
            right,
            charts.cold_warm_comparison(
                rows_with_cold, "ttft_seconds", "TTFT (s)", models
            ),
            "Cold vs warm TTFT",
            "No successful cold or warm TTFT values match this view.",
        )

    st.write("")
    latency_table = [
        {
            "Model": model.model_tag,
            "Count": model.statistics.latency.count,
            "Mean (s)": model.statistics.latency.mean,
            "Median (s)": model.statistics.latency.median,
            "P95 (s)": model.statistics.latency.p95,
            "TTFT mean (s)": model.statistics.ttft.mean,
            "Output tok/s": model.statistics.output_tokens_per_second.mean,
            "Prompt tok/s": model.statistics.prompt_tokens_per_second.mean,
        }
        for model in analysis.models
        if model.model_tag in models
    ]
    if latency_table:
        st.dataframe(
            ui.clean_frame(latency_table),
            width="stretch",
            hide_index=True,
            column_config={
                "Count": st.column_config.NumberColumn(format="%d"),
                "Mean (s)": st.column_config.NumberColumn(format="%.3f"),
                "Median (s)": st.column_config.NumberColumn(format="%.3f"),
                "P95 (s)": st.column_config.NumberColumn(format="%.3f"),
                "TTFT mean (s)": st.column_config.NumberColumn(format="%.3f"),
                "Output tok/s": st.column_config.NumberColumn(format="%.1f"),
                "Prompt tok/s": st.column_config.NumberColumn(format="%.1f"),
            },
        )

    ui.limitations(
        analysis.limitations
        + [
            limitation
            for model in analysis.models
            if model.model_tag in models
            for limitation in model.limitations
        ]
    )


__all__ = ["render"]
