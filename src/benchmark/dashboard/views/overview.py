"""
Overview page.

Responsibilities:
- Show registered models, completed experiments, and benchmark question counts.
- Show the latest experiment and its coverage.
- Show the overall ranking and the best model for each workload profile.
- Show aggregate performance statistics computed from stored responses.
"""

from __future__ import annotations

import streamlit as st

from benchmark.analytics import describe
from benchmark.core.exceptions import BenchmarkError
from benchmark.dashboard import charts, data, ui


def render(context: data.DashboardContext) -> None:
    """Render the overview page."""
    ui.page_header(
        "Overview",
        "Stored benchmark evidence, model ranking, and workload winners.",
    )
    views = data.history(context)
    latest = views[0] if views else None
    models = data.registered_models(context)
    completed = [view for view in views if view.status == "completed"]
    question_count = len(context.dataset.items) if context.dataset else 0

    ui.metric_row(
        [
            (
                "Registered models",
                ui.format_count(len(models)),
                "Models stored in the SQLite database.",
            ),
            (
                "Completed experiments",
                ui.format_count(len(completed)),
                "Experiments whose orchestrator run finished.",
            ),
            (
                "Benchmark questions",
                ui.format_count(question_count),
                "Items in the loaded, hash-validated dataset.",
            ),
            (
                "Latest experiment",
                latest.status.title() if latest else "—",
                latest.experiment_id if latest else "No experiment has been created.",
            ),
        ]
    )

    if context.dataset_error:
        st.warning(f"The dataset could not be loaded: {context.dataset_error}")

    if latest is None:
        ui.empty_state(
            "No experiments yet",
            "Open Run New Experiment to launch the first persisted benchmark run.",
        )
        return

    with st.container(border=True):
        columns = st.columns(4)
        columns[0].markdown(f"**Experiment**  \n`{latest.experiment_id}`")
        columns[1].markdown(
            f"**Dataset**  \n{latest.dataset_name} {latest.dataset_version}"
        )
        columns[2].markdown(
            f"**Created**  \n{latest.created_at.astimezone():%Y-%m-%d %H:%M}"
        )
        columns[3].markdown(
            f"**Stored rows**  \n{latest.run_count} runs · "
            f"{latest.response_count} responses · "
            f"{latest.evaluation_count} evaluations"
        )

    analysis = None
    analysis_error: str | None = None
    try:
        with st.spinner("Computing experiment analytics…"):
            analysis = data.analysis(
                context,
                latest.experiment_id,
                profile=context.settings.benchmark.default_scoring_profile,
            )
    except BenchmarkError as exc:
        analysis_error = str(exc)

    if analysis is None:
        st.info(
            "The latest experiment has no measured responses for the default "
            f"profile yet: {analysis_error}"
        )
        return

    try:
        rows = data.measured_rows(context, latest.experiment_id)
    except BenchmarkError:
        rows = []
    successful_latency = [
        row.latency_seconds
        for row in rows
        if row.response_status == "success" and row.latency_seconds is not None
    ]
    successful_rate = [
        row.output_tokens_per_second
        for row in rows
        if row.response_status == "success"
        and row.output_tokens_per_second is not None
    ]
    latency_stats = describe(successful_latency)
    throughput_stats = describe(successful_rate)
    success_total = sum(
        model.statistics.successful_responses for model in analysis.models
    )
    measured_total = sum(
        model.statistics.measured_responses for model in analysis.models
    )
    success_rate = success_total / measured_total if measured_total else None

    st.write("")
    ui.section(
        "Important performance statistics",
        "Computed from the latest experiment under the default scoring profile.",
    )
    ui.metric_row(
        [
            (
                "Mean latency",
                ui.format_seconds(latency_stats.mean),
                "Mean total latency across successful responses.",
            ),
            (
                "P95 latency",
                ui.format_seconds(latency_stats.p95),
                "95th percentile total latency across successful responses.",
            ),
            (
                "Mean output rate",
                ui.format_rate(throughput_stats.mean),
                "Mean Ollama-reported output tokens per second.",
            ),
            (
                "Response success",
                ui.format_percent(success_rate),
                f"{success_total} successful of {measured_total} measured responses.",
            ),
        ]
    )
    st.caption(
        f"{analysis.evaluated_responses} of {analysis.measured_responses} measured "
        f"responses are evaluated · profile `{analysis.profile}` · "
        f"generated {analysis.generated_at.astimezone():%Y-%m-%d %H:%M}"
    )

    st.write("")
    ui.section(
        "Overall model ranking",
        "Composite scores use the configured profile and exclude smoke and "
        "cold-start runs by default.",
    )
    ranked = sorted(
        analysis.models,
        key=lambda model: model.composite_score
        if model.composite_score is not None
        else -1.0,
        reverse=True,
    )
    ranking = ui.clean_frame(
        [
            {
                "Model": model.model_tag,
                "Composite": model.composite_score,
                "Quality": model.quality_score,
                "P95 latency (s)": model.statistics.latency.p95,
                "Throughput (tok/s)": (
                    model.statistics.output_tokens_per_second.mean
                ),
                "Success rate": (
                    model.statistics.success_rate * 100.0
                    if model.statistics.success_rate is not None
                    else None
                ),
                "Coverage": model.quality.quality_coverage * 100.0,
            }
            for model in ranked
        ]
    )
    st.dataframe(
        ranking,
        width="stretch",
        hide_index=True,
        column_config={
            "Composite": st.column_config.NumberColumn(format="%.1f"),
            "Quality": st.column_config.NumberColumn(format="%.1f"),
            "P95 latency (s)": st.column_config.NumberColumn(format="%.2f"),
            "Throughput (tok/s)": st.column_config.NumberColumn(format="%.1f"),
            "Success rate": st.column_config.ProgressColumn(
                min_value=0.0, max_value=100.0, format="%.0f%%"
            ),
            "Coverage": st.column_config.ProgressColumn(
                min_value=0.0, max_value=100.0, format="%.0f%%"
            ),
        },
    )
    scored_models = [
        model for model in analysis.models if model.composite_score is not None
    ]
    if len(scored_models) >= 2:
        chart = charts.composite_ranking(analysis)
        if chart is not None:
            st.plotly_chart(
                chart,
                width="stretch",
                config={"displayModeBar": False},
            )
    else:
        st.caption(
            "A ranking chart needs at least two models with composite scores; "
            "the table above remains the authoritative ranking."
        )

    st.write("")
    ui.section(
        "Best model by workload",
        "Each workload is ranked independently using the configured profile weights.",
    )
    try:
        with st.spinner("Ranking models for each workload…"):
            recommendation_sets = data.recommendations(
                context, latest.experiment_id
            )
    except BenchmarkError as exc:
        st.warning(f"Workload recommendations are unavailable: {exc}")
        recommendation_sets = []
    winners: list[dict[str, object]] = []
    for result in recommendation_sets:
        if result.available and result.recommendations:
            winner = result.recommendations[0]
            winners.append(
                {
                    "Workload": result.workload,
                    "Best model": winner.model_tag,
                    "Score": ui.format_score(winner.composite_score),
                    "Confidence": winner.confidence,
                    "Tie": ", ".join(winner.tied_with) if winner.tied else "",
                }
            )
        else:
            winners.append(
                {
                    "Workload": result.workload,
                    "Best model": "Not available",
                    "Score": "—",
                    "Confidence": "",
                    "Tie": result.reason or "",
                }
            )
    if winners:
        st.dataframe(
            ui.clean_frame(winners),
            width="stretch",
            hide_index=True,
        )
    ui.limitations(analysis.limitations)

    st.caption(
        "All values on this page are read from the database and computed by the "
        "analytics engine; no benchmark result is hardcoded."
    )


__all__ = ["render"]
