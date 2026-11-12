"""
Experiment history page.

Responsibilities:
- List stored experiments with status and row counts.
- Compare completed experiments per model with absolute and percent deltas.
"""

from __future__ import annotations

import streamlit as st

from benchmark.core.exceptions import BenchmarkError
from benchmark.dashboard import charts, data, ui


def render(context: data.DashboardContext) -> None:
    """Render the experiment history page."""
    ui.page_header(
        "Experiment History",
        "Stored experiments and per-model metric deltas against a baseline.",
    )
    views = data.history(context)
    if not views:
        ui.empty_state(
            "No experiment history",
            "Experiments appear here after the orchestrator creates a database record.",
        )
        return
    st.dataframe(
        ui.clean_frame(
            [
                {
                    "Experiment": view.experiment_id,
                    "Name": view.name or "",
                    "Status": view.status,
                    "Created": view.created_at.astimezone().strftime(
                        "%Y-%m-%d %H:%M"
                    ),
                    "Dataset": f"{view.dataset_name} {view.dataset_version}",
                    "Runs": view.run_count,
                    "Responses": view.response_count,
                    "Evaluations": view.evaluation_count,
                }
                for view in views
            ]
        ),
        width="stretch",
        hide_index=True,
        column_config={
            "Runs": st.column_config.NumberColumn(format="%d"),
            "Responses": st.column_config.NumberColumn(format="%d"),
            "Evaluations": st.column_config.NumberColumn(format="%d"),
        },
    )

    completed = [view for view in views if view.status == "completed"]
    if len(completed) < 2:
        st.info(
            "At least two completed experiments are required for a historical "
            "comparison."
        )
        return

    st.write("")
    ui.section(
        "Compare completed experiments",
        "Deltas are computed on the same scoring profile with the same filters.",
    )
    selector = st.columns([2, 3, 1])
    lookup = {view.experiment_id: view for view in completed}
    with selector[0]:
        baseline_id = st.selectbox(
            "Baseline experiment",
            options=list(lookup),
            format_func=lambda value: ui.experiment_label(lookup[value]),
            key="history_baseline",
        )
    candidates = [view.experiment_id for view in completed if view.experiment_id != baseline_id]
    with selector[1]:
        candidate_ids = st.multiselect(
            "Candidate experiments",
            options=candidates,
            default=candidates[:1],
            format_func=lambda value: ui.experiment_label(lookup[value]),
            key="history_candidates",
        )
    with selector[2]:
        profile = ui.select_profile(
            context,
            "history_profile",
            label="Profile",
        )
    if not candidate_ids:
        st.info("Select at least one candidate experiment.")
        return
    try:
        with st.spinner("Comparing stored experiments…"):
            comparison = data.comparison(
                context,
                baseline_id,
                candidate_ids,
                profile=profile,
            )
    except BenchmarkError as exc:
        st.warning(f"The comparison could not be computed: {exc}")
        return

    for item in comparison.comparisons:
        st.write("")
        with st.container(border=True):
            st.markdown(
                f"**{item.candidate_experiment_id}** versus baseline "
                f"`{item.baseline_experiment_id}`"
            )
            if item.baseline_profile or item.candidate_profile:
                st.caption(
                    f"Profile: {item.baseline_profile or 'unknown'} → "
                    f"{item.candidate_profile or 'unknown'}"
                )
            if not item.models:
                st.info(
                    "No model tags are shared by these experiments, so no metric "
                    "deltas can be computed."
                )
            else:
                delta_rows = []
                for model in item.models:
                    for metric in model.metrics:
                        delta_rows.append(
                            {
                                "Model": model.model_tag,
                                "Metric": charts.METRIC_LABELS.get(
                                    metric.metric,
                                    metric.metric.replace("_", " ").title(),
                                ),
                                "Baseline": metric.baseline,
                                "Candidate": metric.candidate,
                                "Delta": metric.delta,
                                "Change %": metric.percent_change,
                                "Available": metric.available,
                            }
                        )
                st.dataframe(
                    ui.clean_frame(delta_rows),
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "Baseline": st.column_config.NumberColumn(format="%.4f"),
                        "Candidate": st.column_config.NumberColumn(format="%.4f"),
                        "Delta": st.column_config.NumberColumn(format="%+.4f"),
                        "Change %": st.column_config.NumberColumn(format="%+.1f%%"),
                    },
                )
                chart = charts.comparison_changes(item)
                if chart is None:
                    st.info("No metric has values in both experiments.")
                else:
                    st.plotly_chart(
                        chart,
                        width="stretch",
                        config={"displayModeBar": False},
                    )
                ui.limitations(item.limitations)


__all__ = ["render"]
