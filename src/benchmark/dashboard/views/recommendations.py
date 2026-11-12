"""
Recommendations page.

Responsibilities:
- Show workload-specific model recommendations and their explanations.
- Show the weights, evidence counts, strengths, weaknesses, and limitations.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from benchmark.core.exceptions import BenchmarkError
from benchmark.dashboard import data, ui


def render(context: data.DashboardContext) -> None:
    """Render the recommendations page."""
    ui.page_header(
        "Recommendations",
        "Workload-specific rankings computed live from stored experiment evidence.",
    )
    view = ui.select_experiment(context, "recommendation_experiment")
    if view is None:
        return
    profile = ui.select_profile(
        context,
        "recommendation_profile",
        help_text=(
            "Each profile re-weights quality, latency, throughput, efficiency, and "
            "reliability for its workload."
        ),
    )
    st.caption(ui.profile_description(context, profile))
    try:
        with st.spinner("Ranking models for this workload…"):
            results = data.recommendations(
                context,
                view.experiment_id,
                workloads=[profile],
            )
    except BenchmarkError as exc:
        st.warning(f"Recommendations are unavailable: {exc}")
        return
    if not results:
        ui.empty_state(
            "No recommendation",
            "The analytics engine returned no result for this workload.",
        )
        return
    result = results[0]
    weights = context.settings.benchmark.workload_profiles.get(profile)
    with st.expander("Profile weights and evidence basis", expanded=False):
        if weights is not None:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Dimension": name.title(),
                            "Weight": value,
                        }
                        for name, value in weights.weights.items()
                    ]
                ),
                width="stretch",
                hide_index=True,
                column_config={
                    "Weight": st.column_config.NumberColumn(format="%.0f"),
                },
            )
            if weights.category_weights:
                st.caption(
                    "Category weights: "
                    + ", ".join(
                        f"{name} {value:.0%}"
                        for name, value in weights.category_weights.items()
                    )
                )
        st.caption(
            f"Methodology version {result.methodology_version} · "
            f"{result.evaluated_models} evaluated models · "
            f"{result.comparable_models} comparable models."
        )

    if not result.available:
        st.warning(f"No defensible recommendation: {result.reason}")
        if result.excluded_models:
            st.dataframe(
                pd.DataFrame(
                    [
                        {"Model": tag, "Excluded because": reason}
                        for tag, reason in result.excluded_models.items()
                    ]
                ),
                width="stretch",
                hide_index=True,
            )
        ui.limitations(result.limitations)
        return

    top = result.recommendations[0]
    ui.metric_row(
        [
            (
                "Recommended model",
                top.model_tag,
                top.rationale,
            ),
            (
                "Composite score",
                ui.format_score(top.composite_score),
                "Weighted score after the 20% failure-rate gate.",
            ),
            (
                "Confidence",
                top.confidence.title(),
                "Confidence reflects evidence volume, dimension coverage, and ties.",
            ),
            (
                "Evidence",
                ui.format_count(top.evidence_responses),
                f"Quality coverage {ui.format_percent(top.quality_coverage)}.",
            ),
        ]
    )
    if top.tied:
        st.info(
            "This workload has a practical tie within two composite points: "
            + ", ".join(top.tied_with)
        )

    for recommendation in result.recommendations:
        st.write("")
        with st.container(border=True):
            heading = st.columns([1, 3, 1, 1])
            heading[0].markdown(f"### #{recommendation.rank}")
            heading[1].markdown(f"**{recommendation.model_tag}**")
            heading[2].metric("Score", f"{recommendation.composite_score:.1f}")
            heading[3].metric("Confidence", recommendation.confidence.title())
            st.markdown(recommendation.rationale)
            columns = st.columns(2)
            with columns[0]:
                st.markdown("**Strengths**")
                if recommendation.strengths:
                    for strength in recommendation.strengths:
                        st.markdown(f"- {strength}")
                else:
                    st.caption("No leading dimension was identified.")
            with columns[1]:
                st.markdown("**Weaknesses**")
                if recommendation.weaknesses:
                    for weakness in recommendation.weaknesses:
                        st.markdown(f"- {weakness}")
                else:
                    st.caption("No trailing dimension was identified.")
            metric_rows = [
                ("Quality", recommendation.metrics.get("quality_score"), "%.1f"),
                (
                    "P95 latency (s)",
                    recommendation.metrics.get("p95_latency_seconds"),
                    "%.2f",
                ),
                (
                    "Mean TTFT (s)",
                    recommendation.metrics.get("mean_ttft_seconds"),
                    "%.2f",
                ),
                (
                    "Throughput (tok/s)",
                    recommendation.metrics.get("mean_output_tokens_per_second"),
                    "%.1f",
                ),
                (
                    "Success rate",
                    recommendation.metrics.get("success_rate"),
                    "%.1f%%",
                ),
                (
                    "RAM avg (MiB)",
                    recommendation.metrics.get("ram_avg_mib"),
                    "%.0f",
                ),
                (
                    "VRAM avg (MiB)",
                    recommendation.metrics.get("vram_avg_mib"),
                    "%.0f",
                ),
                (
                    "Quality coverage",
                    recommendation.metrics.get("quality_coverage"),
                    "%.1f%%",
                ),
            ]
            display_rows = []
            for label, value, pattern in metric_rows:
                if value is None:
                    display_rows.append({"Metric": label, "Value": "—"})
                elif pattern.endswith("%%"):
                    display_rows.append(
                        {"Metric": label, "Value": f"{value * 100:.1f}%"}
                    )
                else:
                    display_rows.append(
                        {"Metric": label, "Value": pattern % value}
                    )
            st.dataframe(
                pd.DataFrame(display_rows),
                width="stretch",
                hide_index=True,
            )
            if recommendation.tied:
                st.caption(
                    "Tied within two points with: "
                    + ", ".join(recommendation.tied_with)
                )
            if recommendation.missing_dimensions:
                st.caption(
                    "Missing scoring dimensions: "
                    + ", ".join(recommendation.missing_dimensions)
                )
            ui.limitations(recommendation.limitations)

    if result.excluded_models:
        with st.expander("Excluded models"):
            st.dataframe(
                pd.DataFrame(
                    [
                        {"Model": tag, "Excluded because": reason}
                        for tag, reason in result.excluded_models.items()
                    ]
                ),
                width="stretch",
                hide_index=True,
            )
    ui.limitations(result.limitations)
    st.caption(
        "Recommendations are recomputed from the database each time the page "
        "renders. They are not hardcoded and are not persisted by the dashboard."
    )


__all__ = ["render"]
