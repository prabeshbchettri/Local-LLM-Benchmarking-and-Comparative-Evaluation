"""
Reports page.

Responsibilities:
- Select a stored experiment and prepare downloadable report files.
- Refuse to present non-completed experiments as final results.
- Render CSV, JSON, HTML, and PDF download controls through the reporting service.
"""

from __future__ import annotations

import streamlit as st

from benchmark.dashboard import data, exports, ui


def render(context: data.DashboardContext) -> None:
    """Render the reports and export page."""
    ui.page_header(
        "Reports",
        "Reproducible CSV, JSON, HTML, and PDF exports built from stored rows.",
    )
    view = ui.select_experiment(
        context,
        "reports_experiment",
        help_text=(
            "Only completed experiments can be exported as final results. "
            "Incomplete experiments remain visible but are not exported."
        ),
    )
    if view is None:
        return

    if view.status == "completed":
        ui.metric_row(
            [
                (
                    "Runs",
                    ui.format_count(view.run_count),
                    "Stored model runs in this experiment.",
                ),
                (
                    "Responses",
                    ui.format_count(view.response_count),
                    "Stored measured and warm-up responses.",
                ),
                (
                    "Evaluations",
                    ui.format_count(view.evaluation_count),
                    "Stored deterministic and judge evaluations.",
                ),
                (
                    "Dataset",
                    f"{view.dataset_name} {view.dataset_version}",
                    "Dataset version recorded with the experiment.",
                ),
            ]
        )
        st.caption(
            f"Dataset hash `{view.dataset_hash[:16]}…` · "
            f"config hash `{view.config_hash[:16]}…` · "
            f"created {view.created_at.astimezone():%Y-%m-%d %H:%M}"
        )
        st.write("")

    exports.render_export_panel(context, view)
    ui.note(
        "Every exported value is computed from the database by the analytics "
        "engine. Missing measurements are exported as missing, never as zero."
    )


__all__ = ["render"]
