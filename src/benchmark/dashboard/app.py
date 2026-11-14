"""
File: app.py

Description:
Streamlit entry point and page navigation for the benchmark dashboard.

Why this file exists:
One explicit navigation control keeps the ten pages discoverable and stable, and
one top-level error boundary keeps a backend failure from blanking the dashboard.

Responsibilities:
- Configure the page and build the configuration/database context.
- Render the sidebar navigation, runtime summary, and data refresh control.
- Dispatch to the selected page and show readable backend error messages.

Run with:
    streamlit run src/benchmark/dashboard/app.py
"""

from __future__ import annotations

import os

import streamlit as st

from benchmark.core.exceptions import BenchmarkError
from benchmark.dashboard import data, runner, ui
from benchmark.dashboard.views import (
    comparison,
    hardware,
    history,
    inspection,
    overview,
    performance,
    quality,
    recommendations,
    reports,
    run_experiment,
)

PAGES = {
    "Overview": overview.render,
    "Model Comparison": comparison.render,
    "Performance Analysis": performance.render,
    "Quality Evaluation": quality.render,
    "Hardware Monitoring": hardware.render,
    "Response Inspection": inspection.render,
    "Experiment History": history.render,
    "Reports": reports.render,
    "Recommendations": recommendations.render,
    "Run New Experiment": run_experiment.render,
}


def _sidebar(context: data.DashboardContext) -> str:
    """Render the sidebar and return the selected page label."""
    with st.sidebar:
        st.markdown("### Local LLM Benchmark")
        st.caption("Phase 09 · interactive dashboard")
        page = st.radio(
            "Navigation",
            options=list(PAGES),
            key="nav",
            label_visibility="visible",
        )
        st.divider()
        st.caption(f"**Database**  \n`{context.database.path.name}`")
        if context.dataset is not None:
            st.caption(
                f"**Dataset**  \n{context.dataset.name} {context.dataset.version} · "
                f"{len(context.dataset.items)} questions"
            )
        else:
            st.caption("**Dataset**  \nnot loaded")
        st.caption(
            f"**Scoring profile**  \n"
            f"{context.settings.benchmark.default_scoring_profile}"
        )
        recorded = runner.read_run(context.settings)
        if recorded is not None and runner.is_alive(recorded):
            st.info(f"Experiment running: `{recorded.experiment_name}`")
        if st.button(
            "Refresh data",
            width="stretch",
            help="Clear cached settings, dataset, and queries, then rerun.",
        ):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()
        st.caption(
            "Every figure is computed from stored responses by the analytics engine."
        )
    return page


def main() -> None:
    """Run the dashboard."""
    ui.configure_page()
    try:
        with st.spinner("Connecting to the benchmark database…"):
            context = data.get_context(
                os.getenv("BENCHMARK_CONFIG_DIR"),
                os.getenv("BENCHMARK_PROJECT_ROOT"),
            )
    except Exception as exc:
        st.error("The dashboard could not load the benchmark configuration or database.")
        st.markdown(
            "Check `configs/benchmark.yaml`, `configs/models.yaml`, and "
            "`configs/evaluation.yaml`, then restart Streamlit."
        )
        with st.expander("Technical details", expanded=True):
            st.exception(exc)
        st.code(
            "streamlit run src/benchmark/dashboard/app.py",
            language="bash",
        )
        st.stop()

    page = _sidebar(context)
    if context.dataset_error:
        st.warning(f"Dataset unavailable: {context.dataset_error}")

    try:
        PAGES[page](context)
    except BenchmarkError as exc:
        st.error(f"{type(exc).__name__}: {exc}")
    except Exception as exc:  # keep one failing page from blanking the app
        st.error("This page could not be rendered because of a backend failure.")
        with st.expander("Technical details"):
            st.exception(exc)


if __name__ == "__main__":
    main()
