"""
File: ui.py

Description:
Shared Streamlit presentation helpers for the dashboard.

Why this file exists:
The dashboard pages should agree on spacing, formatting, metric cards, empty
states, and selector behavior. Keeping those decisions here avoids drift and keeps
page modules focused on data and charts.

Responsibilities:
- Apply the page configuration and a small amount of theme CSS.
- Format seconds, percentages, rates, and memory consistently.
- Render page headers, metric cards, empty states, and limitations.
- Provide experiment and model selectors with useful labels and tooltips.
"""

from __future__ import annotations

from typing import Sequence

import pandas as pd
import streamlit as st

from benchmark.dashboard import data
from benchmark.database.repositories import ExperimentView

CSS = """
<style>
    .block-container {padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1500px;}
    h1 {font-size: 1.9rem !important; font-weight: 650 !important;}
    h2, h3 {font-weight: 600 !important;}
    div[data-testid="stMetric"] {
        border: 1px solid rgba(148, 163, 184, 0.28);
        border-radius: 0.75rem;
        padding: 0.75rem 0.9rem;
        background: rgba(248, 250, 252, 0.40);
    }
    .app-subtitle {color: #64748b; font-size: 0.95rem; margin-top: -0.5rem;}
    .empty-state {
        border: 1px dashed rgba(148, 163, 184, 0.55);
        border-radius: 0.9rem;
        padding: 2.2rem 1.5rem;
        text-align: center;
        background: rgba(248, 250, 252, 0.35);
        margin: 1rem 0;
    }
    .empty-state h3 {margin: 0 0 0.35rem 0; font-size: 1.05rem;}
    .empty-state p {margin: 0; color: #64748b; font-size: 0.9rem;}
    .small-note {color: #64748b; font-size: 0.82rem;}
    code {font-size: 0.82rem;}
</style>
"""


def configure_page() -> None:
    """Set page configuration and inject the dashboard theme."""
    st.set_page_config(
        page_title="Local LLM Benchmark",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CSS, unsafe_allow_html=True)


def page_header(title: str, subtitle: str | None = None) -> None:
    """Render a consistent page title and subtitle."""
    st.title(title)
    if subtitle:
        st.markdown(f'<div class="app-subtitle">{subtitle}</div>', unsafe_allow_html=True)
        st.write("")


def section(title: str, help_text: str | None = None) -> None:
    """Render a section heading with optional help text."""
    st.subheader(title, help=help_text)


def empty_state(title: str, message: str) -> None:
    """Render a neutral empty state instead of a chart or table."""
    st.markdown(
        f'<div class="empty-state"><h3>{title}</h3><p>{message}</p></div>',
        unsafe_allow_html=True,
    )


def note(text: str) -> None:
    """Render a small muted note."""
    st.markdown(f'<div class="small-note">{text}</div>', unsafe_allow_html=True)


def limitations(items: Sequence[str]) -> None:
    """Render deduplicated limitations in a collapsed expander."""
    unique = list(dict.fromkeys(item for item in items if item))
    if not unique:
        return
    with st.expander("Limitations and caveats", expanded=False):
        for item in unique:
            st.markdown(f"- {item}")


def metric_row(cards: Sequence[tuple[str, str, str | None]]) -> None:
    """Render a row of bordered metric cards."""
    columns = st.columns(len(cards))
    for column, (label, value, help_text) in zip(columns, cards):
        column.metric(label, value, help=help_text, border=True)


def format_seconds(value: float | None, digits: int = 2) -> str:
    """Format a duration in seconds."""
    if value is None:
        return "—"
    if value < 0.01:
        return f"{value * 1000:.1f} ms"
    return f"{value:.{digits}f} s"


def format_percent(value: float | None, digits: int = 1) -> str:
    """Format a 0-1 fraction as a percentage."""
    if value is None:
        return "—"
    return f"{value * 100:.{digits}f}%"


def format_score(value: float | None, digits: int = 1) -> str:
    """Format a 0-100 score."""
    if value is None:
        return "—"
    return f"{value:.{digits}f}"


def format_rate(value: float | None, digits: int = 1) -> str:
    """Format tokens per second."""
    if value is None:
        return "—"
    return f"{value:.{digits}f} tok/s"


def format_mib(value: float | None, digits: int = 0) -> str:
    """Format MiB for compact display."""
    if value is None:
        return "—"
    return f"{value:,.{digits}f} MiB"


def format_count(value: int | None) -> str:
    """Format an integer count."""
    return "—" if value is None else f"{value:,}"


def clean_frame(rows: Sequence[dict[str, object]]) -> pd.DataFrame:
    """Build a DataFrame that renders missing numeric values as blanks.

    Object columns made only of numbers and None are converted to float so the
    Streamlit number columns display an empty cell instead of the text "None".
    """
    frame = pd.DataFrame(rows)
    for column in frame.columns:
        if frame[column].dtype != object:
            continue
        converted = pd.to_numeric(frame[column], errors="coerce")
        if frame[column].notna().sum() == converted.notna().sum():
            frame[column] = converted
    return frame


def experiment_label(view: ExperimentView) -> str:
    """Return a compact experiment selector label."""
    created = view.created_at.astimezone().strftime("%Y-%m-%d %H:%M")
    name = f" · {view.name}" if view.name else ""
    return f"{view.experiment_id} · {view.status} · {created}{name}"


def select_experiment(
    context: data.DashboardContext,
    key: str,
    *,
    label: str = "Experiment",
    completed_only: bool = False,
    help_text: str | None = None,
) -> ExperimentView | None:
    """Render an experiment selector and return the chosen view."""
    views = (
        data.completed_experiments(context)
        if completed_only
        else data.history(context)
    )
    if not views:
        empty_state(
            "No experiments yet",
            "Run an experiment from the Run New Experiment page to populate the dashboard.",
        )
        return None
    lookup = {view.experiment_id: view for view in views}
    selected = st.selectbox(
        label,
        options=list(lookup),
        format_func=lambda experiment_id: experiment_label(lookup[experiment_id]),
        key=key,
        help=help_text,
    )
    return lookup[selected]


def select_profile(
    context: data.DashboardContext,
    key: str,
    *,
    label: str = "Scoring profile",
    help_text: str | None = None,
) -> str:
    """Render a workload profile selector."""
    names = data.profile_names(context)
    default = context.settings.benchmark.default_scoring_profile
    index = names.index(default) if default in names else 0
    return st.selectbox(
        label,
        options=names,
        index=index,
        key=key,
        help=help_text
        or "Profiles weight quality, latency, throughput, efficiency, and reliability.",
    )


def select_models(
    tags: Sequence[str],
    key: str,
    *,
    label: str = "Models",
    default_all: bool = True,
) -> list[str]:
    """Render a model multiselect that never returns an empty selection."""
    if not tags:
        return []
    selected = st.multiselect(
        label,
        options=list(tags),
        default=list(tags) if default_all else list(tags[:1]),
        key=key,
    )
    return selected or list(tags[:1])


def profile_description(context: data.DashboardContext, name: str) -> str:
    """Return the configured description for a profile."""
    profile = context.settings.benchmark.workload_profiles.get(name)
    return profile.description if profile else ""


__all__ = [
    "clean_frame",
    "configure_page",
    "empty_state",
    "experiment_label",
    "format_count",
    "format_mib",
    "format_percent",
    "format_rate",
    "format_score",
    "format_seconds",
    "limitations",
    "metric_row",
    "note",
    "page_header",
    "profile_description",
    "section",
    "select_experiment",
    "select_models",
    "select_profile",
]
