"""Dashboard views package.

This directory is intentionally not named ``pages``: Streamlit treats a sibling
``pages`` directory as automatic multipage navigation, which would bypass the
dashboard's own navigation control and expose non-standalone modules as routes.
"""

from benchmark.dashboard.views import (
    comparison,
    hardware,
    history,
    inspection,
    overview,
    performance,
    quality,
    recommendations,
    run_experiment,
)

__all__ = [
    "comparison",
    "hardware",
    "history",
    "inspection",
    "overview",
    "performance",
    "quality",
    "recommendations",
    "run_experiment",
]
