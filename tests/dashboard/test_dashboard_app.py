"""
File: test_dashboard_app.py

Description:
Headless Streamlit tests for startup, navigation, empty and populated databases,
missing data, configuration failures, and duplicate-run protection.

Responsibilities:
- Run the real app entry point with the Streamlit AppTest harness.
- Verify every page renders without uncaught exceptions.
- Verify empty-state and backend-failure messages are actionable.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import psutil
import pytest
import yaml
from streamlit.testing.v1 import AppTest

from benchmark.core.config import load_settings
from benchmark.dashboard import runner
from benchmark.dashboard.runner import ActiveRun

PAGES = [
    "Overview",
    "Model Comparison",
    "Performance Analysis",
    "Quality Evaluation",
    "Hardware Monitoring",
    "Response Inspection",
    "Experiment History",
    "Reports",
    "Recommendations",
    "Run New Experiment",
]


@pytest.fixture()
def make_app(project_root: Path, dashboard_app_env):
    app_path = project_root / "src" / "benchmark" / "dashboard" / "app.py"

    def _make() -> AppTest:
        app = AppTest.from_file(str(app_path), default_timeout=120)
        app.run()
        return app

    return _make


def _markdown(app: AppTest) -> str:
    return "\n".join(element.value for element in app.markdown)


def test_dashboard_starts_with_an_empty_database(make_app) -> None:
    app = make_app()
    assert not app.exception
    assert app.title[0].value == "Overview"
    metrics = {metric.label: metric.value for metric in app.metric}
    assert metrics["Registered models"] == "0"
    assert metrics["Completed experiments"] == "0"
    assert metrics["Benchmark questions"] == "100"
    assert "No experiments yet" in _markdown(app)


def test_every_page_renders_on_an_empty_database(make_app) -> None:
    app = make_app()
    assert list(app.sidebar.radio[0].options) == PAGES
    for page in PAGES:
        app.sidebar.radio[0].set_value(page)
        app.run()
        assert not app.exception, page
        assert not app.error, page


def test_populated_database_drives_every_page(make_app, seed_database) -> None:
    seed_database()
    app = make_app()
    assert not app.exception
    metrics = {metric.label: metric.value for metric in app.metric}
    assert metrics["Registered models"] == "3"
    assert metrics["Completed experiments"] == "1"
    assert metrics["Latest experiment"] == "Completed"
    frames = [frame.value for frame in app.dataframe]
    assert any("Model" in getattr(frame, "columns", []) for frame in frames)

    for page in PAGES:
        app.sidebar.radio[0].set_value(page)
        app.run()
        assert not app.exception, page
        assert not app.error, page

    app.sidebar.radio[0].set_value("Recommendations")
    app.run()
    recommendation_metrics = [metric.label for metric in app.metric]
    assert "Recommended model" in recommendation_metrics


def test_model_selection_filters_comparison_table(make_app, seed_database) -> None:
    seed_database()
    app = make_app()
    app.sidebar.radio[0].set_value("Model Comparison")
    app.run()
    assert app.multiselect
    chosen = app.multiselect[0].options[0]
    app.multiselect[0].set_value([chosen])
    app.run()
    assert not app.exception
    tables = [frame.value for frame in app.dataframe]
    assert any(
        "Model" in getattr(frame, "columns", [])
        and set(frame["Model"]) == {chosen}
        for frame in tables
    )


def test_history_page_renders_comparison_with_two_completed_experiments(
    make_app, seed_database
) -> None:
    """Historical comparison must render instead of failing on a missing attribute."""
    seed_database(experiment_id="exp-dashboard-a")
    seed_database(
        experiment_id="exp-dashboard-b",
        latency_shift=1.0,
        score_shift=5.0,
    )
    app = make_app()
    app.sidebar.radio[0].set_value("Experiment History")
    app.run()
    assert not app.exception
    assert not app.error
    tables = [frame.value for frame in app.dataframe]
    assert any(
        "Model" in getattr(frame, "columns", [])
        and "Delta" in getattr(frame, "columns", [])
        for frame in tables
    )


def test_reports_page_prepares_real_downloads(make_app, seed_database) -> None:
    seed_database()
    app = make_app()
    app.sidebar.radio[0].set_value("Reports")
    app.run()
    assert not app.exception
    assert not app.error
    assert app.button
    assert app.button[0].label == "Prepare export files"

    app.button[0].click().run()
    assert not app.exception
    assert not app.error
    assert len(app.download_button) >= 4


def test_missing_dataset_is_reported_without_crashing(
    make_app, dashboard_env
) -> None:
    benchmark_path = dashboard_env.config_dir / "benchmark.yaml"
    payload = yaml.safe_load(benchmark_path.read_text(encoding="utf-8"))
    payload["dataset_path"] = str(dashboard_env.project_root / "missing.json")
    benchmark_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    app = make_app()
    assert not app.exception
    assert any("Dataset unavailable" in item.value for item in app.warning)


def test_invalid_configuration_shows_actionable_error(
    make_app, dashboard_env
) -> None:
    (dashboard_env.config_dir / "benchmark.yaml").write_text(
        "name: [unclosed\n", encoding="utf-8"
    )
    app = make_app()
    assert app.error
    assert "could not load the benchmark configuration" in app.error[0].value
    # The app stops before navigation; the exception element is the optional
    # technical-details panel, not an uncaught crash.
    assert not app.title
    assert not app.sidebar.radio


def test_run_page_blocks_while_a_child_process_is_alive(
    make_app, seed_database, dashboard_env
) -> None:
    seed_database()
    settings = load_settings(config_dir=dashboard_env.config_dir)
    run = ActiveRun(
        run_id="live-run",
        pid=os.getpid(),
        experiment_name="dashboard-live-run",
        command=["python", "scripts/run_experiment.py"],
        log_path=str(runner.run_directory(settings) / "live-run.log"),
        started_at="2026-01-01T00:00:00+00:00",
        process_create_time=psutil.Process().create_time(),
        expected_responses=10,
        models=["llama3.1:8b"],
        dataset_path=str(dashboard_env.dataset_path),
        question_count=5,
        question_scope="first 5 questions",
    )
    runner.lock_path(settings).write_text(
        json.dumps(run.as_dict()), encoding="utf-8"
    )
    try:
        app = make_app()
        app.sidebar.radio[0].set_value("Run New Experiment")
        app.run()
        assert not app.exception
        infos = "\n".join(item.value for item in app.info)
        assert "An experiment is running" in infos
        assert app.form_submit_button
        assert app.form_submit_button[0].proto.disabled is True
    finally:
        runner.clear_run(settings)
