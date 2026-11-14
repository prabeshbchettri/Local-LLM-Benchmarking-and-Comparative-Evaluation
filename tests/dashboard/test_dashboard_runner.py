"""
File: test_dashboard_runner.py

Description:
Tests for dashboard experiment command construction, run state, and duplicate guards.

Responsibilities:
- Verify the child command mirrors a direct CLI invocation.
- Verify a live child process blocks a second submission.
- Verify start_run records the lock and expected request count.
- Verify stale or empty selections are rejected without spawning.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import psutil
import pytest

from benchmark.dashboard import runner
from benchmark.dashboard.runner import ActiveRun


def _run(settings, *, pid: int, create_time: float | None) -> ActiveRun:
    return ActiveRun(
        run_id="test-run",
        pid=pid,
        experiment_name="dashboard-test-run",
        command=["python", "scripts/run_experiment.py"],
        log_path=str(runner.run_directory(settings) / "test-run.log"),
        started_at="2026-01-01T00:00:00+00:00",
        process_create_time=create_time,
        expected_responses=11,
        models=["llama3.1:8b"],
        dataset_path="data/datasets/benchmark_v1.json",
        question_count=5,
        question_scope="first 5 questions",
    )


def test_build_command_mirrors_cli_selection() -> None:
    command = runner.build_command(
        python_executable="python",
        project_root=Path("/project"),
        config_dir=Path("/project/configs"),
        models=["llama3.1:8b", "qwen2.5:7b-instruct"],
        profile="quality",
        dataset_path=Path("/project/data/benchmark_v1.json"),
        limit=10,
        repetitions=2,
        warmup=1,
        cold_start=False,
        name="dashboard-test",
    )
    assert command[:2] == ["python", str(Path("/project") / "scripts" / "run_experiment.py")]
    assert command[command.index("--models") + 1 : command.index("--models") + 3] == [
        "llama3.1:8b",
        "qwen2.5:7b-instruct",
    ]
    assert command[command.index("--profile") + 1] == "quality"
    assert command[command.index("--repeats") + 1] == "2"
    assert command[command.index("--limit") + 1] == "10"
    assert command[command.index("--warmup") + 1] == "1"
    assert command[command.index("--dataset") + 1] == str(
        Path("/project/data/benchmark_v1.json")
    )
    assert "--cold-start" not in command


def test_build_command_uses_cold_start_without_warmup() -> None:
    command = runner.build_command(
        python_executable="python",
        project_root=Path("/project"),
        config_dir=Path("/project/configs"),
        models=["llama3.1:8b"],
        profile="cold_start",
        dataset_path=Path("/project/data/benchmark_v1.json"),
        limit=None,
        repetitions=3,
        warmup=1,
        cold_start=True,
        name="dashboard-test",
    )
    assert "--cold-start" in command
    assert "--warmup" not in command
    assert "--limit" not in command


def test_live_run_blocks_a_duplicate_submission(dashboard_context) -> None:
    settings = dashboard_context.settings
    run = _run(settings, pid=os.getpid(), create_time=psutil.Process().create_time())
    runner.lock_path(settings).write_text(
        json.dumps(run.as_dict()), encoding="utf-8"
    )
    try:
        with pytest.raises(runner.RunGuardError, match="already running"):
            runner.start_run(
                settings,
                project_root=settings.project_root,
                models=["llama3.1:8b"],
                profile="quality",
                dataset_path=dashboard_context.settings.benchmark.dataset_path,
                dataset_question_count=100,
                limit=5,
                repetitions=1,
                warmup=1,
                cold_start=False,
            )
    finally:
        runner.clear_run(settings)


def test_stale_run_record_is_not_alive(dashboard_context) -> None:
    settings = dashboard_context.settings
    run = _run(settings, pid=999_999, create_time=1.0)
    assert runner.is_alive(run) is False
    runner.lock_path(settings).write_text(
        json.dumps(run.as_dict()), encoding="utf-8"
    )
    try:
        assert runner.read_run(settings) == run
        assert runner.is_alive(runner.read_run(settings)) is False
    finally:
        runner.clear_run(settings)


def test_start_run_records_lock_and_expected_requests(
    dashboard_context, monkeypatch
) -> None:
    settings = dashboard_context.settings
    calls: dict[str, object] = {}

    class FakeProcess:
        pid = 424242

        def create_time(self) -> float:
            return 1000.0

    def fake_popen(command, **kwargs):
        calls["command"] = command
        calls["kwargs"] = kwargs
        return FakeProcess()

    monkeypatch.setattr(runner.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(runner.psutil, "Process", lambda _pid: FakeProcess())
    try:
        run = runner.start_run(
            settings,
            project_root=settings.project_root,
            models=["llama3.1:8b"],
            profile="quality",
            dataset_path=settings.benchmark.dataset_path,
            dataset_question_count=100,
            limit=5,
            repetitions=2,
            warmup=1,
            cold_start=False,
        )
        assert run.expected_responses == 11
        assert run.pid == 424242
        assert run.experiment_name.startswith("dashboard-")
        assert Path(run.log_path).is_file()
        assert runner.read_run(settings) == run
        assert "--dataset" in run.command
        assert calls["command"] == run.command
        assert calls["kwargs"]["cwd"] == str(settings.project_root)
    finally:
        runner.clear_run(settings)


def test_start_run_rejects_empty_model_selection(dashboard_context) -> None:
    settings = dashboard_context.settings
    with pytest.raises(runner.RunGuardError, match="at least one model"):
        runner.start_run(
            settings,
            project_root=settings.project_root,
            models=[],
            profile="quality",
            dataset_path=settings.benchmark.dataset_path,
            dataset_question_count=100,
            limit=5,
            repetitions=1,
            warmup=1,
            cold_start=False,
        )


def test_fresh_empty_lock_blocks_concurrent_start(dashboard_context) -> None:
    settings = dashboard_context.settings
    lock = runner.lock_path(settings)
    lock.write_text("", encoding="utf-8")
    try:
        with pytest.raises(runner.RunGuardError, match="starting"):
            runner.start_run(
                settings,
                project_root=settings.project_root,
                models=["llama3.1:8b"],
                profile="quality",
                dataset_path=settings.benchmark.dataset_path,
                dataset_question_count=100,
                limit=5,
                repetitions=1,
                warmup=1,
                cold_start=False,
            )
    finally:
        runner.clear_run(settings)


def test_stale_empty_lock_is_replaced(dashboard_context, monkeypatch) -> None:
    settings = dashboard_context.settings
    lock = runner.lock_path(settings)
    lock.write_text("", encoding="utf-8")
    old = 1.0
    os.utime(lock, (old, old))

    class FakeProcess:
        pid = 424242

        def create_time(self) -> float:
            return 1000.0

    monkeypatch.setattr(runner.subprocess, "Popen", lambda *args, **kwargs: FakeProcess())
    monkeypatch.setattr(runner.psutil, "Process", lambda _pid: FakeProcess())
    try:
        run = runner.start_run(
            settings,
            project_root=settings.project_root,
            models=["llama3.1:8b"],
            profile="quality",
            dataset_path=settings.benchmark.dataset_path,
            dataset_question_count=100,
            limit=5,
            repetitions=1,
            warmup=1,
            cold_start=False,
        )
        assert runner.read_run(settings) == run
    finally:
        runner.clear_run(settings)
