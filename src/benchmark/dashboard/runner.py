"""
File: runner.py

Description:
Start a persisted experiment in a child process and track it with a small lock file.

Why this file exists:
Streamlit reruns its script on every interaction. Running the orchestrator in the
dashboard process would block those reruns for the whole experiment, while a job
queue adds infrastructure the project does not need. A child process plus a lock
file keeps the interface responsive and reliably prevents duplicate submissions.

Responsibilities:
- Build the same command an operator would run from the CLI.
- Start the experiment with a unique operator-visible name and a log file.
- Record the child PID under storage/dashboard_runs/active_run.json.
- Detect live, finished, and stale runs without treating a recycled PID as live.
- Refuse to start a second experiment while the recorded child is running.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence
from uuid import uuid4

import psutil

from benchmark.core.config import AppSettings
from benchmark.core.exceptions import BenchmarkError

RUN_DIRECTORY = "dashboard_runs"
LOCK_FILENAME = "active_run.json"
LOCK_START_GRACE_SECONDS = 60.0


class RunGuardError(BenchmarkError):
    """Raised when an experiment cannot be started safely."""


@dataclass(frozen=True)
class ActiveRun:
    """A child experiment recorded on disk."""

    run_id: str
    pid: int
    experiment_name: str
    command: list[str]
    log_path: str
    started_at: str
    process_create_time: float | None
    expected_responses: int
    models: list[str]
    dataset_path: str
    question_count: int
    question_scope: str

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "ActiveRun":
        return cls(
            run_id=str(payload["run_id"]),
            pid=int(payload["pid"]),
            experiment_name=str(payload["experiment_name"]),
            command=[str(part) for part in payload.get("command", [])],
            log_path=str(payload["log_path"]),
            started_at=str(payload["started_at"]),
            process_create_time=(
                float(payload["process_create_time"])
                if payload.get("process_create_time") is not None
                else None
            ),
            expected_responses=int(payload.get("expected_responses", 0)),
            models=[str(tag) for tag in payload.get("models", [])],
            dataset_path=str(payload.get("dataset_path", "")),
            question_count=int(payload.get("question_count", 0)),
            question_scope=str(payload.get("question_scope", "")),
        )

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def run_directory(settings: AppSettings) -> Path:
    """Return the dashboard run state directory, creating it if needed."""
    directory = settings.benchmark.storage_dir / RUN_DIRECTORY
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def lock_path(settings: AppSettings) -> Path:
    """Return the active-run lock path."""
    return run_directory(settings) / LOCK_FILENAME


def read_run(settings: AppSettings) -> ActiveRun | None:
    """Read the recorded run without judging whether it is still alive."""
    path = lock_path(settings)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    try:
        return ActiveRun.from_dict(payload)
    except (KeyError, TypeError, ValueError):
        return None


def is_process_alive(pid: int, process_create_time: float | None = None) -> bool:
    """Return whether a process exists and is the same process that was recorded."""
    try:
        process = psutil.Process(pid)
        if (
            process_create_time is not None
            and abs(process.create_time() - process_create_time) > 1.0
        ):
            return False
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except (psutil.Error, OSError):
        return False


def is_alive(run: ActiveRun) -> bool:
    """Return whether the recorded child process is still running."""
    return is_process_alive(run.pid, run.process_create_time)


def _acquire_lock(settings: AppSettings) -> Path:
    """Atomically create the lock, replacing only a demonstrably stale record."""
    path = lock_path(settings)
    for _ in range(2):
        try:
            handle = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(handle)
            return path
        except FileExistsError:
            existing = read_run(settings)
            if existing is not None and is_alive(existing):
                raise RunGuardError(
                    "An experiment is already running "
                    f"({existing.experiment_name}). Wait for it to finish before "
                    "starting another."
                )
            if existing is None:
                try:
                    age = time.time() - path.stat().st_mtime
                except OSError:
                    age = 0.0
                if age < LOCK_START_GRACE_SECONDS:
                    raise RunGuardError(
                        "Another experiment is starting. Wait for it to finish "
                        "before starting another."
                    )
            path.unlink(missing_ok=True)
    raise RunGuardError("Could not acquire the experiment run lock.")


def build_command(
    *,
    python_executable: str,
    project_root: Path,
    config_dir: Path,
    models: Sequence[str],
    profile: str,
    dataset_path: Path,
    limit: int | None,
    repetitions: int,
    warmup: int,
    cold_start: bool,
    name: str,
) -> list[str]:
    """Build the CLI command used for a dashboard-launched experiment."""
    command = [
        python_executable,
        str(project_root / "scripts" / "run_experiment.py"),
        "--config-dir",
        str(config_dir),
        "--models",
        *models,
        "--profile",
        profile,
        "--repeats",
        str(repetitions),
        "--name",
        name,
        "--dataset",
        str(dataset_path),
    ]
    if limit is not None:
        command.extend(["--limit", str(limit)])
    if cold_start:
        command.append("--cold-start")
    else:
        command.extend(["--warmup", str(warmup)])
    return command


def start_run(
    settings: AppSettings,
    *,
    project_root: Path,
    models: Sequence[str],
    profile: str,
    dataset_path: Path,
    dataset_question_count: int,
    limit: int | None,
    repetitions: int,
    warmup: int,
    cold_start: bool,
) -> ActiveRun:
    """Start one experiment child process, or raise RunGuardError."""
    if not models:
        raise RunGuardError("Select at least one model before starting an experiment.")
    if repetitions < 1:
        raise RunGuardError("Repetitions must be at least 1.")
    if warmup < 0:
        raise RunGuardError("Warm-up requests must not be negative.")
    question_count = limit if limit is not None else dataset_question_count
    if question_count < 1:
        raise RunGuardError("Select at least one benchmark question.")
    expected = len(models) * (
        question_count * repetitions + (0 if cold_start else warmup)
    )
    run_id = uuid4().hex
    name = f"dashboard-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{run_id[:6]}"
    command = build_command(
        python_executable=sys.executable,
        project_root=project_root,
        config_dir=settings.config_dir,
        models=models,
        profile=profile,
        dataset_path=dataset_path,
        limit=limit,
        repetitions=repetitions,
        warmup=warmup,
        cold_start=cold_start,
        name=name,
    )
    log_path = run_directory(settings) / f"{run_id}.log"
    lock = _acquire_lock(settings)
    environment = os.environ.copy()
    source_dir = project_root / "src"
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(source_dir), environment.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    try:
        with log_path.open("w", encoding="utf-8", errors="replace") as log:
            process = subprocess.Popen(
                command,
                cwd=str(project_root),
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                creationflags=creation_flags,
            )
    except Exception:
        lock.unlink(missing_ok=True)
        raise
    try:
        create_time: float | None = psutil.Process(process.pid).create_time()
    except (psutil.Error, OSError):
        create_time = None
    run = ActiveRun(
        run_id=run_id,
        pid=process.pid,
        experiment_name=name,
        command=command,
        log_path=str(log_path),
        started_at=datetime.now(timezone.utc).isoformat(),
        process_create_time=create_time,
        expected_responses=expected,
        models=list(models),
        dataset_path=str(dataset_path),
        question_count=question_count,
        question_scope=(
            "all questions" if limit is None else f"first {question_count} questions"
        ),
    )
    temporary = lock.with_name(f".{lock.name}.tmp")
    try:
        temporary.write_text(json.dumps(run.as_dict(), indent=2), encoding="utf-8")
        temporary.replace(lock)
    except Exception:
        lock.unlink(missing_ok=True)
        temporary.unlink(missing_ok=True)
        try:
            process.terminate()
        except Exception:
            pass
        raise
    return run


def clear_run(settings: AppSettings) -> None:
    """Remove the run record; used after a finished or abandoned run."""
    lock_path(settings).unlink(missing_ok=True)


def tail_log(run: ActiveRun, lines: int = 60) -> str:
    """Return the last lines of a run log, tolerating a missing or huge file."""
    path = Path(run.log_path)
    if not path.is_file():
        return ""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return "\n".join(text.splitlines()[-lines:])


__all__ = [
    "ActiveRun",
    "RunGuardError",
    "build_command",
    "clear_run",
    "is_alive",
    "is_process_alive",
    "lock_path",
    "read_run",
    "run_directory",
    "start_run",
    "tail_log",
]
