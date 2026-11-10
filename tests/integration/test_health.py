"""
File: test_health.py

Description:
Integration tests for application initialization and health checks.

Responsibilities:
- Verify initialization creates runtime directories.
- Verify health checks pass with warnings for future files.
- Verify configuration failures are reported.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from benchmark.core.config import AppSettings
from benchmark.core.health import run_health_check
from benchmark.core.logging import reset_logging
from benchmark.core.startup import ensure_runtime_directories, initialize_application

pytestmark = pytest.mark.integration


def test_initialize_application_creates_directories(temp_settings: AppSettings) -> None:
    logger = initialize_application(temp_settings)
    assert temp_settings.benchmark.storage_dir.is_dir()
    assert temp_settings.benchmark.log_dir.is_dir()
    assert temp_settings.benchmark.results_dir.is_dir()
    assert temp_settings.benchmark.exports_dir.is_dir()
    assert logger.name == "benchmark"
    reset_logging()


def test_health_check_passes_with_warnings(temp_settings: AppSettings) -> None:
    initialize_application(temp_settings)
    report = run_health_check(settings=temp_settings)
    reset_logging()
    assert report.ok
    assert any(check.name == "config_hash" and len(check.detail) == 64 for check in report.checks)
    assert any(check.name == "dataset_file" and check.status == "warn" for check in report.checks)
    assert any(
        check.name == "ollama_connectivity" and check.status in {"ok", "warn"}
        for check in report.checks
    )
    assert any(check.name == "disk_space" for check in report.checks)
    assert not report.failures


def test_health_check_reports_valid_dataset(
    temp_settings: AppSettings,
    dataset_path: Path,
) -> None:
    initialize_application(temp_settings)
    destination = temp_settings.benchmark.dataset_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(dataset_path, destination)
    report = run_health_check(settings=temp_settings)
    reset_logging()
    assert report.ok
    dataset_check = next(check for check in report.checks if check.name == "dataset_file")
    assert dataset_check.status == "ok"
    assert "100 items" in dataset_check.detail


def test_health_check_reports_missing_config_dir(tmp_path: Path) -> None:
    report = run_health_check(config_dir=tmp_path / "missing")
    assert not report.ok
    assert any(check.name == "config_dir" and check.status == "fail" for check in report.checks)


def test_ensure_runtime_directories_returns_paths(temp_settings: AppSettings) -> None:
    paths = ensure_runtime_directories(temp_settings)
    assert len(paths) == 4
    assert all(path.is_dir() for path in paths)
