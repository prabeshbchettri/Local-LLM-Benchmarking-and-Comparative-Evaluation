"""
File: test_cli.py

Description:
Integration tests for the module entry point and preflight script.

Responsibilities:
- Verify init and health commands work with an isolated configuration.
- Verify invalid configuration returns a nonzero exit code.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


def _run_cli(
    project_root: Path,
    config_dir: Path,
    *args: str,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["BENCHMARK_CONFIG_DIR"] = str(config_dir)
    env["PYTHONPATH"] = str(project_root / "src") + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-m", "benchmark", *args],
        cwd=project_root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_init_command(project_root: Path, temp_config_dir: Path) -> None:
    result = _run_cli(project_root, temp_config_dir, "init")
    assert result.returncode == 0, result.stderr
    assert "Initialized project" in result.stdout


def test_health_command(project_root: Path, temp_config_dir: Path) -> None:
    result = _run_cli(project_root, temp_config_dir, "health")
    assert result.returncode == 0, result.stderr
    assert "Health check passed" in result.stdout
    assert "WARN" in result.stdout


def test_invalid_config_returns_nonzero(project_root: Path, temp_config_dir: Path) -> None:
    benchmark_path = temp_config_dir / "benchmark.yaml"
    benchmark_path.write_text("name: [unclosed\n", encoding="utf-8")
    result = _run_cli(project_root, temp_config_dir, "health")
    assert result.returncode == 1
    assert "ERROR" in result.stderr


def test_preflight_script(project_root: Path, temp_config_dir: Path) -> None:
    env = os.environ.copy()
    env["BENCHMARK_CONFIG_DIR"] = str(temp_config_dir)
    env["PYTHONPATH"] = str(project_root / "src") + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, str(project_root / "scripts" / "preflight.py")],
        cwd=project_root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "Health check passed" in result.stdout
