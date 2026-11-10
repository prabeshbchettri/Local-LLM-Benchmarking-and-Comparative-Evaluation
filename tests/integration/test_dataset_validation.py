"""
File: test_dataset_validation.py

Description:
Integration tests for the dataset validation command-line script.

Responsibilities:
- Verify the canonical dataset and probes validate through the script.
- Verify JSON output is machine-readable.
- Verify missing datasets produce a nonzero exit code.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


def _run_validation(
    project_root: Path,
    *args: str,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["BENCHMARK_CONFIG_DIR"] = str(project_root / "configs")
    env["PYTHONPATH"] = str(project_root / "src") + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, str(project_root / "scripts" / "validate_dataset.py"), *args],
        cwd=project_root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_validate_dataset_script_succeeds(project_root: Path) -> None:
    result = _run_validation(project_root)
    assert result.returncode == 0, result.stderr
    assert "Status: VALID" in result.stdout
    assert "Items: 100" in result.stdout
    assert "Performance prompts: 9" in result.stdout
    assert "Consistency items: 10" in result.stdout


def test_validate_dataset_script_json_output(project_root: Path) -> None:
    result = _run_validation(project_root, "--json")
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["valid"] is True
    assert payload["total_items"] == 100
    assert payload["performance_prompts"] == 9
    assert payload["consistency_items"] == 10


def test_validate_dataset_script_can_skip_probes(project_root: Path) -> None:
    result = _run_validation(project_root, "--no-probes")
    assert result.returncode == 0, result.stderr
    assert "Performance prompts: 0" in result.stdout
    assert "Consistency items: 0" in result.stdout


def test_validate_dataset_script_rejects_missing_file(
    project_root: Path,
    tmp_path: Path,
) -> None:
    result = _run_validation(
        project_root,
        "--dataset",
        str(tmp_path / "missing.json"),
        "--no-probes",
    )
    assert result.returncode == 1
    assert "Status: INVALID" in result.stdout
    assert "not found" in result.stdout.lower()
