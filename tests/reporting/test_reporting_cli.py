"""
File: test_reporting_cli.py

Description:
Tests for the `benchmark report` command and the standalone reporting script.

Responsibilities:
- Verify the CLI writes files and prints a JSON summary.
- Verify backend failures produce exit code 1 and a clean error message.
- Verify the standalone script uses the same reporting service.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from benchmark.__main__ import main


def test_report_command_writes_json_summary(
    reporting_env, reporting_context, seed, monkeypatch, capsys
) -> None:
    monkeypatch.setenv("BENCHMARK_CONFIG_DIR", str(reporting_env.config_dir))
    experiment_id, _ = seed()

    exit_code = main(["report", experiment_id, "--format", "json", "--json"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["experiment_id"] == experiment_id
    assert payload["preliminary"] is False
    assert [item["name"] for item in payload["files"]] == [f"{experiment_id}.json"]
    assert Path(payload["manifest"]).is_file()
    assert Path(payload["output_dir"]).is_dir()


def test_report_command_refuses_incomplete_experiment(
    reporting_env, reporting_context, seed, monkeypatch, capsys
) -> None:
    monkeypatch.setenv("BENCHMARK_CONFIG_DIR", str(reporting_env.config_dir))
    experiment_id, _ = seed(experiment_id="exp-report-cli", completed=False)

    exit_code = main(["report", experiment_id])

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "ERROR:" in captured.err
    assert "only completed experiments" in captured.err


def _load_script(project_root: Path):
    script = project_root / "scripts" / "generate_report.py"
    spec = importlib.util.spec_from_file_location("generate_report_script", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_standalone_script_generates_reports(
    project_root: Path, reporting_env, reporting_context, seed, capsys
) -> None:
    experiment_id, _ = seed()
    module = _load_script(project_root)

    exit_code = module.main(
        [
            experiment_id,
            "--config-dir",
            str(reporting_env.config_dir),
            "--format",
            "html",
            "--json",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["experiment_id"] == experiment_id
    assert payload["preliminary"] is False
    assert [item["name"] for item in payload["files"]] == [
        f"{experiment_id}.html"
    ]
