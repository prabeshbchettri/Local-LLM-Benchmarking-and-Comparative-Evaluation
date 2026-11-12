"""
File: test_reporting_service.py

Description:
Tests for report generation: format selection, manifest integrity, preliminary
handling, and refusal of incomplete or nonexistent experiments.

Responsibilities:
- Verify every format writes real files with matching manifest hashes.
- Verify format subsets and unknown formats are handled explicitly.
- Verify non-completed experiments are refused unless explicitly allowed.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from benchmark.reporting import (
    REPORT_SCHEMA_VERSION,
    ReportError,
    generate_reports,
)

EXPECTED_NAMES = {
    "experiment_summary.csv",
    "model_comparison.csv",
    "performance_metrics.csv",
    "hardware_metrics.csv",
    "evaluation_results.csv",
    "responses.csv",
}


def _manifest(result) -> dict:
    assert result.manifest_path is not None
    return json.loads(result.manifest_path.read_text(encoding="utf-8"))


def test_generate_reports_writes_all_formats_and_manifest(
    reporting_context, seed
) -> None:
    experiment_id, _ = seed()
    result = generate_reports(
        reporting_context.database,
        experiment_id,
        settings=reporting_context.settings,
    )

    names = {generated.name for generated in result.files}
    assert names == EXPECTED_NAMES | {
        f"{experiment_id}.json",
        f"{experiment_id}.html",
        f"{experiment_id}.pdf",
    }
    assert result.output_dir == (
        reporting_context.settings.benchmark.exports_dir / experiment_id
    )
    assert result.preliminary is False
    assert result.page_count is not None and result.page_count >= 1

    manifest = _manifest(result)
    assert manifest["schema_version"] == REPORT_SCHEMA_VERSION
    assert manifest["experiment_id"] == experiment_id
    assert manifest["experiment_status"] == "completed"
    assert manifest["preliminary"] is False
    assert manifest["page_count"] == result.page_count
    assert manifest["source_database"] == str(reporting_context.database.path)
    assert manifest["filters"] == {
        "run_profile": None,
        "include_cold_start": False,
        "include_smoke": False,
    }

    entries = {entry["name"]: entry for entry in manifest["files"]}
    assert set(entries) == names
    for generated in result.files:
        data = generated.path.read_bytes()
        assert generated.size_bytes == len(data)
        assert generated.sha256 == hashlib.sha256(data).hexdigest()
        assert entries[generated.name]["sha256"] == generated.sha256
        assert entries[generated.name]["format"] == generated.format


def test_generate_reports_respects_format_selection(reporting_context, seed) -> None:
    experiment_id, _ = seed()
    result = generate_reports(
        reporting_context.database,
        experiment_id,
        settings=reporting_context.settings,
        formats=("json", "html"),
    )

    assert {generated.name for generated in result.files} == {
        f"{experiment_id}.json",
        f"{experiment_id}.html",
    }
    assert result.page_count is None
    assert len(_manifest(result)["files"]) == 2


def test_duplicate_and_uppercase_formats_are_normalized(
    reporting_context, seed
) -> None:
    experiment_id, _ = seed()
    result = generate_reports(
        reporting_context.database,
        experiment_id,
        settings=reporting_context.settings,
        formats=("PDF", "pdf"),
    )

    assert [generated.name for generated in result.files] == [f"{experiment_id}.pdf"]


def test_unknown_format_is_rejected(reporting_context, seed) -> None:
    experiment_id, _ = seed()
    with pytest.raises(ReportError, match="unknown report format"):
        generate_reports(
            reporting_context.database,
            experiment_id,
            settings=reporting_context.settings,
            formats=("docx",),
        )


def test_empty_format_selection_is_rejected(reporting_context, seed) -> None:
    experiment_id, _ = seed()
    with pytest.raises(ReportError, match="at least one report format"):
        generate_reports(
            reporting_context.database,
            experiment_id,
            settings=reporting_context.settings,
            formats=(),
        )


def test_incomplete_experiment_is_refused_by_default(reporting_context, seed) -> None:
    experiment_id, _ = seed(experiment_id="exp-report-preliminary", completed=False)
    with pytest.raises(ReportError, match="only completed experiments"):
        generate_reports(
            reporting_context.database,
            experiment_id,
            settings=reporting_context.settings,
        )


def test_allow_incomplete_marks_every_export_preliminary(
    reporting_context, seed
) -> None:
    experiment_id, _ = seed(experiment_id="exp-report-preliminary", completed=False)
    result = generate_reports(
        reporting_context.database,
        experiment_id,
        settings=reporting_context.settings,
        formats=("json", "html"),
        allow_incomplete=True,
    )

    assert result.preliminary is True
    assert _manifest(result)["preliminary"] is True
    json_payload = json.loads(
        (result.output_dir / f"{experiment_id}.json").read_text(encoding="utf-8")
    )
    assert json_payload["preliminary"] is True
    html = (result.output_dir / f"{experiment_id}.html").read_text(encoding="utf-8")
    assert "PRELIMINARY DATA" in html


def test_missing_experiment_is_rejected(reporting_context) -> None:
    with pytest.raises(ReportError, match="does not exist"):
        generate_reports(
            reporting_context.database,
            "exp-does-not-exist",
            settings=reporting_context.settings,
        )


def test_output_directory_can_be_overridden(reporting_context, seed, tmp_path) -> None:
    experiment_id, _ = seed()
    output_dir = tmp_path / "custom" / "exports" / experiment_id
    result = generate_reports(
        reporting_context.database,
        experiment_id,
        settings=reporting_context.settings,
        output_dir=output_dir,
        formats=("json",),
    )

    assert result.output_dir == output_dir
    assert (output_dir / f"{experiment_id}.json").is_file()
