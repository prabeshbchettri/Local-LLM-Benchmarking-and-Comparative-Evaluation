"""
File: test_reporting_exports.py

Description:
Tests for the CSV, JSON, HTML, and PDF export formats.

Responsibilities:
- Verify CSV headers, units, row counts, Unicode, and missing-value blanks.
- Verify JSON structure, hashes, nulls, and Unicode.
- Verify HTML sections and autoescaping.
- Verify the PDF is a real, readable document with all ten sections.
"""

from __future__ import annotations

import csv
import io
import json

import pytest

from benchmark.reporting import (
    CSV_FILENAMES,
    REPORT_SCHEMA_VERSION,
    build_bundle,
    build_pdf,
    csv_frames,
    csv_payloads,
    html_bytes,
    json_bytes,
    json_payload,
    render_html,
)

SECTION_HEADINGS = (
    "1. Executive summary",
    "2. Experimental environment",
    "3. Methodology",
    "4. Model metadata",
    "5. Performance results",
    "6. Quality results",
    "7. Hardware analysis",
    "8. Reliability analysis",
    "9. Recommendations",
    "10. Limitations",
)


def _rows(payload: bytes) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(payload.decode("utf-8"))))


def test_csv_exports_have_declared_columns_and_parity(completed_bundle) -> None:
    bundle, tags = completed_bundle
    payloads = csv_payloads(bundle)
    frames = csv_frames(bundle)

    assert set(payloads) == set(CSV_FILENAMES.values())
    for key, filename in CSV_FILENAMES.items():
        header = next(csv.reader(io.StringIO(payloads[filename].decode("utf-8"))))
        assert header == list(frames[key].columns)

    summary = _rows(payloads["experiment_summary.csv"])[0]
    assert summary["experiment_id"] == bundle.experiment_id
    assert summary["status"] == "completed"
    assert summary["preliminary"] == "False"
    assert summary["dataset_hash"] == bundle.experiment.dataset_hash
    assert set(summary["models"].split(", ")) == set(tags)
    assert int(summary["measured_responses"]) == bundle.analysis.measured_responses

    comparison = {
        row["model_tag"]: row for row in _rows(payloads["model_comparison.csv"])
    }
    assert set(comparison) == set(tags)
    for model in bundle.analysis.models:
        row = comparison[model.model_tag]
        assert float(row["latency_mean_seconds"]) == pytest.approx(
            model.statistics.latency.mean
        )
        assert float(row["success_rate_pct"]) == pytest.approx(
            model.statistics.success_rate * 100.0
        )
        assert float(row["quality_score_0_100"]) == pytest.approx(
            model.quality_score
        )

    responses = _rows(payloads["responses.csv"])
    assert len(responses) == len(bundle.inspections)
    assert any("café naïve 漢字" in row["response_text"] for row in responses)

    hardware = _rows(payloads["hardware_metrics.csv"])
    assert hardware
    assert all(row["ram_avg_mib"] for row in hardware)
    assert all(row["gpu_name"] == "Test GPU" for row in hardware)

    evaluations = _rows(payloads["evaluation_results.csv"])
    assert evaluations
    assert all(row["deterministic_score_0_100"] for row in evaluations)


def test_csv_missing_measurements_are_blank_not_zero(
    reporting_context, seed
) -> None:
    experiment_id, tags = seed(
        experiment_id="exp-report-sparse",
        with_hardware=False,
        with_evaluations=False,
    )
    bundle = build_bundle(
        reporting_context.database,
        experiment_id,
        settings=reporting_context.settings,
    )
    payloads = csv_payloads(bundle)

    comparison = {
        row["model_tag"]: row for row in _rows(payloads["model_comparison.csv"])
    }
    assert comparison[tags[0]]["cpu_avg_percent"] == ""
    assert comparison[tags[0]]["quality_score_0_100"] == ""
    assert comparison[tags[0]]["composite_score_0_100"] == ""

    assert _rows(payloads["hardware_metrics.csv"]) == []
    evaluations = _rows(payloads["evaluation_results.csv"])
    assert evaluations
    assert evaluations[0]["judge_normalized_score_0_100"] == ""


def test_json_export_is_versioned_and_complete(completed_bundle) -> None:
    bundle, tags = completed_bundle
    payload = json_payload(bundle)
    decoded = json.loads(json_bytes(bundle).decode("utf-8"))
    assert decoded == payload

    assert set(payload) >= {
        "schema_version",
        "generated_at",
        "preliminary",
        "experiment",
        "dataset",
        "configuration",
        "model_metadata",
        "runs",
        "responses",
        "performance_metrics",
        "hardware_measurements",
        "evaluation_results",
        "evaluation_breakdown",
        "analytics",
        "recommendations",
        "errors",
        "limitations",
    }
    assert payload["schema_version"] == REPORT_SCHEMA_VERSION
    assert payload["preliminary"] is False
    assert payload["dataset"]["content_sha256"] == bundle.experiment.dataset_hash
    assert payload["dataset"]["item_count"] == bundle.detail.dataset_item_count
    assert payload["configuration"]["config_hash"] == "a" * 64
    assert payload["configuration"]["database_path"].endswith("benchmark.db")
    assert {model["tag"] for model in payload["model_metadata"]} == set(tags)
    assert len(payload["responses"]) == len(bundle.inspections)
    assert len(payload["performance_metrics"]) == len(bundle.inspections)
    assert len(payload["hardware_measurements"]) == len(
        bundle.hardware_rows
    )
    assert len(payload["evaluation_results"]) == len(bundle.inspections)
    assert payload["recommendations"]
    assert payload["limitations"]
    assert any(
        "café naïve 漢字" in row["response_text"] for row in payload["responses"]
    )


def test_json_export_keeps_missing_values_null(reporting_context, seed) -> None:
    experiment_id, _ = seed(
        experiment_id="exp-report-sparse",
        with_hardware=False,
        with_evaluations=False,
    )
    bundle = build_bundle(
        reporting_context.database,
        experiment_id,
        settings=reporting_context.settings,
    )
    payload = json_payload(bundle)

    assert payload["hardware_measurements"] == []
    assert payload["evaluation_results"][0]["deterministic_normalized_score_0_100"] is None
    assert payload["evaluation_results"][0]["judge_normalized_score_0_100"] is None
    assert payload["analytics"]["models"][0]["quality_score"] is None


def test_html_export_has_all_sections_and_escapes_user_text(
    completed_bundle,
) -> None:
    bundle, tags = completed_bundle
    html = render_html(bundle)

    assert "<!doctype html>" in html.lower()
    for heading in SECTION_HEADINGS:
        assert f"<h2>{heading}</h2>" in html
    assert "Appendix \u2014 response excerpts" in html
    for tag in tags:
        assert tag in html

    assert "&lt;b&gt;seeded&lt;/b&gt;" in html
    assert "<b>seeded</b>" not in html
    assert "café naïve 漢字" in html
    assert html_bytes(bundle).decode("utf-8") == html


def test_pdf_export_is_readable_and_complete(completed_bundle) -> None:
    pypdf = pytest.importorskip("pypdf")
    bundle, tags = completed_bundle
    result = build_pdf(bundle)

    assert result.data.startswith(b"%PDF")
    assert result.page_count >= 1
    reader = pypdf.PdfReader(io.BytesIO(result.data))
    assert len(reader.pages) == result.page_count

    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    assert "Benchmark Report" in text
    for heading in SECTION_HEADINGS:
        assert heading in text
    for tag in tags:
        assert tag in text
