"""
File: __init__.py

Description:
Public exports for CSV, JSON, HTML, and PDF reporting.

Why this file exists:
Callers should import one bundle builder, one view, and one generation service
instead of reaching into individual exporters.

Responsibilities:
- Export bundle collection, presentation, exporters, and the service.
"""

from benchmark.core.exceptions import ReportError
from benchmark.reporting.bundle import (
    COMPLETED_STATUS,
    REPORT_SCHEMA_VERSION,
    ReportBundle,
    build_bundle,
)
from benchmark.reporting.csv_export import (
    CSV_FILENAMES,
    csv_frames,
    csv_payloads,
    write_csv_exports,
)
from benchmark.reporting.html_report import html_bytes, render_html, write_html_report
from benchmark.reporting.json_export import json_bytes, json_payload, write_json_export
from benchmark.reporting.pdf_report import PdfResult, build_pdf, write_pdf_report
from benchmark.reporting.presentation import bar_chart_svg, build_view
from benchmark.reporting.service import (
    DEFAULT_FORMATS,
    KNOWN_FORMATS,
    GeneratedFile,
    ReportResult,
    generate_reports,
    manifest_bytes,
    manifest_payload,
)

__all__ = [
    "COMPLETED_STATUS",
    "CSV_FILENAMES",
    "DEFAULT_FORMATS",
    "KNOWN_FORMATS",
    "REPORT_SCHEMA_VERSION",
    "GeneratedFile",
    "PdfResult",
    "ReportBundle",
    "ReportError",
    "ReportResult",
    "bar_chart_svg",
    "build_bundle",
    "build_pdf",
    "build_view",
    "csv_frames",
    "csv_payloads",
    "generate_reports",
    "html_bytes",
    "json_bytes",
    "json_payload",
    "manifest_bytes",
    "manifest_payload",
    "render_html",
    "write_csv_exports",
    "write_html_report",
    "write_json_export",
    "write_pdf_report",
]
