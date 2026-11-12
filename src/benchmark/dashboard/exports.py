"""
File: exports.py

Description:
Dashboard export controls and cached in-memory report payloads.

Why this file exists:
Operators should be able to download CSV, JSON, HTML, and PDF reports from the
dashboard without running the CLI. Generation is lazy and cached so page reruns
stay fast.

Responsibilities:
- Build report payloads from the database on demand.
- Cache payloads by database version, configuration, experiment, and profile.
- Render download controls and refuse to export incomplete experiments.
"""

from __future__ import annotations

import zipfile
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path

import streamlit as st

from benchmark.core.config import load_settings
from benchmark.core.exceptions import BenchmarkError
from benchmark.dashboard import data, ui
from benchmark.database.db import Database
from benchmark.database.repositories import ExperimentView
from benchmark.reporting.bundle import build_bundle
from benchmark.reporting.csv_export import csv_payloads
from benchmark.reporting.html_report import html_bytes
from benchmark.reporting.json_export import json_bytes
from benchmark.reporting.pdf_report import build_pdf
from benchmark.reporting.service import file_entry, manifest_bytes


@dataclass
class ExportPayloads:
    """Every generated report file held in memory for one experiment."""

    experiment_id: str
    csv_files: dict[str, bytes] = field(default_factory=dict)
    json_bytes: bytes = b""
    html_bytes: bytes = b""
    pdf_bytes: bytes = b""
    manifest_bytes: bytes = b""
    zip_bytes: bytes = b""
    warnings: tuple[str, ...] = ()


def build_export_payloads(
    database_path: str | Path,
    config_dir: str | Path,
    experiment_id: str,
    profile: str | None = None,
) -> ExportPayloads:
    """Build every report payload in memory from stored rows."""
    settings = load_settings(config_dir=Path(config_dir))
    database = Database(Path(database_path))
    database.initialize()
    try:
        bundle = build_bundle(database, experiment_id, settings=settings, profile=profile)
        csv_files = csv_payloads(bundle)
        json_data = json_bytes(bundle)
        html_data = html_bytes(bundle)
        pdf = build_pdf(bundle)
        entries = [
            file_entry(name, "csv", payload) for name, payload in csv_files.items()
        ]
        entries.extend(
            [
                file_entry(f"{experiment_id}.json", "json", json_data),
                file_entry(f"{experiment_id}.html", "html", html_data),
                file_entry(f"{experiment_id}.pdf", "pdf", pdf.data),
            ]
        )
        manifest = manifest_bytes(
            bundle, entries, warnings=pdf.warnings, page_count=pdf.page_count
        )
        buffer = BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            for name, payload in csv_files.items():
                archive.writestr(name, payload)
            archive.writestr(f"{experiment_id}.json", json_data)
            archive.writestr(f"{experiment_id}.html", html_data)
            archive.writestr(f"{experiment_id}.pdf", pdf.data)
            archive.writestr("manifest.json", manifest)
        return ExportPayloads(
            experiment_id=experiment_id,
            csv_files=csv_files,
            json_bytes=json_data,
            html_bytes=html_data,
            pdf_bytes=pdf.data,
            manifest_bytes=manifest,
            zip_bytes=buffer.getvalue(),
            warnings=tuple(pdf.warnings),
        )
    finally:
        database.dispose()


@st.cache_data(ttl=30, show_spinner=False)
def _cached_payloads(
    database_path: str,
    config_dir: str,
    version: str,
    experiment_id: str,
    profile: str | None,
) -> ExportPayloads:
    """Cache report payloads until the database or configuration changes."""
    return build_export_payloads(database_path, config_dir, experiment_id, profile)


def export_payloads(
    context: data.DashboardContext,
    experiment_id: str,
    profile: str | None = None,
) -> ExportPayloads:
    """Return cached export payloads for one experiment."""
    return _cached_payloads(
        str(context.database.path),
        str(context.settings.config_dir),
        data.db_version(context),
        experiment_id,
        profile,
    )


def render_export_panel(context: data.DashboardContext, view: ExperimentView) -> None:
    """Render export controls for one experiment."""
    ui.section(
        "Export reports",
        "CSV, JSON, HTML, and PDF files generated from stored rows. Incomplete "
        "experiments are never exported as final results.",
    )
    if view.status != "completed":
        st.info(
            f"Experiment status is {view.status}; export is disabled because "
            "incomplete data must not be presented as final results."
        )
        return
    state_key = f"export_payloads_{view.experiment_id}"
    if st.button(
        "Prepare export files",
        key=f"export_button_{view.experiment_id}",
        help="Build the report files in memory. Nothing is written to disk.",
    ):
        with st.spinner("Building report files from stored rows…"):
            try:
                st.session_state[state_key] = export_payloads(context, view.experiment_id)
            except BenchmarkError as exc:
                st.error(f"The export could not be generated: {exc}")
                st.session_state.pop(state_key, None)
    payloads = st.session_state.get(state_key)
    if payloads is None:
        st.caption(
            "Files are generated on demand from the database; large experiments can "
            "take a moment."
        )
        return
    columns = st.columns(4)
    columns[0].download_button(
        "ZIP bundle",
        data=payloads.zip_bytes,
        file_name=f"{view.experiment_id}_reports.zip",
        mime="application/zip",
        width="stretch",
    )
    columns[1].download_button(
        "JSON",
        data=payloads.json_bytes,
        file_name=f"{view.experiment_id}_report.json",
        mime="application/json",
        width="stretch",
    )
    columns[2].download_button(
        "HTML",
        data=payloads.html_bytes,
        file_name=f"{view.experiment_id}_report.html",
        mime="text/html",
        width="stretch",
    )
    columns[3].download_button(
        "PDF",
        data=payloads.pdf_bytes,
        file_name=f"{view.experiment_id}_report.pdf",
        mime="application/pdf",
        width="stretch",
    )
    with st.expander("Individual CSV files"):
        for name, payload in payloads.csv_files.items():
            st.download_button(
                name,
                data=payload,
                file_name=name,
                mime="text/csv",
                key=f"csv_{view.experiment_id}_{name}",
                width="stretch",
            )
    if payloads.warnings:
        ui.limitations(payloads.warnings)


__all__ = [
    "ExportPayloads",
    "build_export_payloads",
    "export_payloads",
    "render_export_panel",
]
