"""
File: service.py

Description:
One-call report generation for a completed experiment.

Why this file exists:
The CLI and the dashboard must produce identical files from the same bundle,
with the same names, hashes, and preliminary flag.

Responsibilities:
- Build the report bundle once and feed every exporter.
- Write CSV, JSON, HTML, and PDF files into the experiment's export directory.
- Record file sizes and SHA-256 hashes in a manifest for reproducibility.
- Reject unknown formats and refuse incomplete experiments by default.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from benchmark.core.config import AppSettings
from benchmark.core.exceptions import ReportError
from benchmark.database.db import Database
from benchmark.reporting.bundle import REPORT_SCHEMA_VERSION, ReportBundle, build_bundle
from benchmark.reporting.csv_export import csv_payloads
from benchmark.reporting.html_report import html_bytes
from benchmark.reporting.json_export import json_bytes
from benchmark.reporting.pdf_report import build_pdf

DEFAULT_FORMATS = ("csv", "json", "html", "pdf")
KNOWN_FORMATS = frozenset(DEFAULT_FORMATS)


@dataclass
class GeneratedFile:
    """One written report file with its integrity hash."""

    format: str
    name: str
    path: Path
    size_bytes: int
    sha256: str


@dataclass
class ReportResult:
    """Summary of one report generation run."""

    experiment_id: str
    output_dir: Path
    generated_at: datetime
    preliminary: bool
    files: list[GeneratedFile] = field(default_factory=list)
    manifest_path: Path | None = None
    warnings: list[str] = field(default_factory=list)
    page_count: int | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable summary."""
        return {
            "experiment_id": self.experiment_id,
            "output_dir": str(self.output_dir),
            "generated_at": self.generated_at.isoformat(),
            "preliminary": self.preliminary,
            "page_count": self.page_count,
            "warnings": self.warnings,
            "manifest": str(self.manifest_path) if self.manifest_path else None,
            "files": [
                {
                    "format": item.format,
                    "name": item.name,
                    "path": str(item.path),
                    "size_bytes": item.size_bytes,
                    "sha256": item.sha256,
                }
                for item in self.files
            ],
        }


def file_entry(name: str, format_name: str, data: bytes) -> dict[str, Any]:
    """Build one manifest file entry from in-memory bytes."""
    return {
        "name": name,
        "format": format_name,
        "size_bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def manifest_payload(
    bundle: ReportBundle,
    entries: Sequence[dict[str, Any]],
    *,
    warnings: Sequence[str] = (),
    page_count: int | None = None,
) -> dict[str, Any]:
    """Build the reproducibility manifest payload."""
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "experiment_id": bundle.experiment_id,
        "experiment_status": bundle.experiment.status,
        "preliminary": bundle.preliminary,
        "profile": bundle.profile,
        "generated_at": bundle.generated_at.isoformat(),
        "source_database": bundle.database_path or "storage/benchmark.db",
        "filters": {
            "run_profile": bundle.run_profile,
            "include_cold_start": bundle.include_cold_start,
            "include_smoke": bundle.include_smoke,
        },
        "page_count": page_count,
        "warnings": list(warnings),
        "files": list(entries),
    }


def manifest_bytes(
    bundle: ReportBundle,
    entries: Sequence[dict[str, Any]],
    *,
    warnings: Sequence[str] = (),
    page_count: int | None = None,
) -> bytes:
    """Return the manifest as UTF-8 JSON bytes."""
    return json.dumps(
        manifest_payload(
            bundle, entries, warnings=warnings, page_count=page_count
        ),
        ensure_ascii=False,
        indent=2,
    ).encode("utf-8")


def _normalize_formats(formats: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(dict.fromkeys(item.strip().lower() for item in formats))
    unknown = sorted(set(normalized) - KNOWN_FORMATS)
    if unknown:
        raise ReportError(
            f"unknown report format(s): {', '.join(unknown)}; "
            f"choose from {', '.join(sorted(KNOWN_FORMATS))}"
        )
    if not normalized:
        raise ReportError("at least one report format is required")
    return normalized


def generate_reports(
    database: Database,
    experiment_id: str,
    *,
    settings: AppSettings | None = None,
    output_dir: Path | None = None,
    formats: Sequence[str] = DEFAULT_FORMATS,
    profile: str | None = None,
    allow_incomplete: bool = False,
    run_profile: str | None = None,
    include_cold_start: bool = False,
    include_smoke: bool = False,
) -> ReportResult:
    """Generate the requested report files for one experiment."""
    requested = _normalize_formats(formats)
    bundle = build_bundle(
        database,
        experiment_id,
        settings=settings,
        profile=profile,
        allow_incomplete=allow_incomplete,
        run_profile=run_profile,
        include_cold_start=include_cold_start,
        include_smoke=include_smoke,
    )
    if output_dir is None:
        base = (
            settings.benchmark.exports_dir
            if settings is not None
            else Path("data") / "exports"
        )
        output_dir = base / experiment_id
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files: list[GeneratedFile] = []
    entries: list[dict[str, Any]] = []
    warnings: list[str] = []
    page_count: int | None = None

    def _write(name: str, format_name: str, data: bytes) -> None:
        path = output_dir / name
        path.write_bytes(data)
        entry = file_entry(name, format_name, data)
        entries.append(entry)
        files.append(
            GeneratedFile(
                format=format_name,
                name=name,
                path=path,
                size_bytes=entry["size_bytes"],
                sha256=entry["sha256"],
            )
        )

    if "csv" in requested:
        for name, data in csv_payloads(bundle).items():
            _write(name, "csv", data)
    if "json" in requested:
        _write(f"{experiment_id}.json", "json", json_bytes(bundle))
    if "html" in requested:
        _write(f"{experiment_id}.html", "html", html_bytes(bundle))
    if "pdf" in requested:
        pdf = build_pdf(bundle)
        page_count = pdf.page_count
        warnings.extend(pdf.warnings)
        _write(f"{experiment_id}.pdf", "pdf", pdf.data)

    manifest = manifest_bytes(bundle, entries, warnings=warnings, page_count=page_count)
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_bytes(manifest)
    return ReportResult(
        experiment_id=experiment_id,
        output_dir=output_dir,
        generated_at=bundle.generated_at,
        preliminary=bundle.preliminary,
        files=files,
        manifest_path=manifest_path,
        warnings=warnings,
        page_count=page_count,
    )


__all__ = [
    "DEFAULT_FORMATS",
    "KNOWN_FORMATS",
    "GeneratedFile",
    "ReportResult",
    "file_entry",
    "generate_reports",
    "manifest_bytes",
    "manifest_payload",
]
