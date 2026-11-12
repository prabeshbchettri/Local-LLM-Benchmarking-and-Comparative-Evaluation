"""
File: health.py

Description:
Health checks for the Python environment, configuration, and runtime directories.

Responsibilities:
- Report Python and dependency versions.
- Validate configuration files.
- Check runtime directories and dataset availability.
- Render a human-readable health report.
"""

from __future__ import annotations

import platform
import shutil
import sys
from dataclasses import dataclass
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Literal

import httpx
import jsonschema
import pydantic
import yaml

from benchmark.core.config import AppSettings, load_settings
from benchmark.core.exceptions import BenchmarkError, DatasetError, OllamaError
from benchmark.core.paths import get_config_dir, get_project_root
from benchmark.core.startup import ensure_runtime_directories
from benchmark.execution.ollama_client import OllamaClient

CheckStatus = Literal["ok", "warn", "fail"]
MINIMUM_FREE_DISK_BYTES = 12 * 1024**3


@dataclass(frozen=True)
class CheckResult:
    """A single health check result."""

    name: str
    status: CheckStatus
    detail: str


@dataclass(frozen=True)
class HealthReport:
    """A collection of health check results."""

    checks: tuple[CheckResult, ...]

    @property
    def ok(self) -> bool:
        return all(check.status != "fail" for check in self.checks)

    @property
    def failures(self) -> tuple[CheckResult, ...]:
        return tuple(check for check in self.checks if check.status == "fail")

    @property
    def warnings(self) -> tuple[CheckResult, ...]:
        return tuple(check for check in self.checks if check.status == "warn")


def _python_check() -> CheckResult:
    version = platform.python_version()
    if sys.version_info[:2] == (3, 13):
        return CheckResult("python", "ok", f"Python {version}")
    return CheckResult("python", "fail", f"Python {version}; expected 3.13.x")


def _dependency_checks() -> list[CheckResult]:
    checks = [
        CheckResult("httpx", "ok", f"httpx {httpx.__version__}"),
        CheckResult("jsonschema", "ok", f"jsonschema {package_version('jsonschema')}"),
        CheckResult("pydantic", "ok", f"pydantic {pydantic.VERSION}"),
        CheckResult("pyyaml", "ok", f"PyYAML {yaml.__version__}"),
    ]
    try:
        checks.append(CheckResult("psutil", "ok", f"psutil {package_version('psutil')}"))
    except Exception as exc:
        checks.append(CheckResult("psutil", "fail", f"psutil unavailable: {exc}"))
    try:
        checks.append(
            CheckResult(
                "pynvml",
                "ok",
                f"nvidia-ml-py {package_version('nvidia-ml-py')}",
            )
        )
    except Exception as exc:
        checks.append(
            CheckResult(
                "pynvml",
                "warn",
                f"pynvml unavailable; GPU metrics may be missing: {exc}",
            )
        )
    smi = shutil.which("nvidia-smi")
    checks.append(
        CheckResult(
            "nvidia_smi",
            "ok" if smi else "warn",
            smi or "nvidia-smi not found on PATH; GPU fallback unavailable",
        )
    )
    return checks


def _ollama_check(settings: AppSettings) -> CheckResult:
    client = OllamaClient(
        str(settings.models.ollama_base_url),
        timeout_seconds=3.0,
    )
    try:
        version = client.version().version
        installed = client.list_models()
        configured = [*settings.models.candidates, settings.models.judge]
        installed_tags = {model.tag for model in installed}
        missing = [spec.tag for spec in configured if spec.tag not in installed_tags]
        if missing:
            return CheckResult(
                "ollama_connectivity",
                "warn",
                f"Ollama {version}; {len(installed)} installed; missing: {', '.join(missing)}",
            )
        return CheckResult(
            "ollama_connectivity",
            "ok",
            f"Ollama {version}; all {len(configured)} configured models installed",
        )
    except OllamaError as exc:
        return CheckResult("ollama_connectivity", "warn", f"Unavailable: {exc}")
    finally:
        client.close()


def _disk_check(settings: AppSettings) -> CheckResult:
    try:
        usage = shutil.disk_usage(settings.project_root)
    except OSError as exc:
        return CheckResult("disk_space", "warn", f"Could not check disk space: {exc}")
    free = usage.free
    if free < MINIMUM_FREE_DISK_BYTES:
        return CheckResult(
            "disk_space",
            "warn",
            f"{free / 1024**3:.1f} GiB free; at least 12 GiB recommended",
        )
    return CheckResult("disk_space", "ok", f"{free / 1024**3:.1f} GiB free")


def _dataset_check(settings: AppSettings) -> CheckResult:
    dataset_path = settings.benchmark.dataset_path
    if not dataset_path.is_file():
        return CheckResult(
            "dataset_file",
            "warn",
            f"Not present yet: {dataset_path}; expected in Phase 03",
        )
    try:
        from benchmark.dataset.loader import load_dataset

        dataset = load_dataset(dataset_path)
    except DatasetError as exc:
        return CheckResult("dataset_file", "fail", str(exc))
    return CheckResult(
        "dataset_file",
        "ok",
        f"{dataset.name} v{dataset.version}; {len(dataset.items)} items; "
        f"SHA-256 {dataset.content_sha256[:16]}...",
    )


def _runtime_checks(settings: AppSettings) -> list[CheckResult]:
    checks: list[CheckResult] = []
    try:
        directories = ensure_runtime_directories(settings)
    except BenchmarkError as exc:
        checks.append(CheckResult("runtime_directories", "fail", str(exc)))
        return checks
    checks.append(
        CheckResult(
            "runtime_directories",
            "ok",
            ", ".join(str(path) for path in directories),
        )
    )
    checks.append(_dataset_check(settings))
    checks.append(_disk_check(settings))
    checks.append(_ollama_check(settings))
    return checks


def run_health_check(
    settings: AppSettings | None = None,
    config_dir: Path | None = None,
) -> HealthReport:
    """Run all current health checks and return the report."""
    root = settings.project_root if settings is not None else get_project_root()
    directory = settings.config_dir if settings is not None else (config_dir or get_config_dir(root))
    checks = [_python_check(), *_dependency_checks()]
    if not directory.is_dir():
        checks.append(CheckResult("config_dir", "fail", f"Missing directory: {directory}"))
        return HealthReport(tuple(checks))
    checks.append(CheckResult("config_dir", "ok", str(directory)))
    if settings is None:
        try:
            settings = load_settings(config_dir=directory, project_root=root)
        except BenchmarkError as exc:
            checks.append(CheckResult("config_validation", "fail", str(exc)))
            return HealthReport(tuple(checks))
    checks.append(CheckResult("config_validation", "ok", "Configuration files validated"))
    checks.append(CheckResult("config_hash", "ok", settings.config_hash))
    checks.extend(_runtime_checks(settings))
    return HealthReport(tuple(checks))


def render_health_report(report: HealthReport) -> str:
    """Render a health report as plain text."""
    labels = {"ok": "OK", "warn": "WARN", "fail": "FAIL"}
    lines = [f"[{labels[check.status]}] {check.name}: {check.detail}" for check in report.checks]
    if report.ok:
        summary = "Health check passed" if not report.warnings else "Health check passed with warnings"
    else:
        summary = "Health check failed"
    lines.append(summary)
    return "\n".join(lines)
