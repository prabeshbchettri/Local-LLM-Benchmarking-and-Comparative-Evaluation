"""
File: startup.py

Description:
Application initialization that prepares runtime directories and logging without starting a benchmark.

Responsibilities:
- Create storage, log, result, and export directories.
- Configure logging from BENCHMARK_LOG_LEVEL.
- Record a startup log entry.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from benchmark.core.config import AppSettings
from benchmark.core.exceptions import StartupError
from benchmark.core.logging import configure_logging

LOG_LEVEL_ENV = "BENCHMARK_LOG_LEVEL"


def ensure_runtime_directories(settings: AppSettings) -> tuple[Path, ...]:
    """Create and return the runtime directories required by the application."""
    directories = (
        settings.benchmark.storage_dir,
        settings.benchmark.log_dir,
        settings.benchmark.results_dir,
        settings.benchmark.exports_dir,
    )
    for directory in directories:
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise StartupError(f"Could not create runtime directory {directory}: {exc}") from exc
    return directories


def initialize_application(settings: AppSettings) -> logging.Logger:
    """Prepare directories and logging, returning the configured logger."""
    directories = ensure_runtime_directories(settings)
    level = os.getenv(LOG_LEVEL_ENV, "INFO")
    logger = configure_logging(level=level, log_dir=settings.benchmark.log_dir)
    logger.info(
        "Application initialized | project_root=%s | config_dir=%s | runtime_dirs=%s",
        settings.project_root,
        settings.config_dir,
        ", ".join(str(path) for path in directories),
    )
    return logger
