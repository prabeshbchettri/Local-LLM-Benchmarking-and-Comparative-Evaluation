"""
File: logging.py

Description:
Logging configuration for console and rotating file output.

Responsibilities:
- Configure a single application logger without duplicate handlers.
- Write UTF-8 rotating logs with UTC timestamps under the configured directory.
- Provide child loggers and a reset helper for tests.
"""

from __future__ import annotations

import logging
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path

from benchmark.core.exceptions import LoggingError

LOGGER_NAME = "benchmark"
LOG_FORMAT = "%(asctime)s UTC | %(levelname)s | %(name)s | %(message)s"
MAX_LOG_BYTES = 10_000_000
BACKUP_COUNT = 5


def _resolve_level(level: str) -> int:
    resolved = logging.getLevelName(level.upper())
    if not isinstance(resolved, int):
        raise LoggingError(f"Unknown log level: {level}")
    return resolved


def reset_logging() -> None:
    """Remove and close all handlers on the application logger."""
    logger = logging.getLogger(LOGGER_NAME)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    logger.propagate = False


def configure_logging(
    level: str = "INFO",
    log_dir: Path | None = None,
    log_filename: str = "benchmark.log",
) -> logging.Logger:
    """Configure console and optional rotating file logging."""
    logger = logging.getLogger(LOGGER_NAME)
    reset_logging()
    logger.setLevel(_resolve_level(level))
    formatter = logging.Formatter(LOG_FORMAT)
    formatter.converter = time.gmtime
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)
    if log_dir is not None:
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            file_handler = RotatingFileHandler(
                log_dir / log_filename,
                maxBytes=MAX_LOG_BYTES,
                backupCount=BACKUP_COUNT,
                encoding="utf-8",
            )
        except OSError as exc:
            reset_logging()
            raise LoggingError(f"Could not configure file logging in {log_dir}: {exc}") from exc
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    logger.propagate = False
    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Return the application logger or a child logger."""
    if not name:
        return logging.getLogger(LOGGER_NAME)
    return logging.getLogger(f"{LOGGER_NAME}.{name}")
