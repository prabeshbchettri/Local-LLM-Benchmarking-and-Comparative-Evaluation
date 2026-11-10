"""
File: test_logging.py

Description:
Unit tests for console and rotating file logging configuration.

Responsibilities:
- Verify log records are written to the configured file.
- Verify duplicate handlers are not created.
- Verify invalid levels are rejected.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmark.core.exceptions import LoggingError
from benchmark.core.logging import configure_logging, get_logger, reset_logging

pytestmark = pytest.mark.unit


def test_logging_writes_to_file(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    configure_logging(level="DEBUG", log_dir=log_dir, log_filename="test.log")
    get_logger("test").debug("hello from test")
    reset_logging()
    content = (log_dir / "test.log").read_text(encoding="utf-8")
    assert "hello from test" in content
    assert "DEBUG" in content
    assert "UTC" in content


def test_logging_is_idempotent(tmp_path: Path) -> None:
    logger = configure_logging(level="INFO", log_dir=tmp_path, log_filename="test.log")
    first_count = len(logger.handlers)
    logger = configure_logging(level="INFO", log_dir=tmp_path, log_filename="test.log")
    second_count = len(logger.handlers)
    reset_logging()
    assert first_count == second_count == 2


def test_invalid_log_level_raises(tmp_path: Path) -> None:
    with pytest.raises(LoggingError, match="Unknown log level"):
        configure_logging(level="NOT_A_LEVEL", log_dir=tmp_path)
    reset_logging()
