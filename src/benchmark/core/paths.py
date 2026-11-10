"""
File: paths.py

Description:
Path resolution helpers for the project root, configuration directory, and runtime paths.

Responsibilities:
- Detect the project root from the source tree.
- Honor BENCHMARK_PROJECT_ROOT and BENCHMARK_CONFIG_DIR.
- Resolve relative paths against the project root.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT_ENV = "BENCHMARK_PROJECT_ROOT"
CONFIG_DIR_ENV = "BENCHMARK_CONFIG_DIR"


def get_project_root() -> Path:
    """Return the resolved project root, honoring BENCHMARK_PROJECT_ROOT."""
    override = os.getenv(PROJECT_ROOT_ENV)
    if override:
        return Path(override).expanduser().resolve()
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return current.parents[3]


def get_config_dir(project_root: Path | None = None) -> Path:
    """Return the resolved configuration directory."""
    root = (project_root or get_project_root()).resolve()
    override = os.getenv(CONFIG_DIR_ENV)
    if override:
        candidate = Path(override).expanduser()
        if not candidate.is_absolute():
            candidate = root / candidate
        return candidate.resolve()
    return (root / "configs").resolve()


def resolve_path(value: str | Path, project_root: Path) -> Path:
    """Resolve a configuration path against the project root."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()
