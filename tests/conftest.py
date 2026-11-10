"""
File: conftest.py

Description:
Shared pytest fixtures for configuration, settings, and isolated runtime directories.

Responsibilities:
- Provide the repository configuration directory.
- Provide validated settings.
- Provide a temporary copy of configuration files with isolated paths.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml

from benchmark.core.config import AppSettings, load_settings
from benchmark.core.paths import get_project_root


@pytest.fixture(scope="session")
def project_root() -> Path:
    return get_project_root()


@pytest.fixture(scope="session")
def config_dir(project_root: Path) -> Path:
    return project_root / "configs"


@pytest.fixture()
def settings(config_dir: Path) -> AppSettings:
    return load_settings(config_dir=config_dir)


@pytest.fixture()
def temp_config_dir(tmp_path: Path, config_dir: Path) -> Path:
    target = tmp_path / "configs"
    shutil.copytree(config_dir, target)
    benchmark_path = target / "benchmark.yaml"
    data = yaml.safe_load(benchmark_path.read_text(encoding="utf-8"))
    data.update(
        {
            "dataset_path": str(tmp_path / "data" / "datasets" / "benchmark_v1.0.0.jsonl"),
            "storage_dir": str(tmp_path / "storage"),
            "results_dir": str(tmp_path / "data" / "results"),
            "exports_dir": str(tmp_path / "data" / "exports"),
            "log_dir": str(tmp_path / "storage" / "logs"),
        }
    )
    benchmark_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return target


@pytest.fixture()
def temp_settings(temp_config_dir: Path, tmp_path: Path) -> AppSettings:
    return load_settings(config_dir=temp_config_dir, project_root=tmp_path)
