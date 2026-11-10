"""
File: test_config.py

Description:
Unit tests for configuration loading, validation, path resolution, and model roster rules.

Responsibilities:
- Verify valid configuration loads.
- Verify missing, malformed, and invalid configuration is rejected.
- Verify paths resolve against the project root.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from benchmark.core.config import AppSettings, load_settings
from benchmark.core.exceptions import ConfigurationError

pytestmark = pytest.mark.unit


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _write_yaml(path: Path, data: dict) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def test_loads_valid_configuration(settings: AppSettings) -> None:
    assert settings.benchmark.name == "local-llm-benchmark"
    assert len(settings.models.candidates) == 3
    assert settings.models.judge.tag == "qwen2.5:3b"
    assert settings.benchmark.dataset_path.is_absolute()
    assert settings.benchmark.dataset_path.name == "benchmark_v1.json"
    assert settings.evaluation.category_judge_weights["summarization"] == 0.40
    assert len(settings.config_hash) == 64


def test_config_hash_is_deterministic(settings: AppSettings, config_dir: Path) -> None:
    reloaded = load_settings(config_dir=config_dir)
    assert reloaded.config_hash == settings.config_hash


def test_config_hash_changes_with_experimental_setting(
    temp_config_dir: Path,
) -> None:
    before = load_settings(config_dir=temp_config_dir)
    path = temp_config_dir / "benchmark.yaml"
    data = _load_yaml(path)
    data["temperature"] = 0.5
    _write_yaml(path, data)
    after = load_settings(config_dir=temp_config_dir)
    assert before.config_hash != after.config_hash


def test_missing_configuration_file_raises(temp_config_dir: Path) -> None:
    (temp_config_dir / "models.yaml").unlink()
    with pytest.raises(ConfigurationError, match="models.yaml"):
        load_settings(config_dir=temp_config_dir)


def test_invalid_yaml_raises(temp_config_dir: Path) -> None:
    (temp_config_dir / "benchmark.yaml").write_text("name: [unclosed\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="Invalid YAML"):
        load_settings(config_dir=temp_config_dir)


def test_unknown_key_raises(temp_config_dir: Path) -> None:
    path = temp_config_dir / "benchmark.yaml"
    data = _load_yaml(path)
    data["unexpected_key"] = True
    _write_yaml(path, data)
    with pytest.raises(ConfigurationError, match="unexpected_key"):
        load_settings(config_dir=temp_config_dir)


def test_invalid_generation_setting_raises(temp_config_dir: Path) -> None:
    path = temp_config_dir / "benchmark.yaml"
    data = _load_yaml(path)
    data["temperature"] = 9.0
    _write_yaml(path, data)
    with pytest.raises(ConfigurationError, match="temperature"):
        load_settings(config_dir=temp_config_dir)


def test_fewer_than_three_candidates_raise(temp_config_dir: Path) -> None:
    path = temp_config_dir / "models.yaml"
    data = _load_yaml(path)
    data["candidates"] = data["candidates"][:2]
    _write_yaml(path, data)
    with pytest.raises(ConfigurationError, match="at least 3"):
        load_settings(config_dir=temp_config_dir)


def test_more_than_three_candidates_raise(temp_config_dir: Path) -> None:
    path = temp_config_dir / "models.yaml"
    data = _load_yaml(path)
    extra = dict(data["candidates"][0])
    extra["name"] = "Extra Candidate"
    extra["tag"] = "extra:1b"
    data["candidates"].append(extra)
    _write_yaml(path, data)
    with pytest.raises(ConfigurationError, match="at most 3"):
        load_settings(config_dir=temp_config_dir)


def test_duplicate_candidate_tags_raise(temp_config_dir: Path) -> None:
    path = temp_config_dir / "models.yaml"
    data = _load_yaml(path)
    data["candidates"][1]["tag"] = data["candidates"][0]["tag"]
    _write_yaml(path, data)
    with pytest.raises(ConfigurationError, match="unique"):
        load_settings(config_dir=temp_config_dir)


def test_judge_cannot_be_a_candidate(temp_config_dir: Path) -> None:
    path = temp_config_dir / "models.yaml"
    data = _load_yaml(path)
    data["judge"]["tag"] = data["candidates"][0]["tag"]
    _write_yaml(path, data)
    with pytest.raises(ConfigurationError, match="judge"):
        load_settings(config_dir=temp_config_dir)


def test_missing_category_weight_raises(temp_config_dir: Path) -> None:
    path = temp_config_dir / "evaluation.yaml"
    data = _load_yaml(path)
    del data["category_judge_weights"]["coding"]
    _write_yaml(path, data)
    with pytest.raises(ConfigurationError, match="missing category"):
        load_settings(config_dir=temp_config_dir)


def test_relative_paths_resolve_against_project_root(tmp_path: Path, config_dir: Path) -> None:
    loaded = load_settings(config_dir=config_dir, project_root=tmp_path)
    assert loaded.benchmark.dataset_path == (
        tmp_path / "data" / "datasets" / "benchmark_v1.json"
    ).resolve()
    assert loaded.benchmark.storage_dir == (tmp_path / "storage").resolve()
