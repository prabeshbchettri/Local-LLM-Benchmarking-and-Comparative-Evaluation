"""
File: config.py

Description:
Typed configuration loading and validation for benchmark, model, and evaluation settings.

Responsibilities:
- Load YAML configuration files.
- Validate values with Pydantic.
- Reject unknown keys and inconsistent model definitions.
- Resolve relative paths against the project root.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import AnyHttpUrl, Field, ValidationError, field_validator, model_validator

from benchmark.core.exceptions import ConfigurationError
from benchmark.core.paths import get_config_dir, get_project_root, resolve_path
from benchmark.models.base import BaseSchema

CATEGORY_NAMES = (
    "general_knowledge",
    "logical_reasoning",
    "mathematical_reasoning",
    "coding",
    "instruction_following",
    "structured_json",
    "summarization",
)

DIGEST_PATTERN = re.compile(r"^(?:sha256:)?[0-9a-f]{64}$")


class BenchmarkSettings(BaseSchema):
    """Settings that control dataset selection, repetitions, and generation."""

    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    dataset_path: Path
    random_seed: int = 42
    request_timeout_seconds: int = Field(default=300, ge=1)
    performance_repeats: int = Field(default=5, ge=1)
    consistency_repeats: int = Field(default=3, ge=1)
    cold_start_repeats: int = Field(default=3, ge=1)
    full_runs: int = Field(default=3, ge=1)
    warmup_requests: int = Field(default=1, ge=0)
    cooldown_seconds_between_models: int = Field(default=60, ge=0)
    cooldown_seconds_between_runs: int = Field(default=300, ge=0)
    hardware_sample_interval_ms: int = Field(default=500, ge=250, le=1000)
    num_ctx: int = Field(default=4096, ge=512)
    num_predict: int = Field(default=512, ge=1)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    top_k: int = Field(default=40, ge=1)
    repeat_penalty: float = Field(default=1.1, gt=0.0)
    storage_dir: Path = Path("storage")
    results_dir: Path = Path("data/results")
    exports_dir: Path = Path("data/exports")
    log_dir: Path = Path("storage/logs")

    @field_validator("dataset_path")
    @classmethod
    def validate_dataset_path(cls, value: Path) -> Path:
        if value.suffix.lower() not in {".json", ".jsonl"}:
            raise ValueError("dataset_path must point to a .json or .jsonl file")
        return value


class ModelSpec(BaseSchema):
    """A candidate or judge model entry."""

    name: str = Field(min_length=1)
    tag: str = Field(min_length=1)
    expected_digest: str | None = None
    expected_quantization: str | None = None

    @field_validator("tag")
    @classmethod
    def validate_tag(cls, value: str) -> str:
        if ":" not in value:
            raise ValueError("model tag must include an explicit tag, for example llama3.1:8b")
        return value

    @field_validator("expected_digest")
    @classmethod
    def validate_digest(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.lower()
        if not DIGEST_PATTERN.fullmatch(normalized):
            raise ValueError("expected_digest must be a 64-character SHA-256 hex digest")
        return normalized


class ModelsSettings(BaseSchema):
    """Ollama connection and model roster settings."""

    ollama_base_url: AnyHttpUrl = "http://127.0.0.1:11434"
    keep_alive: str = Field(default="30m", min_length=1)
    candidates: list[ModelSpec] = Field(min_length=3, max_length=3)
    judge: ModelSpec

    @model_validator(mode="after")
    def validate_roster(self) -> "ModelsSettings":
        candidate_tags = [candidate.tag for candidate in self.candidates]
        if len(set(candidate_tags)) != len(candidate_tags):
            raise ValueError("candidate model tags must be unique")
        candidate_names = [candidate.name for candidate in self.candidates]
        if len(set(candidate_names)) != len(candidate_names):
            raise ValueError("candidate model names must be unique")
        if self.judge.tag in candidate_tags:
            raise ValueError("judge model tag must not also be a candidate tag")
        return self


class EvaluationSettings(BaseSchema):
    """Judge, validation, and scoring settings."""

    rubric_version: str = Field(default="1.0.0", min_length=1)
    judge_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    judge_seed: int = 42
    judge_context_length: int = Field(default=4096, ge=512)
    judge_max_output_tokens: int = Field(default=300, ge=1)
    manual_validation_fraction: float = Field(default=0.1, ge=0.0, le=1.0)
    reliability_sample_fraction: float = Field(default=0.1, ge=0.0, le=1.0)
    reliability_temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    category_judge_weights: dict[str, float] = Field(default_factory=dict)

    @field_validator("category_judge_weights")
    @classmethod
    def validate_category_weights(cls, value: dict[str, float]) -> dict[str, float]:
        missing = set(CATEGORY_NAMES) - set(value)
        unknown = set(value) - set(CATEGORY_NAMES)
        if missing:
            raise ValueError(f"missing category judge weights: {sorted(missing)}")
        if unknown:
            raise ValueError(f"unknown category judge weights: {sorted(unknown)}")
        for category, weight in value.items():
            if not 0.0 <= weight <= 1.0:
                raise ValueError(f"judge weight for {category} must be between 0 and 1")
        return value


class AppSettings(BaseSchema):
    """Fully validated application settings."""

    benchmark: BenchmarkSettings
    models: ModelsSettings
    evaluation: EvaluationSettings
    project_root: Path
    config_dir: Path
    config_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigurationError(f"Configuration file not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigurationError(f"Could not read configuration file {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"Invalid YAML in {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigurationError(f"Configuration file must contain a mapping: {path}")
    return raw


def _validate(schema: type[BaseSchema], data: dict[str, Any], source: Path) -> Any:
    try:
        return schema.model_validate(data)
    except ValidationError as exc:
        raise ConfigurationError(f"Invalid configuration in {source}: {exc}") from exc


def _resolve_benchmark_paths(settings: BenchmarkSettings, project_root: Path) -> BenchmarkSettings:
    return settings.model_copy(
        update={
            "dataset_path": resolve_path(settings.dataset_path, project_root),
            "storage_dir": resolve_path(settings.storage_dir, project_root),
            "results_dir": resolve_path(settings.results_dir, project_root),
            "exports_dir": resolve_path(settings.exports_dir, project_root),
            "log_dir": resolve_path(settings.log_dir, project_root),
        }
    )


def compute_config_hash(
    benchmark: BenchmarkSettings,
    models: ModelsSettings,
    evaluation: EvaluationSettings,
    project_root: Path,
) -> str:
    """Compute a stable hash of the experimental configuration."""
    benchmark_data = benchmark.model_dump(
        mode="json",
        exclude={"storage_dir", "results_dir", "exports_dir", "log_dir"},
    )
    dataset_path = Path(benchmark_data["dataset_path"])
    try:
        benchmark_data["dataset_path"] = dataset_path.relative_to(project_root).as_posix()
    except ValueError:
        benchmark_data["dataset_path"] = dataset_path.as_posix()
    payload = {
        "schema": "1",
        "benchmark": benchmark_data,
        "models": models.model_dump(mode="json"),
        "evaluation": evaluation.model_dump(mode="json"),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_settings(config_dir: Path | None = None, project_root: Path | None = None) -> AppSettings:
    """Load and validate all configuration files."""
    root = (project_root or get_project_root()).resolve()
    if config_dir is None:
        directory = get_config_dir(root)
    else:
        directory = config_dir.expanduser()
        if not directory.is_absolute():
            directory = root / directory
        directory = directory.resolve()
    if not directory.is_dir():
        raise ConfigurationError(f"Configuration directory not found: {directory}")
    benchmark_path = directory / "benchmark.yaml"
    models_path = directory / "models.yaml"
    evaluation_path = directory / "evaluation.yaml"
    benchmark = _validate(BenchmarkSettings, _load_yaml(benchmark_path), benchmark_path)
    models = _validate(ModelsSettings, _load_yaml(models_path), models_path)
    evaluation = _validate(EvaluationSettings, _load_yaml(evaluation_path), evaluation_path)
    benchmark = _resolve_benchmark_paths(benchmark, root)
    return AppSettings(
        benchmark=benchmark,
        models=models,
        evaluation=evaluation,
        project_root=root,
        config_dir=directory,
        config_hash=compute_config_hash(benchmark, models, evaluation, root),
    )
