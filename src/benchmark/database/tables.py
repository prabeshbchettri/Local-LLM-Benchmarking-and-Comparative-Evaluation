"""
File: tables.py

Description:
SQLAlchemy table definitions for models, datasets, experiments, runs, responses,
performance metrics, hardware measurements, evaluations, recommendations, and errors.

Why this file exists:
Every experiment artifact must be stored once, linked by foreign keys, and kept
queryable without overwriting earlier results.

Responsibilities:
- Define the persistent schema with primary keys, foreign keys, and indexes.
- Enforce uniqueness for responses, evaluations, and experiment runs.
- Represent measured values as nullable columns because unavailable metrics are unknown.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Declarative base for all benchmark tables."""


class SchemaVersionRow(Base):
    """One applied database schema version."""

    __tablename__ = "schema_versions"

    id: Mapped[int] = mapped_column(primary_key=True)
    version: Mapped[str] = mapped_column(String(32), unique=True)
    applied_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class ModelRow(Base):
    """A candidate or judge model artifact and its metadata."""

    __tablename__ = "models"

    id: Mapped[int] = mapped_column(primary_key=True)
    tag: Mapped[str] = mapped_column(String(200), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(200))
    role: Mapped[str] = mapped_column(String(20), index=True)
    installed: Mapped[bool] = mapped_column(Boolean, default=False)
    digest: Mapped[str | None] = mapped_column(String(64), index=True)
    family: Mapped[str | None] = mapped_column(String(80))
    parameter_count: Mapped[int | None] = mapped_column(Integer)
    parameter_size: Mapped[str | None] = mapped_column(String(40))
    quantization: Mapped[str | None] = mapped_column(String(40))
    size_bytes: Mapped[int | None] = mapped_column(Integer)
    context_length: Mapped[int | None] = mapped_column(Integer)
    embedding_length: Mapped[int | None] = mapped_column(Integer)
    license: Mapped[str | None] = mapped_column(Text)
    ollama_version: Mapped[str | None] = mapped_column(String(40))
    model_info: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    first_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    last_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))

    runs: Mapped[list["RunRow"]] = relationship(back_populates="model")
    recommendations: Mapped[list["RecommendationRow"]] = relationship(back_populates="model")


class DatasetRow(Base):
    """A registered dataset version and its verified content hash."""

    __tablename__ = "datasets"
    __table_args__ = (UniqueConstraint("name", "version", name="uq_dataset_name_version"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(120))
    version: Mapped[str] = mapped_column(String(40))
    schema_version: Mapped[str] = mapped_column(String(40))
    content_sha256: Mapped[str] = mapped_column(String(64), index=True)
    item_count: Mapped[int] = mapped_column(Integer)
    source_path: Mapped[str | None] = mapped_column(Text)
    random_seed: Mapped[int] = mapped_column(Integer, default=42)
    registered_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))

    experiments: Mapped[list["ExperimentRow"]] = relationship(back_populates="dataset")


class ExperimentRow(Base):
    """One immutable experiment configuration and its execution state."""

    __tablename__ = "experiments"

    id: Mapped[int] = mapped_column(primary_key=True)
    experiment_id: Mapped[str] = mapped_column(String(80), unique=True, index=True)
    name: Mapped[str | None] = mapped_column(String(200))
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id"), index=True)
    config_hash: Mapped[str] = mapped_column(String(64), index=True)
    status: Mapped[str] = mapped_column(String(20), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    config_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    host_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    software_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    notes: Mapped[str | None] = mapped_column(Text)

    dataset: Mapped[DatasetRow] = relationship(back_populates="experiments")
    runs: Mapped[list["RunRow"]] = relationship(back_populates="experiment")
    recommendations: Mapped[list["RecommendationRow"]] = relationship(back_populates="experiment")
    errors: Mapped[list["ErrorRow"]] = relationship(back_populates="experiment")


class RunRow(Base):
    """One model's execution group inside an experiment."""

    __tablename__ = "runs"
    __table_args__ = (
        UniqueConstraint(
            "experiment_id",
            "model_id",
            "profile",
            "cold_start",
            name="uq_run_identity",
        ),
        Index("ix_runs_experiment_status", "experiment_id", "status"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    experiment_id: Mapped[int] = mapped_column(ForeignKey("experiments.id"), index=True)
    model_id: Mapped[int] = mapped_column(ForeignKey("models.id"), index=True)
    profile: Mapped[str] = mapped_column(String(20), index=True)
    status: Mapped[str] = mapped_column(String(20), index=True)
    cold_start: Mapped[bool] = mapped_column(Boolean, default=False)
    warmup_requests: Mapped[int] = mapped_column(Integer, default=0)
    repetitions: Mapped[int] = mapped_column(Integer, default=1)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    notes: Mapped[str | None] = mapped_column(Text)
    error_message: Mapped[str | None] = mapped_column(Text)

    experiment: Mapped[ExperimentRow] = relationship(back_populates="runs")
    model: Mapped[ModelRow] = relationship(back_populates="runs")
    responses: Mapped[list["ResponseRow"]] = relationship(back_populates="run")


class ResponseRow(Base):
    """One stored prompt-response record, including failed and warm-up requests."""

    __tablename__ = "responses"
    __table_args__ = (
        UniqueConstraint(
            "run_id",
            "question_id",
            "repetition",
            "is_warmup",
            name="uq_response_identity",
        ),
        Index("ix_responses_run_question", "run_id", "question_id"),
        Index("ix_responses_question", "question_id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id"), index=True)
    response_uuid: Mapped[str] = mapped_column(String(64), unique=True)
    question_id: Mapped[str] = mapped_column(String(40))
    category: Mapped[str | None] = mapped_column(String(60), index=True)
    prompt: Mapped[str] = mapped_column(Text)
    system_prompt: Mapped[str | None] = mapped_column(Text)
    response_text: Mapped[str] = mapped_column(Text, default="")
    finish_reason: Mapped[str | None] = mapped_column(String(60))
    status: Mapped[str] = mapped_column(String(20), index=True)
    error_type: Mapped[str | None] = mapped_column(String(120))
    error_message: Mapped[str | None] = mapped_column(Text)
    is_warmup: Mapped[bool] = mapped_column(Boolean, default=False)
    cold_start: Mapped[bool] = mapped_column(Boolean, default=False)
    repetition: Mapped[int] = mapped_column(Integer, default=1)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    stream_chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    config_hash: Mapped[str | None] = mapped_column(String(64))
    raw_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))

    run: Mapped[RunRow] = relationship(back_populates="responses")
    performance: Mapped["PerformanceMetricRow | None"] = relationship(
        back_populates="response",
        uselist=False,
    )
    hardware: Mapped["HardwareMeasurementRow | None"] = relationship(
        back_populates="response",
        uselist=False,
    )
    evaluation: Mapped["EvaluationRow | None"] = relationship(
        back_populates="response",
        uselist=False,
    )


class PerformanceMetricRow(Base):
    """Client and Ollama timing plus token counts for one response."""

    __tablename__ = "performance_metrics"

    id: Mapped[int] = mapped_column(primary_key=True)
    response_id: Mapped[int] = mapped_column(ForeignKey("responses.id"), unique=True)
    ttft_seconds: Mapped[float | None] = mapped_column(Float)
    total_latency_seconds: Mapped[float | None] = mapped_column(Float)
    load_duration_seconds: Mapped[float | None] = mapped_column(Float)
    prompt_eval_duration_seconds: Mapped[float | None] = mapped_column(Float)
    eval_duration_seconds: Mapped[float | None] = mapped_column(Float)
    prompt_eval_count: Mapped[int | None] = mapped_column(Integer)
    eval_count: Mapped[int | None] = mapped_column(Integer)
    output_tokens_per_second: Mapped[float | None] = mapped_column(Float)
    prompt_tokens_per_second: Mapped[float | None] = mapped_column(Float)
    options_json: Mapped[dict[str, Any]] = mapped_column(JSON)

    response: Mapped[ResponseRow] = relationship(back_populates="performance")


class HardwareMeasurementRow(Base):
    """Hardware summary and raw samples attached to one response."""

    __tablename__ = "hardware_measurements"

    id: Mapped[int] = mapped_column(primary_key=True)
    response_id: Mapped[int] = mapped_column(ForeignKey("responses.id"), unique=True)
    schema_version: Mapped[str] = mapped_column(String(20))
    sampling_interval_ms: Mapped[int] = mapped_column(Integer)
    baseline_seconds: Mapped[float] = mapped_column(Float)
    sample_count: Mapped[int] = mapped_column(Integer)
    baseline_sample_count: Mapped[int] = mapped_column(Integer)
    inference_sample_count: Mapped[int] = mapped_column(Integer)
    cpu_avg_percent: Mapped[float | None] = mapped_column(Float)
    cpu_peak_percent: Mapped[float | None] = mapped_column(Float)
    ram_avg_used_bytes: Mapped[int | None] = mapped_column(Integer)
    ram_peak_used_bytes: Mapped[int | None] = mapped_column(Integer)
    gpu_utilization_avg_percent: Mapped[float | None] = mapped_column(Float)
    gpu_utilization_peak_percent: Mapped[float | None] = mapped_column(Float)
    vram_avg_used_bytes: Mapped[int | None] = mapped_column(Integer)
    vram_peak_used_bytes: Mapped[int | None] = mapped_column(Integer)
    gpu_temperature_avg_c: Mapped[float | None] = mapped_column(Float)
    gpu_temperature_peak_c: Mapped[float | None] = mapped_column(Float)
    ollama_rss_avg_bytes: Mapped[int | None] = mapped_column(Integer)
    ollama_rss_peak_bytes: Mapped[int | None] = mapped_column(Integer)
    ollama_model_size_vram_bytes: Mapped[int | None] = mapped_column(Integer)
    ollama_model_vram_ratio: Mapped[float | None] = mapped_column(Float)
    gpu_available: Mapped[bool] = mapped_column(Boolean, default=False)
    gpu_utilization_available: Mapped[bool] = mapped_column(Boolean, default=False)
    gpu_memory_available: Mapped[bool] = mapped_column(Boolean, default=False)
    gpu_temperature_available: Mapped[bool] = mapped_column(Boolean, default=False)
    ollama_process_available: Mapped[bool] = mapped_column(Boolean, default=False)
    ollama_model_vram_available: Mapped[bool] = mapped_column(Boolean, default=False)
    gpu_name: Mapped[str | None] = mapped_column(String(120))
    gpu_source: Mapped[str | None] = mapped_column(String(20))
    sampler_overhead_avg_ms: Mapped[float | None] = mapped_column(Float)
    sampler_overhead_peak_ms: Mapped[float | None] = mapped_column(Float)
    gpu_error: Mapped[str | None] = mapped_column(Text)
    sampler_error: Mapped[str | None] = mapped_column(Text)
    attribution_note: Mapped[str | None] = mapped_column(Text)
    summary_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    samples_json: Mapped[list[dict[str, Any]]] = mapped_column(JSON)

    response: Mapped[ResponseRow] = relationship(back_populates="hardware")


class EvaluationRow(Base):
    """Deterministic and judge evaluation attached to one response."""

    __tablename__ = "evaluations"
    __table_args__ = (Index("ix_evaluations_question", "question_id"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    response_id: Mapped[int] = mapped_column(ForeignKey("responses.id"), unique=True)
    question_id: Mapped[str] = mapped_column(String(40))
    category: Mapped[str | None] = mapped_column(String(60), index=True)
    evaluation_type: Mapped[str] = mapped_column(String(40))
    evaluator: Mapped[str | None] = mapped_column(String(80))
    evaluator_version: Mapped[str | None] = mapped_column(String(20))
    status: Mapped[str] = mapped_column(String(20), index=True)
    deterministic_score: Mapped[float | None] = mapped_column(Float)
    deterministic_max_score: Mapped[float | None] = mapped_column(Float)
    deterministic_normalized_score: Mapped[float | None] = mapped_column(Float)
    passed: Mapped[bool | None] = mapped_column(Boolean)
    judge_status: Mapped[str | None] = mapped_column(String(30))
    judge_model_tag: Mapped[str | None] = mapped_column(String(200), index=True)
    judge_model_digest: Mapped[str | None] = mapped_column(String(64))
    judge_same_family: Mapped[bool | None] = mapped_column(Boolean)
    judge_raw_score: Mapped[float | None] = mapped_column(Float)
    judge_normalized_score: Mapped[float | None] = mapped_column(Float)
    judge_missing: Mapped[bool] = mapped_column(Boolean, default=True)
    composite_score: Mapped[float | None] = mapped_column(Float)
    checks_json: Mapped[list[dict[str, Any]]] = mapped_column(JSON)
    judge_dimensions_json: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    judge_config_json: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    errors_json: Mapped[list[dict[str, Any]]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))

    response: Mapped[ResponseRow] = relationship(back_populates="evaluation")


class RecommendationRow(Base):
    """A workload-specific recommendation produced from stored results."""

    __tablename__ = "recommendations"
    __table_args__ = (
        UniqueConstraint(
            "experiment_id",
            "workload",
            "rank",
            name="uq_recommendation_rank",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    experiment_id: Mapped[int] = mapped_column(ForeignKey("experiments.id"), index=True)
    model_id: Mapped[int] = mapped_column(ForeignKey("models.id"), index=True)
    workload: Mapped[str] = mapped_column(String(80), index=True)
    rank: Mapped[int] = mapped_column(Integer)
    score: Mapped[float] = mapped_column(Float)
    confidence: Mapped[str] = mapped_column(String(20))
    weights_json: Mapped[dict[str, float]] = mapped_column(JSON)
    rationale: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))

    experiment: Mapped[ExperimentRow] = relationship(back_populates="recommendations")
    model: Mapped[ModelRow] = relationship(back_populates="recommendations")


class ErrorRow(Base):
    """A structured failure recorded outside an individual response."""

    __tablename__ = "errors"

    id: Mapped[int] = mapped_column(primary_key=True)
    experiment_id: Mapped[int | None] = mapped_column(
        ForeignKey("experiments.id"),
        index=True,
    )
    run_id: Mapped[int | None] = mapped_column(ForeignKey("runs.id"), index=True)
    stage: Mapped[str] = mapped_column(String(40))
    error_type: Mapped[str] = mapped_column(String(120))
    message: Mapped[str] = mapped_column(Text)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))

    experiment: Mapped[ExperimentRow | None] = relationship(back_populates="errors")
