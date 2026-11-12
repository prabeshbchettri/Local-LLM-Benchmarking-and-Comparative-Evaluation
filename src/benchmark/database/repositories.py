"""
File: repositories.py

Description:
Small repository layer that isolates SQLAlchemy queries from orchestration logic.

Why this file exists:
Experiment code should ask for "save this response" or "list this history" without
building SQL, and historical records must be rejected rather than overwritten.

Responsibilities:
- Register models, datasets, experiments, runs, responses, metrics, hardware, evaluations.
- Enforce append-only behavior for completed runs and experiments.
- Handle duplicate inserts idempotently or fail loudly on conflicting history.
- Provide retrieval helpers for history, per-model results, and question-level results.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import Field
from sqlalchemy import and_, case, func, select
from sqlalchemy.orm import Session

from benchmark.core.exceptions import (
    DatabaseConflictError,
    DatabaseDuplicateError,
    DatabaseError,
    ExperimentStateError,
)
from benchmark.database.db import Database
from benchmark.database.migrations import utc_now
from benchmark.database.tables import (
    DatasetRow,
    ErrorRow,
    EvaluationRow,
    ExperimentRow,
    HardwareMeasurementRow,
    ModelRow,
    PerformanceMetricRow,
    RecommendationRow,
    ResponseRow,
    RunRow,
)
from benchmark.dataset.models import BenchmarkDataset
from benchmark.evaluation.models import EvaluationResult
from benchmark.execution.metadata import ModelMetadata
from benchmark.models.base import BaseSchema
from benchmark.models.inference import ResponseRecord

OPEN_EXPERIMENT_STATUSES = {"created", "running"}
CLOSED_EXPERIMENT_STATUSES = {"completed", "failed", "cancelled"}


class RegisteredModel(BaseSchema):
    """Stored model identity and metadata used by callers instead of ORM objects."""

    id: int
    tag: str
    name: str
    role: str
    installed: bool
    digest: str | None = None
    family: str | None = None
    parameter_count: int | None = None
    parameter_size: str | None = None
    quantization: str | None = None
    size_bytes: int | None = None
    context_length: int | None = None
    license: str | None = None
    ollama_version: str | None = None


class ExperimentView(BaseSchema):
    """Stored experiment with summary counts."""

    id: int
    experiment_id: str
    name: str | None = None
    status: str
    dataset_name: str
    dataset_version: str
    dataset_hash: str
    config_hash: str
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    run_count: int
    response_count: int
    evaluation_count: int


class RunDetailView(BaseSchema):
    """One stored run with model metadata for report configuration sections."""

    id: int
    model_tag: str
    model_name: str
    role: str
    profile: str
    status: str
    cold_start: bool
    warmup_requests: int
    repetitions: int
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error_message: str | None = None


class ExperimentDetailView(BaseSchema):
    """Full experiment snapshot used by reports and structured exports."""

    experiment: ExperimentView
    config: dict[str, Any] = Field(default_factory=dict)
    host: dict[str, Any] = Field(default_factory=dict)
    software: dict[str, Any] = Field(default_factory=dict)
    notes: str | None = None
    dataset_schema_version: str | None = None
    dataset_item_count: int = 0
    dataset_source_path: str | None = None
    dataset_random_seed: int | None = None
    runs: list[RunDetailView] = Field(default_factory=list)
    models: list[RegisteredModel] = Field(default_factory=list)


class RunView(BaseSchema):
    """One stored model run."""

    id: int
    experiment_id: str
    model_tag: str
    profile: str
    status: str
    cold_start: bool
    warmup_requests: int
    repetitions: int
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error_message: str | None = None


class ModelResultView(BaseSchema):
    """Aggregated results for one model run inside an experiment."""

    run_id: int
    model_tag: str
    model_name: str
    profile: str
    status: str
    cold_start: bool
    measured_responses: int
    successful_responses: int
    failed_responses: int
    mean_latency_seconds: float | None = None
    mean_ttft_seconds: float | None = None
    mean_output_tokens_per_second: float | None = None
    mean_deterministic_score: float | None = None
    mean_judge_score: float | None = None
    mean_composite_score: float | None = None


class QuestionResultView(BaseSchema):
    """One question-level result for inspection."""

    question_id: str
    model_tag: str
    status: str
    repetition: int
    response_text: str
    total_latency_seconds: float | None = None
    ttft_seconds: float | None = None
    deterministic_normalized_score: float | None = None
    judge_normalized_score: float | None = None
    composite_score: float | None = None


class ResponseInspectionView(BaseSchema):
    """One stored response with its prompt, timing, tokens, evaluation, and hardware."""

    question_id: str
    model_tag: str
    model_name: str
    run_profile: str
    cold_start: bool
    repetition: int
    category: str | None = None
    prompt: str
    system_prompt: str | None = None
    response_text: str
    response_status: str
    finish_reason: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    started_at: datetime
    completed_at: datetime
    stream_chunk_count: int = 0
    latency_seconds: float | None = None
    ttft_seconds: float | None = None
    load_duration_seconds: float | None = None
    prompt_eval_duration_seconds: float | None = None
    eval_duration_seconds: float | None = None
    prompt_eval_count: int | None = None
    eval_count: int | None = None
    output_tokens_per_second: float | None = None
    prompt_tokens_per_second: float | None = None
    evaluation_status: str | None = None
    evaluation_type: str | None = None
    deterministic_score: float | None = None
    deterministic_max_score: float | None = None
    deterministic_normalized_score: float | None = None
    passed: bool | None = None
    judge_status: str | None = None
    judge_model_tag: str | None = None
    judge_raw_score: float | None = None
    judge_normalized_score: float | None = None
    judge_missing: bool | None = None
    composite_score: float | None = None
    checks: list[dict[str, Any]] = Field(default_factory=list)
    judge_dimensions: dict[str, Any] | None = None
    evaluation_errors: list[dict[str, Any]] = Field(default_factory=list)
    hardware_summary: dict[str, Any] | None = None


class EvaluationBreakdownView(BaseSchema):
    """Pass counts and score means grouped by model and evaluation type."""

    model_tag: str
    evaluation_type: str | None = None
    evaluated_responses: int = Field(default=0, ge=0)
    passed_responses: int = Field(default=0, ge=0)
    deterministic_score: float | None = None
    judge_score: float | None = None
    composite_score: float | None = None


class HardwareUsageRow(BaseSchema):
    """Per-response average and peak hardware usage stored with one response."""

    model_tag: str
    question_id: str
    repetition: int
    cold_start: bool = False
    sample_count: int = Field(default=0, ge=0)
    baseline_sample_count: int = Field(default=0, ge=0)
    inference_sample_count: int = Field(default=0, ge=0)
    cpu_avg_percent: float | None = None
    cpu_peak_percent: float | None = None
    ram_avg_used_bytes: int | None = None
    ram_peak_used_bytes: int | None = None
    gpu_utilization_avg_percent: float | None = None
    gpu_utilization_peak_percent: float | None = None
    vram_avg_used_bytes: int | None = None
    vram_peak_used_bytes: int | None = None
    gpu_temperature_avg_c: float | None = None
    gpu_temperature_peak_c: float | None = None
    ollama_rss_avg_bytes: int | None = None
    ollama_rss_peak_bytes: int | None = None
    ollama_model_size_vram_bytes: int | None = None
    gpu_name: str | None = None
    gpu_available: bool = False
    gpu_utilization_available: bool = False
    gpu_memory_available: bool = False
    gpu_temperature_available: bool = False
    model_vram_available: bool = False
    gpu_error: str | None = None
    sampler_error: str | None = None
    attribution_note: str | None = None


class AnalyticsRow(BaseSchema):
    """One measured response with metrics, evaluation, hardware, and run context."""

    model_tag: str
    model_name: str
    run_profile: str | None = None
    cold_start: bool = False
    question_id: str
    category: str | None = None
    response_status: str
    repetition: int
    response_text: str
    latency_seconds: float | None = None
    ttft_seconds: float | None = None
    output_tokens_per_second: float | None = None
    prompt_tokens_per_second: float | None = None
    evaluation_status: str | None = None
    evaluation_type: str | None = None
    deterministic_score: float | None = None
    judge_score: float | None = None
    composite_score: float | None = None
    judge_missing: bool | None = None
    hardware_present: bool = False
    cpu_avg_percent: float | None = None
    ram_avg_used_bytes: int | None = None
    gpu_utilization_avg_percent: float | None = None
    vram_avg_used_bytes: int | None = None
    ollama_model_vram_bytes: int | None = None
    ollama_model_vram_available: bool = False
    gpu_available: bool = False
    gpu_memory_available: bool = False
    gpu_name: str | None = None


class PendingEvaluation(BaseSchema):
    """A stored response that still needs evaluation."""

    response_id: int
    question_id: str
    response_text: str
    status: str
    repetition: int


class RecommendationInput(BaseSchema):
    """A recommendation to persist; ranking logic belongs to Phase 08."""

    workload: str = Field(min_length=1)
    rank: int = Field(ge=1)
    model_tag: str = Field(min_length=1)
    score: float
    confidence: str = Field(min_length=1)
    weights: dict[str, float] = Field(default_factory=dict)
    details: dict[str, Any] = Field(default_factory=dict)
    rationale: str = Field(min_length=1)


def _experiment_view(session: Session, row: ExperimentRow) -> ExperimentView:
    dataset = session.get(DatasetRow, row.dataset_id)
    run_count = session.scalar(
        select(func.count()).select_from(RunRow).where(RunRow.experiment_id == row.id)
    )
    response_count = session.scalar(
        select(func.count())
        .select_from(ResponseRow)
        .join(RunRow, RunRow.id == ResponseRow.run_id)
        .where(RunRow.experiment_id == row.id)
    )
    evaluation_count = session.scalar(
        select(func.count())
        .select_from(EvaluationRow)
        .join(ResponseRow, ResponseRow.id == EvaluationRow.response_id)
        .join(RunRow, RunRow.id == ResponseRow.run_id)
        .where(RunRow.experiment_id == row.id)
    )
    return ExperimentView(
        id=row.id,
        experiment_id=row.experiment_id,
        name=row.name,
        status=row.status,
        dataset_name=dataset.name if dataset else "unknown",
        dataset_version=dataset.version if dataset else "unknown",
        dataset_hash=dataset.content_sha256 if dataset else "unknown",
        config_hash=row.config_hash,
        created_at=row.created_at,
        started_at=row.started_at,
        completed_at=row.completed_at,
        run_count=run_count or 0,
        response_count=response_count or 0,
        evaluation_count=evaluation_count or 0,
    )


class BaseRepository:
    """Shared database handle for repositories."""

    def __init__(self, database: Database) -> None:
        self.database = database


def _registered_model(row: ModelRow) -> RegisteredModel:
    """Map an ORM model row to the public read model."""
    return RegisteredModel(
        id=row.id,
        tag=row.tag,
        name=row.name,
        role=row.role,
        installed=row.installed,
        digest=row.digest,
        family=row.family,
        parameter_count=row.parameter_count,
        parameter_size=row.parameter_size,
        quantization=row.quantization,
        size_bytes=row.size_bytes,
        context_length=row.context_length,
        license=row.license,
        ollama_version=row.ollama_version,
    )


class ModelRepository(BaseRepository):
    """Register and look up model artifacts."""

    def register(self, metadata: ModelMetadata) -> RegisteredModel:
        """Insert a model or refresh its metadata without touching historical rows."""
        now = utc_now()
        with self.database.session_scope() as session:
            row = session.scalar(select(ModelRow).where(ModelRow.tag == metadata.tag))
            if row is None:
                row = ModelRow(
                    tag=metadata.tag,
                    name=metadata.name,
                    role=metadata.role.value,
                    installed=metadata.installed,
                    digest=metadata.digest,
                    family=metadata.family,
                    parameter_count=metadata.parameter_count,
                    parameter_size=metadata.parameter_size,
                    quantization=metadata.quantization,
                    size_bytes=metadata.size_bytes,
                    context_length=metadata.context_length,
                    embedding_length=metadata.embedding_length,
                    license=metadata.license,
                    ollama_version=metadata.ollama_version,
                    model_info=metadata.model_info,
                    first_seen_at=now,
                    last_seen_at=now,
                )
                session.add(row)
            else:
                row.name = metadata.name
                row.role = metadata.role.value
                row.installed = metadata.installed
                row.digest = metadata.digest or row.digest
                row.family = metadata.family or row.family
                row.parameter_count = metadata.parameter_count or row.parameter_count
                row.parameter_size = metadata.parameter_size or row.parameter_size
                row.quantization = metadata.quantization or row.quantization
                row.size_bytes = metadata.size_bytes or row.size_bytes
                row.context_length = metadata.context_length or row.context_length
                row.embedding_length = metadata.embedding_length or row.embedding_length
                row.license = metadata.license or row.license
                row.ollama_version = metadata.ollama_version or row.ollama_version
                row.model_info = metadata.model_info or row.model_info
                row.last_seen_at = now
            session.flush()
            return _registered_model(row)

    def get_by_tag(self, tag: str) -> RegisteredModel | None:
        """Return a stored model by exact Ollama tag."""
        with self.database.session_scope() as session:
            row = session.scalar(select(ModelRow).where(ModelRow.tag == tag))
            if row is None:
                return None
            return _registered_model(row)

    def list(self) -> list[RegisteredModel]:
        """Return all registered models ordered by tag."""
        with self.database.session_scope() as session:
            rows = session.scalars(select(ModelRow).order_by(ModelRow.tag)).all()
            return [_registered_model(row) for row in rows]


class DatasetRepository(BaseRepository):
    """Register dataset versions and verify that a version is not redefined."""

    def register(
        self,
        dataset: BenchmarkDataset,
        source_path: Path | None = None,
    ) -> int:
        """Store a dataset version once, rejecting a changed hash for the same version."""
        with self.database.session_scope() as session:
            row = session.scalar(
                select(DatasetRow).where(
                    DatasetRow.name == dataset.name,
                    DatasetRow.version == dataset.version,
                )
            )
            if row is not None:
                if row.content_sha256 != dataset.content_sha256:
                    raise DatabaseConflictError(
                        f"dataset {dataset.name} {dataset.version} is already stored "
                        "with a different content hash"
                    )
                return row.id
            row = DatasetRow(
                name=dataset.name,
                version=dataset.version,
                schema_version=dataset.schema_version,
                content_sha256=dataset.content_sha256,
                item_count=len(dataset.items),
                source_path=str(source_path) if source_path else None,
                random_seed=dataset.random_seed,
                registered_at=utc_now(),
            )
            session.add(row)
            session.flush()
            return row.id

    def content_hash(self, dataset_id: int) -> str | None:
        """Return the stored content hash for one dataset row."""
        with self.database.session_scope() as session:
            row = session.get(DatasetRow, dataset_id)
            return row.content_sha256 if row else None


class ExperimentRepository(BaseRepository):
    """Create and transition experiment records."""

    def create(
        self,
        *,
        experiment_id: str,
        name: str | None,
        dataset_id: int,
        config_hash: str,
        config: dict[str, Any],
        host: dict[str, Any],
        software: dict[str, Any],
        notes: str | None = None,
    ) -> ExperimentView:
        """Insert a new experiment in the created state."""
        with self.database.session_scope() as session:
            existing = session.scalar(
                select(ExperimentRow).where(ExperimentRow.experiment_id == experiment_id)
            )
            if existing is not None:
                return _experiment_view(session, existing)
            row = ExperimentRow(
                experiment_id=experiment_id,
                name=name,
                dataset_id=dataset_id,
                config_hash=config_hash,
                status="created",
                created_at=utc_now(),
                config_json=config,
                host_json=host,
                software_json=software,
                notes=notes,
            )
            session.add(row)
            session.flush()
            return _experiment_view(session, row)

    def get(self, experiment_id: str) -> ExperimentView | None:
        """Return an experiment by external identifier."""
        with self.database.session_scope() as session:
            row = session.scalar(
                select(ExperimentRow).where(ExperimentRow.experiment_id == experiment_id)
            )
            return _experiment_view(session, row) if row else None

    def history(self, limit: int = 50) -> list[ExperimentView]:
        """Return recent experiments, newest first, without modifying them."""
        with self.database.session_scope() as session:
            rows = session.scalars(
                select(ExperimentRow)
                .order_by(ExperimentRow.created_at.desc(), ExperimentRow.id.desc())
                .limit(limit)
            ).all()
            return [_experiment_view(session, row) for row in rows]

    def mark_running(self, experiment_id: str) -> None:
        """Move an experiment into the running state."""
        with self.database.session_scope() as session:
            row = self._require(session, experiment_id)
            if row.status in CLOSED_EXPERIMENT_STATUSES:
                raise ExperimentStateError(
                    f"experiment {experiment_id} is {row.status} and cannot be restarted"
                )
            row.status = "running"
            if row.started_at is None:
                row.started_at = utc_now()

    def mark_completed(self, experiment_id: str) -> None:
        """Close an experiment after all runs finish."""
        with self.database.session_scope() as session:
            row = self._require(session, experiment_id)
            row.status = "completed"
            row.completed_at = utc_now()

    def mark_failed(self, experiment_id: str, message: str) -> None:
        """Record a failed experiment without deleting any stored results."""
        with self.database.session_scope() as session:
            row = self._require(session, experiment_id)
            row.status = "failed"
            row.completed_at = utc_now()
            row.notes = f"{row.notes}\n{message}".strip() if row.notes else message

    def require_open(self, experiment_id: str) -> ExperimentView:
        """Return an experiment only when new runs are still allowed."""
        with self.database.session_scope() as session:
            row = self._require(session, experiment_id)
            if row.status in CLOSED_EXPERIMENT_STATUSES:
                raise ExperimentStateError(
                    f"experiment {experiment_id} is {row.status}; "
                    "new runs are not allowed"
                )
            return _experiment_view(session, row)

    def _require(self, session: Session, experiment_id: str) -> ExperimentRow:
        row = session.scalar(
            select(ExperimentRow).where(ExperimentRow.experiment_id == experiment_id)
        )
        if row is None:
            raise DatabaseError(f"experiment {experiment_id} does not exist")
        return row


class RunRepository(BaseRepository):
    """Create, inspect, and close per-model runs."""

    def create(
        self,
        *,
        experiment_external_id: str,
        model_id: int,
        profile: str,
        cold_start: bool = False,
        warmup_requests: int = 0,
        repetitions: int = 1,
    ) -> RunView:
        """Create one run, or return the existing run for the same identity."""
        with self.database.session_scope() as session:
            experiment = session.scalar(
                select(ExperimentRow).where(
                    ExperimentRow.experiment_id == experiment_external_id
                )
            )
            if experiment is None:
                raise DatabaseError(f"experiment {experiment_external_id} does not exist")
            if experiment.status in CLOSED_EXPERIMENT_STATUSES:
                raise ExperimentStateError(
                    f"experiment {experiment_external_id} is {experiment.status}; "
                    "new runs are not allowed"
                )
            row = session.scalar(
                select(RunRow).where(
                    RunRow.experiment_id == experiment.id,
                    RunRow.model_id == model_id,
                    RunRow.profile == profile,
                    RunRow.cold_start.is_(cold_start),
                )
            )
            if row is None:
                row = RunRow(
                    experiment_id=experiment.id,
                    model_id=model_id,
                    profile=profile,
                    status="running",
                    cold_start=cold_start,
                    warmup_requests=warmup_requests,
                    repetitions=repetitions,
                    started_at=utc_now(),
                )
                session.add(row)
                session.flush()
            model = session.get(ModelRow, row.model_id)
            return RunView(
                id=row.id,
                experiment_id=experiment_external_id,
                model_tag=model.tag if model else "unknown",
                profile=row.profile,
                status=row.status,
                cold_start=row.cold_start,
                warmup_requests=row.warmup_requests,
                repetitions=row.repetitions,
                started_at=row.started_at,
                completed_at=row.completed_at,
                error_message=row.error_message,
            )

    def mark_completed(self, run_id: int) -> None:
        """Close a run normally."""
        self._close(run_id, "completed", None)

    def mark_failed(self, run_id: int, message: str) -> None:
        """Close a run as failed and keep the reason."""
        self._close(run_id, "failed", message)

    def mark_cancelled(self, run_id: int) -> None:
        """Close a run that was cancelled by the operator."""
        self._close(run_id, "cancelled", None)

    def reopen(self, run_id: int) -> RunView:
        """Reopen a cancelled run so a resumable experiment can finish it."""
        with self.database.session_scope() as session:
            row = session.get(RunRow, run_id)
            if row is None:
                raise DatabaseError(f"run {run_id} does not exist")
            if row.status == "cancelled":
                row.status = "running"
                row.completed_at = None
                row.error_message = None
            experiment = session.get(ExperimentRow, row.experiment_id)
            model = session.get(ModelRow, row.model_id)
            return RunView(
                id=row.id,
                experiment_id=experiment.experiment_id if experiment else "unknown",
                model_tag=model.tag if model else "unknown",
                profile=row.profile,
                status=row.status,
                cold_start=row.cold_start,
                warmup_requests=row.warmup_requests,
                repetitions=row.repetitions,
                started_at=row.started_at,
                completed_at=row.completed_at,
                error_message=row.error_message,
            )

    def _close(self, run_id: int, status: str, message: str | None) -> None:
        with self.database.session_scope() as session:
            row = session.get(RunRow, run_id)
            if row is None:
                raise DatabaseError(f"run {run_id} does not exist")
            row.status = status
            row.completed_at = utc_now()
            row.error_message = message

    def get(self, run_id: int) -> RunView | None:
        """Return one run."""
        with self.database.session_scope() as session:
            row = session.get(RunRow, run_id)
            if row is None:
                return None
            model = session.get(ModelRow, row.model_id)
            experiment = session.get(ExperimentRow, row.experiment_id)
            return RunView(
                id=row.id,
                experiment_id=experiment.experiment_id if experiment else "unknown",
                model_tag=model.tag if model else "unknown",
                profile=row.profile,
                status=row.status,
                cold_start=row.cold_start,
                warmup_requests=row.warmup_requests,
                repetitions=row.repetitions,
                started_at=row.started_at,
                completed_at=row.completed_at,
                error_message=row.error_message,
            )


class ResponseRepository(BaseRepository):
    """Store responses with metrics and hardware in one transaction."""

    def save(self, record: ResponseRecord, run_id: int) -> int:
        """Insert a response once; identical duplicates are idempotent, conflicts raise."""
        with self.database.session_scope() as session:
            run = session.get(RunRow, run_id)
            if run is None:
                raise DatabaseError(f"run {run_id} does not exist")
            if run.status == "completed":
                raise ExperimentStateError(
                    f"run {run_id} is completed; responses are append-only"
                )
            question_id = record.item_id or f"unlabelled-{record.response_id[:8]}"
            existing = session.scalar(
                select(ResponseRow).where(
                    ResponseRow.run_id == run_id,
                    ResponseRow.question_id == question_id,
                    ResponseRow.repetition == record.repetition,
                    ResponseRow.is_warmup.is_(record.is_warmup),
                )
            )
            payload = record.model_dump(mode="json")
            if existing is not None:
                if existing.raw_json == payload:
                    return existing.id
                raise DatabaseDuplicateError(
                    f"response for {question_id} repetition {record.repetition} "
                    f"already exists with different content in run {run_id}"
                )
            row = ResponseRow(
                run_id=run_id,
                response_uuid=record.response_id,
                question_id=question_id,
                category=record.category,
                prompt=record.prompt,
                system_prompt=record.system_prompt,
                response_text=record.response_text,
                finish_reason=record.finish_reason,
                status=record.status.value,
                error_type=record.error_type,
                error_message=record.error_message,
                is_warmup=record.is_warmup,
                cold_start=record.cold_start,
                repetition=record.repetition,
                started_at=record.started_at,
                completed_at=record.completed_at,
                stream_chunk_count=record.stream_chunk_count,
                config_hash=record.config_hash,
                raw_json=payload,
                created_at=utc_now(),
            )
            session.add(row)
            session.flush()
            session.add(
                PerformanceMetricRow(
                    response_id=row.id,
                    ttft_seconds=record.timing.ttft_seconds,
                    total_latency_seconds=record.timing.total_latency_seconds,
                    load_duration_seconds=record.timing.load_duration_seconds,
                    prompt_eval_duration_seconds=record.timing.prompt_eval_duration_seconds,
                    eval_duration_seconds=record.timing.eval_duration_seconds,
                    prompt_eval_count=record.tokens.prompt_eval_count,
                    eval_count=record.tokens.eval_count,
                    output_tokens_per_second=record.tokens.output_tokens_per_second,
                    prompt_tokens_per_second=record.tokens.prompt_tokens_per_second,
                    options_json=record.generation_options.model_dump(mode="json"),
                )
            )
            if record.hardware is not None:
                session.add(_hardware_row(row.id, record))
            return row.id

    def existing_keys(self, run_id: int) -> set[tuple[str, int, bool]]:
        """Return stored (question, repetition, warm-up) keys for resume."""
        with self.database.session_scope() as session:
            rows = session.execute(
                select(
                    ResponseRow.question_id,
                    ResponseRow.repetition,
                    ResponseRow.is_warmup,
                ).where(ResponseRow.run_id == run_id)
            ).all()
            return {(row[0], row[1], bool(row[2])) for row in rows}

    def pending_evaluations(self, run_id: int) -> list[PendingEvaluation]:
        """Return measured responses that do not yet have an evaluation."""
        with self.database.session_scope() as session:
            rows = session.execute(
                select(
                    ResponseRow.id,
                    ResponseRow.question_id,
                    ResponseRow.response_text,
                    ResponseRow.status,
                    ResponseRow.repetition,
                )
                .outerjoin(EvaluationRow, EvaluationRow.response_id == ResponseRow.id)
                .where(
                    ResponseRow.run_id == run_id,
                    ResponseRow.is_warmup.is_(False),
                    EvaluationRow.id.is_(None),
                )
                .order_by(ResponseRow.question_id, ResponseRow.repetition)
            ).all()
            return [
                PendingEvaluation(
                    response_id=row[0],
                    question_id=row[1],
                    response_text=row[2],
                    status=row[3],
                    repetition=row[4],
                )
                for row in rows
            ]


def _hardware_row(response_id: int, record: ResponseRecord) -> HardwareMeasurementRow:
    summary = record.hardware
    if summary is None:  # pragma: no cover - guarded by the caller
        raise DatabaseError("hardware row requested without a hardware summary")
    payload = summary.model_dump(mode="json")
    inference = summary.inference
    return HardwareMeasurementRow(
        response_id=response_id,
        schema_version=summary.schema_version,
        sampling_interval_ms=summary.sampling_interval_ms,
        baseline_seconds=summary.baseline_seconds,
        sample_count=summary.sample_count,
        baseline_sample_count=summary.baseline.sample_count,
        inference_sample_count=inference.sample_count,
        cpu_avg_percent=inference.cpu_avg_percent,
        cpu_peak_percent=inference.cpu_peak_percent,
        ram_avg_used_bytes=inference.ram_avg_used_bytes,
        ram_peak_used_bytes=inference.ram_peak_used_bytes,
        gpu_utilization_avg_percent=inference.gpu_utilization_avg_percent,
        gpu_utilization_peak_percent=inference.gpu_utilization_peak_percent,
        vram_avg_used_bytes=inference.vram_avg_used_bytes,
        vram_peak_used_bytes=inference.vram_peak_used_bytes,
        gpu_temperature_avg_c=inference.gpu_temperature_avg_c,
        gpu_temperature_peak_c=inference.gpu_temperature_peak_c,
        ollama_rss_avg_bytes=inference.ollama_rss_avg_bytes,
        ollama_rss_peak_bytes=inference.ollama_rss_peak_bytes,
        ollama_model_size_vram_bytes=summary.ollama_model_size_vram_bytes,
        ollama_model_vram_ratio=summary.ollama_model_vram_ratio,
        gpu_available=summary.gpu_available,
        gpu_utilization_available=summary.gpu_utilization_available,
        gpu_memory_available=summary.gpu_memory_available,
        gpu_temperature_available=summary.gpu_temperature_available,
        ollama_process_available=summary.ollama_process_available,
        ollama_model_vram_available=summary.ollama_model_vram_available,
        gpu_name=summary.gpu_name,
        gpu_source=summary.gpu_source.value if summary.gpu_source else None,
        sampler_overhead_avg_ms=summary.sampler_overhead_avg_ms,
        sampler_overhead_peak_ms=summary.sampler_overhead_peak_ms,
        gpu_error=summary.gpu_error,
        sampler_error=summary.sampler_error,
        attribution_note=summary.attribution_note,
        summary_json=payload,
        samples_json=[sample.model_dump(mode="json") for sample in summary.samples],
    )


class EvaluationRepository(BaseRepository):
    """Store one evaluation per response without overwriting historical scores."""

    def save(self, result: EvaluationResult, response_id: int) -> int:
        """Insert an evaluation, returning the existing row when it is identical."""
        with self.database.session_scope() as session:
            existing = session.scalar(
                select(EvaluationRow).where(EvaluationRow.response_id == response_id)
            )
            judge = result.judge
            if existing is not None:
                if existing.status == result.status.value and (
                    existing.deterministic_normalized_score
                    == result.deterministic_normalized_score
                    and existing.judge_status
                    == (judge.status.value if judge else None)
                    and existing.composite_score == result.composite_score
                ):
                    return existing.id
                raise DatabaseDuplicateError(
                    f"response {response_id} already has a different evaluation"
                )
            row = EvaluationRow(
                response_id=response_id,
                question_id=result.question_id,
                category=result.category,
                evaluation_type=result.evaluation_type.value,
                evaluator=result.evaluator,
                evaluator_version=result.evaluator_version,
                status=result.status.value,
                deterministic_score=result.deterministic_score,
                deterministic_max_score=result.deterministic_max_score,
                deterministic_normalized_score=result.deterministic_normalized_score,
                passed=result.passed,
                judge_status=judge.status.value if judge else None,
                judge_model_tag=judge.config.tag if judge and judge.config else None,
                judge_model_digest=judge.config.digest if judge and judge.config else None,
                judge_same_family=judge.same_family if judge else None,
                judge_raw_score=judge.raw_score if judge else None,
                judge_normalized_score=judge.normalized_score if judge else None,
                judge_missing=result.judge_missing,
                composite_score=result.composite_score,
                checks_json=[check.model_dump(mode="json") for check in result.checks],
                judge_dimensions_json=(
                    {
                        name: dimension.model_dump(mode="json")
                        for name, dimension in judge.dimensions.items()
                    }
                    if judge and judge.dimensions
                    else None
                ),
                judge_config_json=(
                    judge.config.model_dump(mode="json") if judge and judge.config else None
                ),
                errors_json=[error.model_dump(mode="json") for error in result.errors],
                created_at=utc_now(),
            )
            session.add(row)
            session.flush()
            return row.id

    def count_for_experiment(self, experiment_external_id: str) -> int:
        """Return how many evaluations belong to an experiment."""
        with self.database.session_scope() as session:
            return (
                session.scalar(
                    select(func.count())
                    .select_from(EvaluationRow)
                    .join(ResponseRow, ResponseRow.id == EvaluationRow.response_id)
                    .join(RunRow, RunRow.id == ResponseRow.run_id)
                    .join(
                        ExperimentRow,
                        ExperimentRow.id == RunRow.experiment_id,
                    )
                    .where(ExperimentRow.experiment_id == experiment_external_id)
                )
                or 0
            )


class RecommendationRepository(BaseRepository):
    """Store workload recommendations produced by later analytics."""

    def save_many(
        self,
        experiment_external_id: str,
        recommendations: list[RecommendationInput],
    ) -> int:
        """Store recommendations, skipping identical rows and rejecting conflicts."""
        stored = 0
        with self.database.session_scope() as session:
            experiment = session.scalar(
                select(ExperimentRow).where(
                    ExperimentRow.experiment_id == experiment_external_id
                )
            )
            if experiment is None:
                raise DatabaseError(f"experiment {experiment_external_id} does not exist")
            for item in recommendations:
                model = session.scalar(select(ModelRow).where(ModelRow.tag == item.model_tag))
                if model is None:
                    raise DatabaseError(f"model {item.model_tag} is not registered")
                existing = session.scalar(
                    select(RecommendationRow).where(
                        RecommendationRow.experiment_id == experiment.id,
                        RecommendationRow.workload == item.workload,
                        RecommendationRow.rank == item.rank,
                    )
                )
                if existing is not None:
                    if existing.model_id == model.id and existing.score == item.score:
                        continue
                    raise DatabaseDuplicateError(
                        f"recommendation {item.workload} rank {item.rank} "
                        "already exists with different content"
                    )
                session.add(
                    RecommendationRow(
                        experiment_id=experiment.id,
                        model_id=model.id,
                        workload=item.workload,
                        rank=item.rank,
                        score=item.score,
                        confidence=item.confidence,
                        weights_json=item.weights,
                        details_json=item.details or None,
                        rationale=item.rationale,
                        created_at=utc_now(),
                    )
                )
                stored += 1
        return stored

    def list_for_experiment(self, experiment_external_id: str) -> list[dict[str, Any]]:
        """Return stored recommendations joined with model tags."""
        with self.database.session_scope() as session:
            rows = session.execute(
                select(
                    RecommendationRow.workload,
                    RecommendationRow.rank,
                    ModelRow.tag,
                    RecommendationRow.score,
                    RecommendationRow.confidence,
                    RecommendationRow.weights_json,
                    RecommendationRow.details_json,
                    RecommendationRow.rationale,
                )
                .join(ModelRow, ModelRow.id == RecommendationRow.model_id)
                .join(
                    ExperimentRow,
                    ExperimentRow.id == RecommendationRow.experiment_id,
                )
                .where(ExperimentRow.experiment_id == experiment_external_id)
                .order_by(RecommendationRow.workload, RecommendationRow.rank)
            ).all()
            return [
                {
                    "workload": row[0],
                    "rank": row[1],
                    "model_tag": row[2],
                    "score": row[3],
                    "confidence": row[4],
                    "weights": row[5],
                    "details": row[6],
                    "rationale": row[7],
                }
                for row in rows
            ]


class ErrorRepository(BaseRepository):
    """Record failures that are not tied to one stored response."""

    def record(
        self,
        *,
        stage: str,
        error_type: str,
        message: str,
        experiment_external_id: str | None = None,
        run_id: int | None = None,
    ) -> int:
        """Append a structured error."""
        with self.database.session_scope() as session:
            experiment_id = None
            if experiment_external_id is not None:
                experiment = session.scalar(
                    select(ExperimentRow).where(
                        ExperimentRow.experiment_id == experiment_external_id
                    )
                )
                if experiment is None:
                    raise DatabaseError(f"experiment {experiment_external_id} does not exist")
                experiment_id = experiment.id
            row = ErrorRow(
                experiment_id=experiment_id,
                run_id=run_id,
                stage=stage,
                error_type=error_type,
                message=message,
                occurred_at=utc_now(),
            )
            session.add(row)
            session.flush()
            return row.id

    def list_for_experiment(self, experiment_external_id: str) -> list[dict[str, Any]]:
        """Return recorded errors for an experiment."""
        with self.database.session_scope() as session:
            rows = session.execute(
                select(
                    ErrorRow.stage,
                    ErrorRow.error_type,
                    ErrorRow.message,
                    ErrorRow.occurred_at,
                )
                .join(ExperimentRow, ExperimentRow.id == ErrorRow.experiment_id)
                .where(ExperimentRow.experiment_id == experiment_external_id)
                .order_by(ErrorRow.occurred_at)
            ).all()
            return [
                {
                    "stage": row[0],
                    "error_type": row[1],
                    "message": row[2],
                    "occurred_at": row[3],
                }
                for row in rows
            ]


class ReadRepository(BaseRepository):
    """Read-only queries used by the CLI, dashboard, and reports."""

    def experiment_history(self, limit: int = 50) -> list[ExperimentView]:
        """Return recent experiments."""
        return ExperimentRepository(self.database).history(limit)

    def experiment_detail(self, experiment_external_id: str) -> ExperimentDetailView:
        """Return the experiment snapshot, dataset identity, runs, and used models."""
        with self.database.session_scope() as session:
            experiment = session.scalar(
                select(ExperimentRow).where(
                    ExperimentRow.experiment_id == experiment_external_id
                )
            )
            if experiment is None:
                raise DatabaseError(f"experiment {experiment_external_id} does not exist")
            dataset = session.get(DatasetRow, experiment.dataset_id)
            run_rows = session.scalars(
                select(RunRow)
                .join(ModelRow, ModelRow.id == RunRow.model_id)
                .where(RunRow.experiment_id == experiment.id)
                .order_by(ModelRow.tag, RunRow.cold_start)
            ).all()
            runs = [
                RunDetailView(
                    id=run.id,
                    model_tag=run.model.tag,
                    model_name=run.model.name,
                    role=run.model.role,
                    profile=run.profile,
                    status=run.status,
                    cold_start=run.cold_start,
                    warmup_requests=run.warmup_requests,
                    repetitions=run.repetitions,
                    started_at=run.started_at,
                    completed_at=run.completed_at,
                    error_message=run.error_message,
                )
                for run in run_rows
            ]
            models: dict[str, RegisteredModel] = {}
            for run in run_rows:
                models.setdefault(run.model.tag, _registered_model(run.model))
            return ExperimentDetailView(
                experiment=_experiment_view(session, experiment),
                config=dict(experiment.config_json or {}),
                host=dict(experiment.host_json or {}),
                software=dict(experiment.software_json or {}),
                notes=experiment.notes,
                dataset_schema_version=dataset.schema_version if dataset else None,
                dataset_item_count=dataset.item_count if dataset else 0,
                dataset_source_path=dataset.source_path if dataset else None,
                dataset_random_seed=dataset.random_seed if dataset else None,
                runs=runs,
                models=[models[tag] for tag in sorted(models)],
            )

    def model_results(self, experiment_external_id: str) -> list[ModelResultView]:
        """Aggregate measured performance and quality per model run."""
        success = case((ResponseRow.status == "success", 1), else_=0)
        with self.database.session_scope() as session:
            experiment = session.scalar(
                select(ExperimentRow).where(
                    ExperimentRow.experiment_id == experiment_external_id
                )
            )
            if experiment is None:
                raise DatabaseError(f"experiment {experiment_external_id} does not exist")
            rows = session.execute(
                select(
                    RunRow.id,
                    ModelRow.tag,
                    ModelRow.name,
                    RunRow.profile,
                    RunRow.status,
                    RunRow.cold_start,
                    func.count(ResponseRow.id),
                    func.sum(success),
                    func.avg(PerformanceMetricRow.total_latency_seconds),
                    func.avg(PerformanceMetricRow.ttft_seconds),
                    func.avg(PerformanceMetricRow.output_tokens_per_second),
                    func.avg(EvaluationRow.deterministic_normalized_score),
                    func.avg(EvaluationRow.judge_normalized_score),
                    func.avg(EvaluationRow.composite_score),
                )
                .join(ModelRow, ModelRow.id == RunRow.model_id)
                .outerjoin(
                    ResponseRow,
                    and_(
                        ResponseRow.run_id == RunRow.id,
                        ResponseRow.is_warmup.is_(False),
                    ),
                )
                .outerjoin(
                    PerformanceMetricRow,
                    PerformanceMetricRow.response_id == ResponseRow.id,
                )
                .outerjoin(EvaluationRow, EvaluationRow.response_id == ResponseRow.id)
                .where(RunRow.experiment_id == experiment.id)
                .group_by(RunRow.id)
                .order_by(ModelRow.tag)
            ).all()
            results = []
            for row in rows:
                measured = int(row[6] or 0)
                successes = int(row[7] or 0)
                results.append(
                    ModelResultView(
                        run_id=row[0],
                        model_tag=row[1],
                        model_name=row[2],
                        profile=row[3],
                        status=row[4],
                        cold_start=bool(row[5]),
                        measured_responses=measured,
                        successful_responses=successes,
                        failed_responses=measured - successes,
                        mean_latency_seconds=row[8],
                        mean_ttft_seconds=row[9],
                        mean_output_tokens_per_second=row[10],
                        mean_deterministic_score=row[11],
                        mean_judge_score=row[12],
                        mean_composite_score=row[13],
                    )
                )
            return results

    def analytics_rows(
        self,
        experiment_external_id: str,
        *,
        run_profile: str | None = None,
        include_cold_start: bool = False,
        include_smoke: bool = False,
    ) -> list[AnalyticsRow]:
        """Return measured responses with timing, evaluation, and hardware columns.

        Smoke and cold-start runs are excluded unless requested. An explicit
        ``run_profile`` filter takes precedence over the smoke exclusion.
        """
        with self.database.session_scope() as session:
            experiment = session.scalar(
                select(ExperimentRow).where(
                    ExperimentRow.experiment_id == experiment_external_id
                )
            )
            if experiment is None:
                raise DatabaseError(f"experiment {experiment_external_id} does not exist")
            query = (
                select(
                    ModelRow.tag.label("model_tag"),
                    ModelRow.name.label("model_name"),
                    RunRow.profile.label("run_profile"),
                    RunRow.cold_start.label("cold_start"),
                    ResponseRow.question_id.label("question_id"),
                    ResponseRow.category.label("category"),
                    ResponseRow.status.label("response_status"),
                    ResponseRow.repetition.label("repetition"),
                    ResponseRow.response_text.label("response_text"),
                    PerformanceMetricRow.total_latency_seconds.label("latency_seconds"),
                    PerformanceMetricRow.ttft_seconds.label("ttft_seconds"),
                    PerformanceMetricRow.output_tokens_per_second.label(
                        "output_tokens_per_second"
                    ),
                    PerformanceMetricRow.prompt_tokens_per_second.label(
                        "prompt_tokens_per_second"
                    ),
                    EvaluationRow.status.label("evaluation_status"),
                    EvaluationRow.evaluation_type.label("evaluation_type"),
                    EvaluationRow.deterministic_normalized_score.label(
                        "deterministic_score"
                    ),
                    EvaluationRow.judge_normalized_score.label("judge_score"),
                    EvaluationRow.composite_score.label("composite_score"),
                    EvaluationRow.judge_missing.label("judge_missing"),
                    HardwareMeasurementRow.id.label("hardware_id"),
                    HardwareMeasurementRow.cpu_avg_percent.label("cpu_avg_percent"),
                    HardwareMeasurementRow.ram_avg_used_bytes.label(
                        "ram_avg_used_bytes"
                    ),
                    HardwareMeasurementRow.gpu_utilization_avg_percent.label(
                        "gpu_utilization_avg_percent"
                    ),
                    HardwareMeasurementRow.vram_avg_used_bytes.label(
                        "vram_avg_used_bytes"
                    ),
                    HardwareMeasurementRow.ollama_model_size_vram_bytes.label(
                        "ollama_model_vram_bytes"
                    ),
                    HardwareMeasurementRow.ollama_model_vram_available.label(
                        "ollama_model_vram_available"
                    ),
                    HardwareMeasurementRow.gpu_available.label("gpu_available"),
                    HardwareMeasurementRow.gpu_memory_available.label(
                        "gpu_memory_available"
                    ),
                    HardwareMeasurementRow.gpu_name.label("gpu_name"),
                )
                .join(RunRow, RunRow.id == ResponseRow.run_id)
                .join(ModelRow, ModelRow.id == RunRow.model_id)
                .outerjoin(
                    PerformanceMetricRow,
                    PerformanceMetricRow.response_id == ResponseRow.id,
                )
                .outerjoin(EvaluationRow, EvaluationRow.response_id == ResponseRow.id)
                .outerjoin(
                    HardwareMeasurementRow,
                    HardwareMeasurementRow.response_id == ResponseRow.id,
                )
                .where(
                    RunRow.experiment_id == experiment.id,
                    ResponseRow.is_warmup.is_(False),
                )
            )
            if run_profile is not None:
                query = query.where(RunRow.profile == run_profile)
            elif not include_smoke:
                query = query.where(RunRow.profile != "smoke")
            if not include_cold_start:
                query = query.where(RunRow.cold_start.is_(False))
            rows = session.execute(
                query.order_by(
                    ModelRow.tag,
                    ResponseRow.question_id,
                    ResponseRow.repetition,
                )
            ).all()
            return [
                AnalyticsRow(
                    model_tag=row.model_tag,
                    model_name=row.model_name,
                    run_profile=row.run_profile,
                    cold_start=bool(row.cold_start),
                    question_id=row.question_id,
                    category=row.category,
                    response_status=row.response_status,
                    repetition=row.repetition,
                    response_text=row.response_text,
                    latency_seconds=row.latency_seconds,
                    ttft_seconds=row.ttft_seconds,
                    output_tokens_per_second=row.output_tokens_per_second,
                    prompt_tokens_per_second=row.prompt_tokens_per_second,
                    evaluation_status=row.evaluation_status,
                    evaluation_type=row.evaluation_type,
                    deterministic_score=row.deterministic_score,
                    judge_score=row.judge_score,
                    composite_score=row.composite_score,
                    judge_missing=row.judge_missing,
                    hardware_present=row.hardware_id is not None,
                    cpu_avg_percent=row.cpu_avg_percent,
                    ram_avg_used_bytes=row.ram_avg_used_bytes,
                    gpu_utilization_avg_percent=row.gpu_utilization_avg_percent,
                    vram_avg_used_bytes=row.vram_avg_used_bytes,
                    ollama_model_vram_bytes=row.ollama_model_vram_bytes,
                    ollama_model_vram_available=bool(
                        row.ollama_model_vram_available
                    ),
                    gpu_available=bool(row.gpu_available),
                    gpu_memory_available=bool(row.gpu_memory_available),
                    gpu_name=row.gpu_name,
                )
                for row in rows
            ]

    def response_inspection(
        self,
        experiment_external_id: str,
        *,
        model_tag: str | None = None,
        question_id: str | None = None,
        category: str | None = None,
        include_cold_start: bool = True,
        include_smoke: bool = True,
    ) -> list[ResponseInspectionView]:
        """Return stored responses with prompt, timing, evaluation, and hardware detail."""
        with self.database.session_scope() as session:
            experiment = session.scalar(
                select(ExperimentRow).where(
                    ExperimentRow.experiment_id == experiment_external_id
                )
            )
            if experiment is None:
                raise DatabaseError(f"experiment {experiment_external_id} does not exist")
            query = (
                select(
                    ResponseRow.question_id,
                    ModelRow.tag,
                    ModelRow.name,
                    RunRow.profile,
                    RunRow.cold_start,
                    ResponseRow.repetition,
                    ResponseRow.category,
                    ResponseRow.prompt,
                    ResponseRow.system_prompt,
                    ResponseRow.response_text,
                    ResponseRow.status,
                    ResponseRow.finish_reason,
                    ResponseRow.error_type,
                    ResponseRow.error_message,
                    ResponseRow.started_at,
                    ResponseRow.completed_at,
                    ResponseRow.stream_chunk_count,
                    PerformanceMetricRow.total_latency_seconds,
                    PerformanceMetricRow.ttft_seconds,
                    PerformanceMetricRow.load_duration_seconds,
                    PerformanceMetricRow.prompt_eval_duration_seconds,
                    PerformanceMetricRow.eval_duration_seconds,
                    PerformanceMetricRow.prompt_eval_count,
                    PerformanceMetricRow.eval_count,
                    PerformanceMetricRow.output_tokens_per_second,
                    PerformanceMetricRow.prompt_tokens_per_second,
                    EvaluationRow.status,
                    EvaluationRow.evaluation_type,
                    EvaluationRow.deterministic_score,
                    EvaluationRow.deterministic_max_score,
                    EvaluationRow.deterministic_normalized_score,
                    EvaluationRow.passed,
                    EvaluationRow.judge_status,
                    EvaluationRow.judge_model_tag,
                    EvaluationRow.judge_raw_score,
                    EvaluationRow.judge_normalized_score,
                    EvaluationRow.judge_missing,
                    EvaluationRow.composite_score,
                    EvaluationRow.checks_json,
                    EvaluationRow.judge_dimensions_json,
                    EvaluationRow.errors_json,
                    HardwareMeasurementRow.summary_json,
                )
                .join(RunRow, RunRow.id == ResponseRow.run_id)
                .join(ModelRow, ModelRow.id == RunRow.model_id)
                .outerjoin(
                    PerformanceMetricRow,
                    PerformanceMetricRow.response_id == ResponseRow.id,
                )
                .outerjoin(EvaluationRow, EvaluationRow.response_id == ResponseRow.id)
                .outerjoin(
                    HardwareMeasurementRow,
                    HardwareMeasurementRow.response_id == ResponseRow.id,
                )
                .where(
                    RunRow.experiment_id == experiment.id,
                    ResponseRow.is_warmup.is_(False),
                )
            )
            if not include_smoke:
                query = query.where(RunRow.profile != "smoke")
            if not include_cold_start:
                query = query.where(RunRow.cold_start.is_(False))
            if model_tag is not None:
                query = query.where(ModelRow.tag == model_tag)
            if question_id is not None:
                query = query.where(ResponseRow.question_id == question_id)
            if category is not None:
                query = query.where(ResponseRow.category == category)
            rows = session.execute(
                query.order_by(
                    ResponseRow.question_id,
                    ModelRow.tag,
                    ResponseRow.repetition,
                )
            ).all()
            return [
                ResponseInspectionView(
                    question_id=row[0],
                    model_tag=row[1],
                    model_name=row[2],
                    run_profile=row[3],
                    cold_start=bool(row[4]),
                    repetition=row[5],
                    category=row[6],
                    prompt=row[7],
                    system_prompt=row[8],
                    response_text=row[9],
                    response_status=row[10],
                    finish_reason=row[11],
                    error_type=row[12],
                    error_message=row[13],
                    started_at=row[14],
                    completed_at=row[15],
                    stream_chunk_count=row[16] or 0,
                    latency_seconds=row[17],
                    ttft_seconds=row[18],
                    load_duration_seconds=row[19],
                    prompt_eval_duration_seconds=row[20],
                    eval_duration_seconds=row[21],
                    prompt_eval_count=row[22],
                    eval_count=row[23],
                    output_tokens_per_second=row[24],
                    prompt_tokens_per_second=row[25],
                    evaluation_status=row[26],
                    evaluation_type=row[27],
                    deterministic_score=row[28],
                    deterministic_max_score=row[29],
                    deterministic_normalized_score=row[30],
                    passed=row[31],
                    judge_status=row[32],
                    judge_model_tag=row[33],
                    judge_raw_score=row[34],
                    judge_normalized_score=row[35],
                    judge_missing=row[36],
                    composite_score=row[37],
                    checks=list(row[38] or []),
                    judge_dimensions=row[39],
                    evaluation_errors=list(row[40] or []),
                    hardware_summary=row[41],
                )
                for row in rows
            ]

    def evaluation_breakdown(
        self,
        experiment_external_id: str,
        *,
        run_profile: str | None = None,
        include_cold_start: bool = False,
        include_smoke: bool = False,
    ) -> list[EvaluationBreakdownView]:
        """Return evaluation pass counts and score means by model and method."""
        passed = case((EvaluationRow.status == "passed", 1), else_=0)
        with self.database.session_scope() as session:
            experiment = session.scalar(
                select(ExperimentRow).where(
                    ExperimentRow.experiment_id == experiment_external_id
                )
            )
            if experiment is None:
                raise DatabaseError(f"experiment {experiment_external_id} does not exist")
            query = (
                select(
                    ModelRow.tag,
                    EvaluationRow.evaluation_type,
                    func.count(EvaluationRow.id),
                    func.coalesce(func.sum(passed), 0),
                    func.avg(EvaluationRow.deterministic_normalized_score),
                    func.avg(EvaluationRow.judge_normalized_score),
                    func.avg(EvaluationRow.composite_score),
                )
                .join(ResponseRow, ResponseRow.id == EvaluationRow.response_id)
                .join(RunRow, RunRow.id == ResponseRow.run_id)
                .join(ModelRow, ModelRow.id == RunRow.model_id)
                .where(
                    RunRow.experiment_id == experiment.id,
                    ResponseRow.is_warmup.is_(False),
                )
            )
            if run_profile is not None:
                query = query.where(RunRow.profile == run_profile)
            elif not include_smoke:
                query = query.where(RunRow.profile != "smoke")
            if not include_cold_start:
                query = query.where(RunRow.cold_start.is_(False))
            rows = session.execute(
                query.group_by(ModelRow.tag, EvaluationRow.evaluation_type).order_by(
                    ModelRow.tag,
                    EvaluationRow.evaluation_type,
                )
            ).all()
            return [
                EvaluationBreakdownView(
                    model_tag=row[0],
                    evaluation_type=row[1],
                    evaluated_responses=int(row[2] or 0),
                    passed_responses=int(row[3] or 0),
                    deterministic_score=row[4],
                    judge_score=row[5],
                    composite_score=row[6],
                )
                for row in rows
            ]

    def hardware_usage(
        self,
        experiment_external_id: str,
        *,
        run_profile: str | None = None,
        include_cold_start: bool = False,
        include_smoke: bool = False,
    ) -> list[HardwareUsageRow]:
        """Return per-response average and peak hardware measurements."""
        with self.database.session_scope() as session:
            experiment = session.scalar(
                select(ExperimentRow).where(
                    ExperimentRow.experiment_id == experiment_external_id
                )
            )
            if experiment is None:
                raise DatabaseError(f"experiment {experiment_external_id} does not exist")
            query = (
                select(
                    ModelRow.tag,
                    ResponseRow.question_id,
                    ResponseRow.repetition,
                    RunRow.cold_start,
                    HardwareMeasurementRow.sample_count,
                    HardwareMeasurementRow.baseline_sample_count,
                    HardwareMeasurementRow.inference_sample_count,
                    HardwareMeasurementRow.cpu_avg_percent,
                    HardwareMeasurementRow.cpu_peak_percent,
                    HardwareMeasurementRow.ram_avg_used_bytes,
                    HardwareMeasurementRow.ram_peak_used_bytes,
                    HardwareMeasurementRow.gpu_utilization_avg_percent,
                    HardwareMeasurementRow.gpu_utilization_peak_percent,
                    HardwareMeasurementRow.vram_avg_used_bytes,
                    HardwareMeasurementRow.vram_peak_used_bytes,
                    HardwareMeasurementRow.gpu_temperature_avg_c,
                    HardwareMeasurementRow.gpu_temperature_peak_c,
                    HardwareMeasurementRow.ollama_rss_avg_bytes,
                    HardwareMeasurementRow.ollama_rss_peak_bytes,
                    HardwareMeasurementRow.ollama_model_size_vram_bytes,
                    HardwareMeasurementRow.gpu_name,
                    HardwareMeasurementRow.gpu_available,
                    HardwareMeasurementRow.gpu_utilization_available,
                    HardwareMeasurementRow.gpu_memory_available,
                    HardwareMeasurementRow.gpu_temperature_available,
                    HardwareMeasurementRow.ollama_model_vram_available,
                    HardwareMeasurementRow.gpu_error,
                    HardwareMeasurementRow.sampler_error,
                    HardwareMeasurementRow.attribution_note,
                )
                .join(ResponseRow, ResponseRow.id == HardwareMeasurementRow.response_id)
                .join(RunRow, RunRow.id == ResponseRow.run_id)
                .join(ModelRow, ModelRow.id == RunRow.model_id)
                .where(
                    RunRow.experiment_id == experiment.id,
                    ResponseRow.is_warmup.is_(False),
                )
            )
            if run_profile is not None:
                query = query.where(RunRow.profile == run_profile)
            elif not include_smoke:
                query = query.where(RunRow.profile != "smoke")
            if not include_cold_start:
                query = query.where(RunRow.cold_start.is_(False))
            rows = session.execute(
                query.order_by(
                    ModelRow.tag,
                    ResponseRow.question_id,
                    ResponseRow.repetition,
                )
            ).all()
            return [
                HardwareUsageRow(
                    model_tag=row[0],
                    question_id=row[1],
                    repetition=row[2],
                    cold_start=bool(row[3]),
                    sample_count=row[4] or 0,
                    baseline_sample_count=row[5] or 0,
                    inference_sample_count=row[6] or 0,
                    cpu_avg_percent=row[7],
                    cpu_peak_percent=row[8],
                    ram_avg_used_bytes=row[9],
                    ram_peak_used_bytes=row[10],
                    gpu_utilization_avg_percent=row[11],
                    gpu_utilization_peak_percent=row[12],
                    vram_avg_used_bytes=row[13],
                    vram_peak_used_bytes=row[14],
                    gpu_temperature_avg_c=row[15],
                    gpu_temperature_peak_c=row[16],
                    ollama_rss_avg_bytes=row[17],
                    ollama_rss_peak_bytes=row[18],
                    ollama_model_size_vram_bytes=row[19],
                    gpu_name=row[20],
                    gpu_available=bool(row[21]),
                    gpu_utilization_available=bool(row[22]),
                    gpu_memory_available=bool(row[23]),
                    gpu_temperature_available=bool(row[24]),
                    model_vram_available=bool(row[25]),
                    gpu_error=row[26],
                    sampler_error=row[27],
                    attribution_note=row[28],
                )
                for row in rows
            ]

    def question_results(
        self,
        experiment_external_id: str,
        question_id: str | None = None,
    ) -> list[QuestionResultView]:
        """Return question-level responses ordered by question and model."""
        with self.database.session_scope() as session:
            experiment = session.scalar(
                select(ExperimentRow).where(
                    ExperimentRow.experiment_id == experiment_external_id
                )
            )
            if experiment is None:
                raise DatabaseError(f"experiment {experiment_external_id} does not exist")
            query = (
                select(
                    ResponseRow.question_id,
                    ModelRow.tag,
                    ResponseRow.status,
                    ResponseRow.repetition,
                    ResponseRow.response_text,
                    PerformanceMetricRow.total_latency_seconds,
                    PerformanceMetricRow.ttft_seconds,
                    EvaluationRow.deterministic_normalized_score,
                    EvaluationRow.judge_normalized_score,
                    EvaluationRow.composite_score,
                )
                .join(RunRow, RunRow.id == ResponseRow.run_id)
                .join(ModelRow, ModelRow.id == RunRow.model_id)
                .outerjoin(
                    PerformanceMetricRow,
                    PerformanceMetricRow.response_id == ResponseRow.id,
                )
                .outerjoin(EvaluationRow, EvaluationRow.response_id == ResponseRow.id)
                .where(
                    RunRow.experiment_id == experiment.id,
                    ResponseRow.is_warmup.is_(False),
                )
                .order_by(ResponseRow.question_id, ModelRow.tag, ResponseRow.repetition)
            )
            if question_id is not None:
                query = query.where(ResponseRow.question_id == question_id)
            rows = session.execute(query).all()
            return [
                QuestionResultView(
                    question_id=row[0],
                    model_tag=row[1],
                    status=row[2],
                    repetition=row[3],
                    response_text=row[4],
                    total_latency_seconds=row[5],
                    ttft_seconds=row[6],
                    deterministic_normalized_score=row[7],
                    judge_normalized_score=row[8],
                    composite_score=row[9],
                )
                for row in rows
            ]


class Repositories:
    """Container that gives callers one database handle per repository area."""

    def __init__(self, database: Database) -> None:
        self.database = database
        self.models = ModelRepository(database)
        self.datasets = DatasetRepository(database)
        self.experiments = ExperimentRepository(database)
        self.runs = RunRepository(database)
        self.responses = ResponseRepository(database)
        self.evaluations = EvaluationRepository(database)
        self.recommendations = RecommendationRepository(database)
        self.errors = ErrorRepository(database)
        self.reads = ReadRepository(database)
