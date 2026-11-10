"""
File: orchestrator.py

Description:
Persisted experiment orchestration across models, prompts, profiles, and evaluations.

Why this file exists:
Running a benchmark used to produce JSONL files only. Experiments now need durable
rows for every request, metric, hardware sample, and evaluation so they survive a
restart and can be inspected later.

Responsibilities:
- Register the dataset and model roster before executing an experiment.
- Create one run per model, execute warm-ups and measured repetitions, and persist
  every response as soon as it completes.
- Resume an open experiment without repeating stored requests.
- Evaluate stored responses after the model is unloaded and persist the results.
- Mark runs and experiments failed instead of losing partial evidence.
"""

from __future__ import annotations

import os
import platform
import sys
import time
from datetime import datetime, timezone
from importlib import metadata as package_metadata
from typing import Callable, Sequence

import psutil

from benchmark.core.config import AppSettings, ModelSpec
from benchmark.core.exceptions import BenchmarkError, DatabaseError, ExperimentStateError
from benchmark.database.db import Database
from benchmark.database.repositories import Repositories
from benchmark.dataset.models import BenchmarkDataset, DatasetItem
from benchmark.evaluation.engine import EvaluationEngine, EvaluationRequest
from benchmark.evaluation.judge import Judge
from benchmark.execution.engine import InferenceEngine
from benchmark.execution.metadata import (
    ModelRole,
    build_missing_metadata,
)
from benchmark.execution.registry import ModelManager
from benchmark.models.base import BaseSchema
from benchmark.models.inference import (
    InferenceProfile,
    ResponseRecord,
    ResponseStatus,
)

CLOSED_STATUSES = frozenset({"completed", "failed", "cancelled"})


def _utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


def _new_experiment_id() -> str:
    """Return a sortable, collision-resistant experiment identifier."""
    return f"{_utc_now():%Y%m%dT%H%M%SZ}-{os.urandom(3).hex()}"


def _package_version(name: str) -> str:
    """Return an installed package version or a readable fallback."""
    try:
        return package_metadata.version(name)
    except package_metadata.PackageNotFoundError:
        return "unknown"


def _host_metadata() -> dict[str, object]:
    """Collect stable host facts for the experiment record."""
    memory = psutil.virtual_memory()
    return {
        "hostname": platform.node() or "unknown",
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "python": sys.version.split()[0],
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "memory_total_bytes": memory.total,
    }


def _software_metadata() -> dict[str, str]:
    """Collect versions that affect how results were produced."""
    return {
        "benchmark": _package_version("local-llm-benchmark"),
        "sqlalchemy": _package_version("SQLAlchemy"),
        "pydantic": _package_version("pydantic"),
        "httpx": _package_version("httpx"),
        "psutil": _package_version("psutil"),
    }


class ExperimentReport(BaseSchema):
    """Summary returned after an experiment finishes or resumes."""

    experiment_id: str
    status: str
    profile: str
    dataset_hash: str
    database_path: str
    models: list[str]
    runs: int
    responses: int
    evaluations: int
    successful_responses: int
    failed_responses: int
    resumed: bool


class ExperimentOrchestrator:
    """Execute and persist one benchmark experiment sequentially across models."""

    def __init__(
        self,
        settings: AppSettings,
        database: Database,
        dataset: BenchmarkDataset,
        engine: InferenceEngine,
        *,
        manager: ModelManager | None = None,
        judge: Judge | None = None,
        progress: Callable[[ResponseRecord], None] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.settings = settings
        self.database = database
        self.dataset = dataset
        self.engine = engine
        self.manager = manager
        self.judge = judge
        self.repos = Repositories(database)
        self.evaluation = EvaluationEngine(judge=judge, settings=settings.evaluation)
        self._progress = progress
        self._sleep = sleep

    def run(
        self,
        models: Sequence[str] | None = None,
        *,
        profile: InferenceProfile | str = InferenceProfile.QUALITY,
        item_ids: Sequence[str] | None = None,
        limit: int | None = None,
        repetitions: int | None = None,
        warmup_requests: int | None = None,
        cold_start: bool = False,
        cooldown_seconds: float | None = None,
        name: str | None = None,
        resume_experiment: str | None = None,
    ) -> ExperimentReport:
        """Create or resume an experiment and execute one run per model."""
        profile_value = InferenceProfile(profile)
        selected = (
            list(models)
            if models
            else [spec.tag for spec in self.settings.models.candidates]
        )
        self._validate_models(selected)
        items = self._select_items(item_ids, limit)
        repeats = (
            repetitions
            if repetitions is not None
            else self._default_repetitions(profile_value)
        )
        if repeats < 1:
            raise BenchmarkError("repetitions must be at least 1")
        if warmup_requests is not None and warmup_requests < 0:
            raise BenchmarkError("warm-up requests must not be negative")
        warmup = 0 if cold_start else (
            self.settings.benchmark.warmup_requests
            if warmup_requests is None
            else warmup_requests
        )
        cooldown = (
            self.settings.benchmark.cooldown_seconds_between_models
            if cooldown_seconds is None
            else cooldown_seconds
        )
        if cooldown < 0:
            raise BenchmarkError("cooldown must not be negative")

        resumed = resume_experiment is not None
        if resume_experiment is not None:
            view = self.repos.experiments.get(resume_experiment)
            if view is None:
                raise DatabaseError(f"experiment {resume_experiment} does not exist")
            if view.dataset_hash != self.dataset.content_sha256:
                raise ExperimentStateError(
                    f"experiment {resume_experiment} used dataset "
                    f"{view.dataset_hash[:12]}, not the loaded dataset"
                )
            if view.status in CLOSED_STATUSES:
                raise ExperimentStateError(
                    f"experiment {resume_experiment} is already {view.status}"
                )
            experiment_id = view.experiment_id
        else:
            dataset_id = self.repos.datasets.register(
                self.dataset,
                self.settings.benchmark.dataset_path,
            )
            self._register_models()
            experiment_id = _new_experiment_id()
            self.repos.experiments.create(
                experiment_id=experiment_id,
                name=name,
                dataset_id=dataset_id,
                config_hash=self.settings.config_hash,
                config=self.settings.model_dump(mode="json"),
                host=_host_metadata(),
                software=_software_metadata(),
            )

        self.repos.experiments.mark_running(experiment_id)
        try:
            for index, tag in enumerate(selected):
                self._run_model(
                    experiment_id,
                    tag,
                    items,
                    profile_value,
                    repeats,
                    warmup,
                    cold_start,
                )
                if self._cancelled():
                    break
                if cooldown > 0 and index < len(selected) - 1:
                    self._sleep(cooldown)
            self.repos.experiments.mark_completed(experiment_id)
        except Exception as exc:
            self.repos.errors.record(
                stage="orchestrator",
                error_type=type(exc).__name__,
                message=str(exc),
                experiment_external_id=experiment_id,
            )
            self.repos.experiments.mark_failed(
                experiment_id,
                f"{type(exc).__name__}: {exc}",
            )
            raise

        return self._report(experiment_id, profile_value, selected, resumed)

    def _validate_models(self, models: Sequence[str]) -> None:
        configured = {spec.tag for spec in self.settings.models.candidates}
        unknown = [tag for tag in models if tag not in configured]
        if unknown:
            raise BenchmarkError(
                f"models are not in the configured roster: {', '.join(unknown)}"
            )
        if len(set(models)) != len(models):
            raise BenchmarkError("selected models must be unique")
        if not models:
            raise BenchmarkError("at least one model must be selected")

    def _default_repetitions(self, profile: InferenceProfile) -> int:
        benchmark = self.settings.benchmark
        if profile is InferenceProfile.PERFORMANCE:
            return benchmark.performance_repeats
        if profile is InferenceProfile.CONSISTENCY:
            return benchmark.consistency_repeats
        if profile is InferenceProfile.COLD_START:
            return benchmark.cold_start_repeats
        return 1

    def _select_items(
        self,
        item_ids: Sequence[str] | None,
        limit: int | None,
    ) -> list[DatasetItem]:
        ordered = self.dataset.ordered_items(self.dataset.random_seed)
        if item_ids:
            by_id = {item.id: item for item in ordered}
            missing = [item_id for item_id in item_ids if item_id not in by_id]
            if missing:
                raise BenchmarkError(
                    f"unknown dataset items: {', '.join(sorted(missing))}"
                )
            items = [by_id[item_id] for item_id in item_ids]
        else:
            items = ordered
        if limit is not None:
            if limit < 1:
                raise BenchmarkError("limit must be at least 1")
            items = items[:limit]
        if not items:
            raise BenchmarkError("no dataset items selected")
        return items

    def _register_models(self) -> None:
        """Register candidates and the judge before the experiment starts."""
        for spec in self.settings.models.candidates:
            self._register_spec(spec, ModelRole.CANDIDATE)
        if self.judge is not None:
            self._register_spec(self.settings.models.judge, ModelRole.JUDGE)

    def _register_spec(self, spec: ModelSpec, role: ModelRole) -> None:
        if self.manager is not None:
            try:
                inspection = self.manager.inspect(spec.tag, require_installed=False)
            except BenchmarkError:
                inspection = None
            if inspection is not None:
                self.repos.models.register(inspection.metadata)
                return
        self.repos.models.register(
            build_missing_metadata(spec, "unknown", role)
        )

    def _cancelled(self) -> bool:
        return bool(getattr(self.engine, "interrupted", False))

    def _request(
        self,
        experiment_id: str,
        tag: str,
        item: DatasetItem,
        profile: InferenceProfile,
        repetition: int,
        *,
        is_warmup: bool,
        cold_start: bool,
    ) -> ResponseRecord:
        return self.engine.run_request(
            tag,
            item.prompt,
            experiment_id=experiment_id,
            item_id=item.id,
            category=item.category.value,
            repetition=repetition,
            profile=profile,
            system_instruction=item.system_instruction,
            max_tokens=item.maximum_output_tokens,
            is_warmup=is_warmup,
            cold_start=cold_start,
        )

    def _notify(self, record: ResponseRecord) -> None:
        if self._progress is not None:
            self._progress(record)

    def _run_model(
        self,
        experiment_id: str,
        tag: str,
        items: list[DatasetItem],
        profile: InferenceProfile,
        repetitions: int,
        warmup: int,
        cold_start: bool,
    ) -> None:
        model = self.repos.models.get_by_tag(tag)
        if model is None:
            raise DatabaseError(f"model {tag} is not registered")
        run = self.repos.runs.create(
            experiment_external_id=experiment_id,
            model_id=model.id,
            profile=profile.value,
            cold_start=cold_start,
            warmup_requests=warmup,
            repetitions=repetitions,
        )
        if run.status in CLOSED_STATUSES:
            return
        existing = self.repos.responses.existing_keys(run.id)
        try:
            if self.manager is not None and not cold_start:
                self.manager.load(tag)
            for index in range(warmup):
                item = items[index % len(items)]
                if (item.id, 1, True) in existing:
                    continue
                record = self._request(
                    experiment_id,
                    tag,
                    item,
                    profile,
                    1,
                    is_warmup=True,
                    cold_start=False,
                )
                self.repos.responses.save(record, run.id)
                self._notify(record)
            for item in items:
                for repetition in range(1, repetitions + 1):
                    if self._cancelled():
                        self.repos.runs.mark_cancelled(run.id)
                        return
                    if (item.id, repetition, False) in existing:
                        continue
                    if cold_start and self.manager is not None:
                        self.manager.unload(tag)
                    record = self._request(
                        experiment_id,
                        tag,
                        item,
                        profile,
                        repetition,
                        is_warmup=False,
                        cold_start=cold_start,
                    )
                    self.repos.responses.save(record, run.id)
                    self._notify(record)
        except Exception as exc:
            self.repos.runs.mark_failed(run.id, f"{type(exc).__name__}: {exc}")
            raise
        finally:
            if self.manager is not None:
                try:
                    self.manager.unload(tag)
                except BenchmarkError:
                    pass
        self._evaluate_run(experiment_id, run.id, tag, model.family, items)
        self.repos.runs.mark_completed(run.id)

    def _evaluate_run(
        self,
        experiment_id: str,
        run_id: int,
        tag: str,
        family: str | None,
        items: list[DatasetItem],
    ) -> None:
        by_id = {item.id: item for item in items}
        for pending in self.repos.responses.pending_evaluations(run_id):
            item = by_id.get(pending.question_id)
            if item is None:
                continue
            try:
                result = self.evaluation.evaluate(
                    EvaluationRequest(
                        item=item,
                        response_text=pending.response_text,
                        model_id=tag,
                        candidate_family=family,
                        response_status=ResponseStatus(pending.status),
                    )
                )
                self.repos.evaluations.save(result, pending.response_id)
            except Exception as exc:
                self.repos.errors.record(
                    stage="evaluation",
                    error_type=type(exc).__name__,
                    message=f"{pending.question_id}: {exc}",
                    experiment_external_id=experiment_id,
                    run_id=run_id,
                )

    def _report(
        self,
        experiment_id: str,
        profile: InferenceProfile,
        models: Sequence[str],
        resumed: bool,
    ) -> ExperimentReport:
        view = self.repos.experiments.get(experiment_id)
        if view is None:
            raise DatabaseError(f"experiment {experiment_id} disappeared")
        results = self.repos.reads.model_results(experiment_id)
        return ExperimentReport(
            experiment_id=experiment_id,
            status=view.status,
            profile=profile.value,
            dataset_hash=self.dataset.content_sha256,
            database_path=str(self.database.path),
            models=list(models),
            runs=view.run_count,
            responses=view.response_count,
            evaluations=view.evaluation_count,
            successful_responses=sum(item.successful_responses for item in results),
            failed_responses=sum(item.failed_responses for item in results),
            resumed=resumed,
        )
