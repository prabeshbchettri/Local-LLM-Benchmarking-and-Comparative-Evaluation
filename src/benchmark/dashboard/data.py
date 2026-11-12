"""
File: data.py

Description:
Cached, read-only data access for every dashboard page.

Why this file exists:
Dashboard figures must come from the persisted experiment rows and the analytics
engine, never from literals in the UI. This module is the single seam between
Streamlit and the database, so caching keys, filters, and error handling stay
consistent across pages.

Responsibilities:
- Build the configuration and database context once per process.
- Read experiment history, registered models, and dataset items.
- Compute analytics, recommendations, and historical comparisons through the engine.
- Expose response-level, evaluation-level, and hardware-level read models.
- Include the database file version in cache keys so new rows appear promptly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import streamlit as st

from benchmark.analytics import AnalyticsEngine
from benchmark.analytics.models import (
    ExperimentAnalytics,
    HistoricalComparison,
    RecommendationSet,
)
from benchmark.core.config import AppSettings, load_settings
from benchmark.core.exceptions import BenchmarkError
from benchmark.database.db import Database
from benchmark.database.repositories import (
    AnalyticsRow,
    EvaluationBreakdownView,
    ExperimentView,
    HardwareUsageRow,
    RegisteredModel,
    Repositories,
    ResponseInspectionView,
)
from benchmark.dataset.loader import load_dataset
from benchmark.dataset.models import BenchmarkDataset, DatasetItem

BYTES_PER_MIB = 1024 * 1024


@dataclass
class DashboardContext:
    """Everything a dashboard page needs to read stored benchmark evidence."""

    settings: AppSettings
    database: Database
    dataset: BenchmarkDataset | None
    dataset_error: str | None


@st.cache_resource(show_spinner=False)
def get_context(config_dir: str | None, project_root: str | None) -> DashboardContext:
    """Load settings, open the database, and load the validated dataset."""
    settings = load_settings(
        config_dir=Path(config_dir) if config_dir else None,
        project_root=Path(project_root) if project_root else None,
    )
    database = Database.from_settings(settings)
    database.initialize()
    dataset: BenchmarkDataset | None = None
    dataset_error: str | None = None
    try:
        dataset = load_dataset(settings.benchmark.dataset_path)
    except Exception as exc:  # surfaced as a dashboard warning, never hidden
        dataset_error = f"{type(exc).__name__}: {exc}"
    return DashboardContext(
        settings=settings,
        database=database,
        dataset=dataset,
        dataset_error=dataset_error,
    )


def _db_version(database: Database) -> str:
    """Return a cache key that changes whenever the SQLite files change."""
    parts: list[str] = []
    for suffix in ("", "-wal"):
        path = Path(f"{database.path}{suffix}")
        try:
            stat = path.stat()
        except OSError:
            continue
        parts.append(f"{stat.st_mtime_ns}:{stat.st_size}")
    return "|".join(parts)


def db_version(context: DashboardContext) -> str:
    """Return the database cache key for a dashboard context."""
    return _db_version(context.database)


def _open_database(path: str) -> Database:
    database = Database(Path(path))
    database.initialize()
    return database


def _profiles_config(config_dir: str) -> AppSettings:
    return load_settings(config_dir=Path(config_dir))


@st.cache_data(ttl=30, show_spinner=False)
def _history(db_path: str, version: str, limit: int) -> list[ExperimentView]:
    database = _open_database(db_path)
    try:
        return Repositories(database).experiments.history(limit)
    finally:
        database.dispose()


@st.cache_data(ttl=30, show_spinner=False)
def _registered_models(db_path: str, version: str) -> list[RegisteredModel]:
    database = _open_database(db_path)
    try:
        return Repositories(database).models.list()
    finally:
        database.dispose()


@st.cache_data(ttl=30, show_spinner=False)
def _analysis(
    db_path: str,
    config_dir: str,
    version: str,
    experiment_id: str,
    profile: str | None,
    run_profile: str | None,
    include_cold_start: bool,
    include_smoke: bool,
) -> ExperimentAnalytics:
    database = _open_database(db_path)
    try:
        settings = _profiles_config(config_dir)
        engine = AnalyticsEngine(database, settings=settings.benchmark)
        return engine.analyze(
            experiment_id,
            profile=profile,
            run_profile=run_profile,
            include_cold_start=include_cold_start,
            include_smoke=include_smoke,
        )
    finally:
        database.dispose()


@st.cache_data(ttl=30, show_spinner=False)
def _recommendations(
    db_path: str,
    config_dir: str,
    version: str,
    experiment_id: str,
    workloads: tuple[str, ...] | None,
    run_profile: str | None,
    include_cold_start: bool,
    include_smoke: bool,
) -> list[RecommendationSet]:
    database = _open_database(db_path)
    try:
        settings = _profiles_config(config_dir)
        engine = AnalyticsEngine(database, settings=settings.benchmark)
        return engine.recommend(
            experiment_id,
            workloads=list(workloads) if workloads else None,
            persist=False,
            run_profile=run_profile,
            include_cold_start=include_cold_start,
            include_smoke=include_smoke,
        )
    finally:
        database.dispose()


@st.cache_data(ttl=30, show_spinner=False)
def _comparison(
    db_path: str,
    config_dir: str,
    version: str,
    baseline_id: str,
    candidate_ids: tuple[str, ...],
    profile: str | None,
    run_profile: str | None,
    include_cold_start: bool,
    include_smoke: bool,
) -> HistoricalComparison:
    database = _open_database(db_path)
    try:
        settings = _profiles_config(config_dir)
        engine = AnalyticsEngine(database, settings=settings.benchmark)
        return engine.compare(
            baseline_id,
            list(candidate_ids),
            profile=profile,
            run_profile=run_profile,
            include_cold_start=include_cold_start,
            include_smoke=include_smoke,
        )
    finally:
        database.dispose()


@st.cache_data(ttl=30, show_spinner=False)
def _measured_rows(
    db_path: str,
    version: str,
    experiment_id: str,
    run_profile: str | None,
    include_cold_start: bool,
    include_smoke: bool,
) -> list[AnalyticsRow]:
    database = _open_database(db_path)
    try:
        return Repositories(database).reads.analytics_rows(
            experiment_id,
            run_profile=run_profile,
            include_cold_start=include_cold_start,
            include_smoke=include_smoke,
        )
    finally:
        database.dispose()


@st.cache_data(ttl=30, show_spinner=False)
def _inspections(
    db_path: str,
    version: str,
    experiment_id: str,
    model_tag: str | None,
    question_id: str | None,
    category: str | None,
) -> list[ResponseInspectionView]:
    database = _open_database(db_path)
    try:
        return Repositories(database).reads.response_inspection(
            experiment_id,
            model_tag=model_tag,
            question_id=question_id,
            category=category,
        )
    finally:
        database.dispose()


@st.cache_data(ttl=30, show_spinner=False)
def _evaluation_breakdown(
    db_path: str,
    version: str,
    experiment_id: str,
    run_profile: str | None,
    include_cold_start: bool,
    include_smoke: bool,
) -> list[EvaluationBreakdownView]:
    database = _open_database(db_path)
    try:
        return Repositories(database).reads.evaluation_breakdown(
            experiment_id,
            run_profile=run_profile,
            include_cold_start=include_cold_start,
            include_smoke=include_smoke,
        )
    finally:
        database.dispose()


@st.cache_data(ttl=30, show_spinner=False)
def _hardware_rows(
    db_path: str,
    version: str,
    experiment_id: str,
    run_profile: str | None,
    include_cold_start: bool,
    include_smoke: bool,
) -> list[HardwareUsageRow]:
    database = _open_database(db_path)
    try:
        return Repositories(database).reads.hardware_usage(
            experiment_id,
            run_profile=run_profile,
            include_cold_start=include_cold_start,
            include_smoke=include_smoke,
        )
    finally:
        database.dispose()


def history(context: DashboardContext, limit: int = 100) -> list[ExperimentView]:
    """Return recent experiments, newest first."""
    return _history(
        str(context.database.path),
        _db_version(context.database),
        limit,
    )


def completed_experiments(context: DashboardContext, limit: int = 100) -> list[ExperimentView]:
    """Return completed experiments only."""
    return [view for view in history(context, limit) if view.status == "completed"]


def latest_experiment(context: DashboardContext) -> ExperimentView | None:
    """Return the newest experiment, regardless of status."""
    views = history(context, 1)
    return views[0] if views else None


def experiment_by_name(
    context: DashboardContext, name: str
) -> ExperimentView | None:
    """Find an experiment by its operator-visible name."""
    return next(
        (view for view in history(context, 200) if view.name == name),
        None,
    )


def registered_models(context: DashboardContext) -> list[RegisteredModel]:
    """Return every model stored in the database."""
    return _registered_models(
        str(context.database.path),
        _db_version(context.database),
    )


def analysis(
    context: DashboardContext,
    experiment_id: str,
    *,
    profile: str | None = None,
    run_profile: str | None = None,
    include_cold_start: bool = False,
    include_smoke: bool = False,
) -> ExperimentAnalytics:
    """Compute profile-specific analytics through the Phase 08 engine."""
    return _analysis(
        str(context.database.path),
        str(context.settings.config_dir),
        _db_version(context.database),
        experiment_id,
        profile,
        run_profile,
        include_cold_start,
        include_smoke,
    )


def recommendations(
    context: DashboardContext,
    experiment_id: str,
    *,
    workloads: Sequence[str] | None = None,
    run_profile: str | None = None,
    include_cold_start: bool = False,
    include_smoke: bool = False,
) -> list[RecommendationSet]:
    """Return workload recommendations without persisting them."""
    return _recommendations(
        str(context.database.path),
        str(context.settings.config_dir),
        _db_version(context.database),
        experiment_id,
        tuple(workloads) if workloads else None,
        run_profile,
        include_cold_start,
        include_smoke,
    )


def comparison(
    context: DashboardContext,
    baseline_id: str,
    candidate_ids: Sequence[str],
    *,
    profile: str | None = None,
    run_profile: str | None = None,
    include_cold_start: bool = False,
    include_smoke: bool = False,
) -> HistoricalComparison:
    """Compare stored experiments through the Phase 08 engine."""
    return _comparison(
        str(context.database.path),
        str(context.settings.config_dir),
        _db_version(context.database),
        baseline_id,
        tuple(candidate_ids),
        profile,
        run_profile,
        include_cold_start,
        include_smoke,
    )


def measured_rows(
    context: DashboardContext,
    experiment_id: str,
    *,
    run_profile: str | None = None,
    include_cold_start: bool = False,
    include_smoke: bool = False,
) -> list[AnalyticsRow]:
    """Return stored response rows used for distributions and timelines."""
    return _measured_rows(
        str(context.database.path),
        _db_version(context.database),
        experiment_id,
        run_profile,
        include_cold_start,
        include_smoke,
    )


def inspections(
    context: DashboardContext,
    experiment_id: str,
    *,
    model_tag: str | None = None,
    question_id: str | None = None,
    category: str | None = None,
) -> list[ResponseInspectionView]:
    """Return full response detail for inspection."""
    return _inspections(
        str(context.database.path),
        _db_version(context.database),
        experiment_id,
        model_tag,
        question_id,
        category,
    )


def evaluation_breakdown(
    context: DashboardContext,
    experiment_id: str,
    *,
    run_profile: str | None = None,
    include_cold_start: bool = False,
    include_smoke: bool = False,
) -> list[EvaluationBreakdownView]:
    """Return pass counts and means by evaluation method."""
    return _evaluation_breakdown(
        str(context.database.path),
        _db_version(context.database),
        experiment_id,
        run_profile,
        include_cold_start,
        include_smoke,
    )


def hardware_rows(
    context: DashboardContext,
    experiment_id: str,
    *,
    run_profile: str | None = None,
    include_cold_start: bool = False,
    include_smoke: bool = False,
) -> list[HardwareUsageRow]:
    """Return stored average and peak hardware measurements per response."""
    return _hardware_rows(
        str(context.database.path),
        _db_version(context.database),
        experiment_id,
        run_profile,
        include_cold_start,
        include_smoke,
    )


def dataset_items(context: DashboardContext) -> dict[str, DatasetItem]:
    """Return dataset items by identifier, or an empty mapping."""
    if context.dataset is None:
        return {}
    return {item.id: item for item in context.dataset.items}


def configured_candidates(context: DashboardContext) -> list[str]:
    """Return the configured candidate model tags."""
    return [spec.tag for spec in context.settings.models.candidates]


def profile_names(context: DashboardContext) -> list[str]:
    """Return configured workload profile names in configuration order."""
    return list(context.settings.benchmark.workload_profiles)


def _mean(values: Sequence[float | None]) -> float | None:
    clean = [value for value in values if value is not None]
    return float(sum(clean) / len(clean)) if clean else None


def _max(values: Sequence[float | None]) -> float | None:
    clean = [value for value in values if value is not None]
    return float(max(clean)) if clean else None


def _bytes_to_mib(value: float | None) -> float | None:
    return None if value is None else value / BYTES_PER_MIB


def hardware_summary_rows(
    rows: Sequence[HardwareUsageRow],
) -> list[dict[str, object]]:
    """Aggregate per-response hardware usage into average and peak model rows."""
    grouped: dict[str, list[HardwareUsageRow]] = {}
    for row in rows:
        grouped.setdefault(row.model_tag, []).append(row)
    summaries: list[dict[str, object]] = []
    for tag in sorted(grouped):
        items = grouped[tag]
        notes = list(
            dict.fromkeys(
                note
                for item in items
                for note in (item.gpu_error, item.sampler_error)
                if note
            )
        )
        summaries.append(
            {
                "model_tag": tag,
                "responses": len(items),
                "samples": sum(item.inference_sample_count for item in items),
                "cpu_avg_percent": _mean([item.cpu_avg_percent for item in items]),
                "cpu_peak_percent": _max([item.cpu_peak_percent for item in items]),
                "ram_avg_mib": _bytes_to_mib(
                    _mean([item.ram_avg_used_bytes for item in items])
                ),
                "ram_peak_mib": _bytes_to_mib(
                    _max([item.ram_peak_used_bytes for item in items])
                ),
                "gpu_avg_percent": _mean(
                    [item.gpu_utilization_avg_percent for item in items]
                ),
                "gpu_peak_percent": _max(
                    [item.gpu_utilization_peak_percent for item in items]
                ),
                "vram_avg_mib": _bytes_to_mib(
                    _mean([item.vram_avg_used_bytes for item in items])
                ),
                "vram_peak_mib": _bytes_to_mib(
                    _max([item.vram_peak_used_bytes for item in items])
                ),
                "temperature_avg_c": _mean(
                    [item.gpu_temperature_avg_c for item in items]
                ),
                "temperature_peak_c": _max(
                    [item.gpu_temperature_peak_c for item in items]
                ),
                "gpu_name": next(
                    (item.gpu_name for item in items if item.gpu_name),
                    None,
                ),
                "notes": notes,
            }
        )
    return summaries


__all__ = [
    "BenchmarkError",
    "DashboardContext",
    "analysis",
    "comparison",
    "completed_experiments",
    "configured_candidates",
    "dataset_items",
    "db_version",
    "evaluation_breakdown",
    "experiment_by_name",
    "get_context",
    "hardware_rows",
    "hardware_summary_rows",
    "history",
    "inspections",
    "latest_experiment",
    "measured_rows",
    "profile_names",
    "recommendations",
    "registered_models",
]
