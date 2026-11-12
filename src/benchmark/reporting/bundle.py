"""
File: bundle.py

Description:
Collect every stored artifact for one experiment into a single report bundle.

Why this file exists:
CSV, JSON, HTML, and PDF exports must agree on the same evidence, filters, and
analytics. Building the bundle once guarantees that consistency and keeps the
exporters free of database queries.

Responsibilities:
- Refuse to report an experiment that is not completed unless explicitly allowed.
- Compute analytics and recommendations through the Phase 08 engine.
- Read measured responses, evaluation breakdowns, hardware rows, and errors.
- Record the filters and limitations that qualify the report.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from benchmark.analytics import AnalyticsEngine
from benchmark.analytics.models import ExperimentAnalytics, RecommendationSet
from benchmark.core.config import AppSettings
from benchmark.core.exceptions import InsufficientDataError, ReportError
from benchmark.database.db import Database
from benchmark.database.repositories import (
    AnalyticsRow,
    EvaluationBreakdownView,
    ExperimentDetailView,
    ExperimentView,
    HardwareUsageRow,
    Repositories,
    ResponseInspectionView,
)

REPORT_SCHEMA_VERSION = "1.0.0"
COMPLETED_STATUS = "completed"


@dataclass
class ReportBundle:
    """All stored evidence and derived analytics for one experiment report."""

    detail: ExperimentDetailView
    analysis: ExperimentAnalytics
    recommendations: list[RecommendationSet]
    measured_rows: list[AnalyticsRow]
    inspections: list[ResponseInspectionView]
    hardware_rows: list[HardwareUsageRow]
    evaluation_breakdown: list[EvaluationBreakdownView]
    errors: list[dict[str, Any]]
    generated_at: datetime
    profile: str
    preliminary: bool
    run_profile: str | None
    include_cold_start: bool
    include_smoke: bool
    limitations: list[str] = field(default_factory=list)
    database_path: str = ""

    @property
    def experiment(self) -> ExperimentView:
        """Return the stored experiment summary."""
        return self.detail.experiment

    @property
    def experiment_id(self) -> str:
        """Return the experiment identifier."""
        return self.detail.experiment.experiment_id


def build_bundle(
    database: Database,
    experiment_id: str,
    *,
    settings: AppSettings | None = None,
    profile: str | None = None,
    allow_incomplete: bool = False,
    run_profile: str | None = None,
    include_cold_start: bool = False,
    include_smoke: bool = False,
) -> ReportBundle:
    """Collect a report bundle for one experiment.

    A non-completed experiment is refused unless ``allow_incomplete`` is set; in
    that case the bundle is flagged preliminary and every export carries the flag.
    """
    repos = Repositories(database)
    view = repos.experiments.get(experiment_id)
    if view is None:
        raise ReportError(f"experiment {experiment_id} does not exist")
    preliminary = view.status != COMPLETED_STATUS
    if preliminary and not allow_incomplete:
        raise ReportError(
            f"experiment {experiment_id} is {view.status}; only completed experiments "
            "can be reported. Pass allow_incomplete to export explicitly preliminary data."
        )
    detail = repos.reads.experiment_detail(experiment_id)
    engine = AnalyticsEngine(
        database,
        settings=settings.benchmark if settings is not None else None,
    )
    try:
        analysis = engine.analyze(
            experiment_id,
            profile=profile,
            run_profile=run_profile,
            include_cold_start=include_cold_start,
            include_smoke=include_smoke,
        )
    except InsufficientDataError as exc:
        raise ReportError(
            f"experiment {experiment_id} has no measurable rows for this report: {exc}"
        ) from exc
    recommendations = engine.recommend(
        experiment_id,
        persist=False,
        run_profile=run_profile,
        include_cold_start=include_cold_start,
        include_smoke=include_smoke,
    )
    limitations = list(analysis.limitations)
    if preliminary:
        limitations.insert(
            0,
            f"experiment status is {view.status}; this report is preliminary and "
            "must not be presented as a final result",
        )
    if not include_smoke:
        limitations.append("smoke runs are excluded from the reported metrics")
    if not include_cold_start:
        limitations.append("cold-start runs are excluded from the reported metrics")
    if not recommendations:
        limitations.append("no workload recommendation could be computed")
    return ReportBundle(
        detail=detail,
        analysis=analysis,
        recommendations=recommendations,
        measured_rows=repos.reads.analytics_rows(
            experiment_id,
            run_profile=run_profile,
            include_cold_start=include_cold_start,
            include_smoke=include_smoke,
        ),
        inspections=repos.reads.response_inspection(
            experiment_id,
            include_cold_start=include_cold_start,
            include_smoke=include_smoke,
        ),
        hardware_rows=repos.reads.hardware_usage(
            experiment_id,
            run_profile=run_profile,
            include_cold_start=include_cold_start,
            include_smoke=include_smoke,
        ),
        evaluation_breakdown=repos.reads.evaluation_breakdown(
            experiment_id,
            run_profile=run_profile,
            include_cold_start=include_cold_start,
            include_smoke=include_smoke,
        ),
        errors=repos.errors.list_for_experiment(experiment_id),
        generated_at=datetime.now(timezone.utc),
        profile=analysis.profile,
        preliminary=preliminary,
        run_profile=run_profile,
        include_cold_start=include_cold_start,
        include_smoke=include_smoke,
        limitations=limitations,
        database_path=str(database.path),
    )


__all__ = ["COMPLETED_STATUS", "REPORT_SCHEMA_VERSION", "ReportBundle", "build_bundle"]
