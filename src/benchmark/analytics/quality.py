"""
File: quality.py

Description:
Aggregation of stored deterministic, judge, and composite evaluation scores.

Why this file exists:
A single "quality" number hides category weaknesses and judge coverage. This module
keeps deterministic and subjective scores separate, gives each measured category
equal influence in the primary score, and reports what was not measured.

Responsibilities:
- Aggregate overall deterministic, judge, and composite scores.
- Aggregate each of the seven benchmark categories independently.
- Report quality and judge coverage, item-weighted sensitivity, and failure-as-zero.
- Preserve missing scores rather than imputing judge or evaluation values.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence

from benchmark.analytics.models import CategoryQuality, QualityAnalysis
from benchmark.database.repositories import AnalyticsRow
from benchmark.dataset.models import Category


def _mean(values: Sequence[float | None]) -> float | None:
    clean = [value for value in values if value is not None]
    if not clean:
        return None
    return float(sum(clean) / len(clean))


def _category_order() -> dict[str, int]:
    return {category.value: index for index, category in enumerate(Category)}


def _has_score(row: AnalyticsRow) -> bool:
    return any(
        score is not None
        for score in (
            row.deterministic_score,
            row.judge_score,
            row.composite_score,
        )
    )


def _category_quality(
    category: str,
    category_rows: Sequence[AnalyticsRow],
    *,
    category_weight: float,
) -> CategoryQuality:
    measured = len(category_rows)
    evaluated = [row for row in category_rows if _has_score(row)]
    judged = [row for row in evaluated if row.judge_score is not None]
    return CategoryQuality(
        category=category,
        measured_responses=measured,
        evaluated_responses=len(evaluated),
        judge_evaluated_responses=len(judged),
        quality_coverage=len(evaluated) / measured if measured else 0.0,
        judge_coverage=len(judged) / len(evaluated) if evaluated else 0.0,
        category_weight=category_weight,
        deterministic_score=_mean([row.deterministic_score for row in evaluated]),
        judge_score=_mean([row.judge_score for row in evaluated]),
        composite_score=_mean([row.composite_score for row in evaluated]),
    )


def quality_analysis(
    rows: Sequence[AnalyticsRow],
    *,
    category_weights: Mapping[str, float] | None = None,
) -> QualityAnalysis:
    """Aggregate stored evaluation scores without imputing missing judge values.

    The primary overall score is the mean of measured category scores. This prevents
    categories with more prompts from dominating the result. The item-weighted mean
    is retained as a sensitivity value, while failed responses are treated as zero
    only in the explicitly named failure-as-zero score.
    """
    measured = len(rows)
    evaluated = [row for row in rows if _has_score(row)]
    judged = [row for row in evaluated if row.judge_score is not None]

    grouped: dict[str, list[AnalyticsRow]] = defaultdict(list)
    for row in rows:
        grouped[row.category or "unknown"].append(row)
    order = _category_order()
    weights = dict(category_weights or {})
    categories = [
        _category_quality(
            category,
            grouped[category],
            category_weight=weights.get(category, 0.0),
        )
        for category in sorted(grouped, key=lambda name: (order.get(name, 999), name))
    ]

    category_composites = [
        category.composite_score
        for category in categories
        if category.composite_score is not None
    ]
    category_deterministic = [
        category.deterministic_score
        for category in categories
        if category.deterministic_score is not None
    ]
    category_judge = [
        category.judge_score
        for category in categories
        if category.judge_score is not None
    ]
    item_weighted = _mean([row.composite_score for row in evaluated])

    failed_responses = sum(
        1 for row in rows if row.response_status not in {"success", "cancelled"}
    )
    scored_successes = sum(
        1
        for row in evaluated
        if row.response_status == "success" and row.composite_score is not None
    )
    failure_denominator = scored_successes + failed_responses
    failure_numerator = sum(
        row.composite_score or 0.0
        for row in evaluated
        if row.response_status == "success" and row.composite_score is not None
    )
    failure_as_zero = (
        failure_numerator / failure_denominator if failure_denominator else None
    )

    limitations: list[str] = []
    if not evaluated:
        limitations.append("no measured response has a stored evaluation score")
    if measured and len(evaluated) < measured:
        limitations.append(
            f"evaluations cover {len(evaluated)} of {measured} measured responses"
        )
    if evaluated and len(judged) < len(evaluated):
        limitations.append(
            f"judge scores cover {len(judged)} of {len(evaluated)} evaluated responses"
        )
    if not category_composites:
        limitations.append("no category has a composite quality score")
    if evaluated and item_weighted is None:
        limitations.append(
            "composite scores are unavailable; deterministic scores are reported"
        )
    if failed_responses:
        limitations.append(
            f"{failed_responses} failed responses are omitted from the primary "
            "score and counted as zero only in the failure-as-zero sensitivity"
        )

    return QualityAnalysis(
        measured_responses=measured,
        evaluated_responses=len(evaluated),
        judge_evaluated_responses=len(judged),
        quality_coverage=len(evaluated) / measured if measured else 0.0,
        judge_coverage=len(judged) / len(evaluated) if evaluated else 0.0,
        deterministic_score=_mean(category_deterministic),
        judge_score=_mean(category_judge),
        composite_score=_mean(category_composites),
        item_weighted_composite_score=item_weighted,
        failure_as_zero_composite_score=failure_as_zero,
        categories=categories,
        limitations=limitations,
    )
