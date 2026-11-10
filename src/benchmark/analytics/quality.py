"""
File: quality.py

Description:
Aggregation of stored deterministic, judge, and composite evaluation scores.

Why this file exists:
A single "quality" number hides category weaknesses and judge coverage. This module
keeps deterministic and subjective scores separate and reports what was not measured.

Responsibilities:
- Aggregate overall deterministic, judge, and composite scores.
- Aggregate each of the seven benchmark categories independently.
- Report judge coverage and explicit limitations.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

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


def quality_analysis(rows: Sequence[AnalyticsRow]) -> QualityAnalysis:
    """Aggregate stored evaluation scores without imputing missing judge values."""
    evaluated = [row for row in rows if row.evaluation_status is not None]
    judged = [row for row in evaluated if row.judge_score is not None]
    deterministic = _mean([row.deterministic_score for row in evaluated])
    judge = _mean([row.judge_score for row in evaluated])
    composite = _mean([row.composite_score for row in evaluated])
    limitations: list[str] = []
    if not evaluated:
        limitations.append("no measured response has a stored evaluation")
    if evaluated and len(judged) < len(evaluated):
        limitations.append(
            f"judge scores cover {len(judged)} of {len(evaluated)} evaluated responses"
        )
    if composite is None and deterministic is not None:
        limitations.append(
            "composite scores are unavailable; deterministic scores are reported"
        )

    grouped: dict[str, list[AnalyticsRow]] = defaultdict(list)
    for row in evaluated:
        grouped[row.category or "unknown"].append(row)
    order = _category_order()
    categories: list[CategoryQuality] = []
    for category in sorted(grouped, key=lambda name: (order.get(name, 999), name)):
        category_rows = grouped[category]
        category_judged = [
            row for row in category_rows if row.judge_score is not None
        ]
        categories.append(
            CategoryQuality(
                category=category,
                evaluated_responses=len(category_rows),
                judge_evaluated_responses=len(category_judged),
                judge_coverage=len(category_judged) / len(category_rows),
                deterministic_score=_mean(
                    [row.deterministic_score for row in category_rows]
                ),
                judge_score=_mean([row.judge_score for row in category_rows]),
                composite_score=_mean(
                    [row.composite_score for row in category_rows]
                ),
            )
        )

    return QualityAnalysis(
        evaluated_responses=len(evaluated),
        judge_evaluated_responses=len(judged),
        judge_coverage=len(judged) / len(evaluated) if evaluated else 0.0,
        deterministic_score=deterministic,
        judge_score=judge,
        composite_score=composite if composite is not None else deterministic,
        categories=categories,
        limitations=limitations,
    )
