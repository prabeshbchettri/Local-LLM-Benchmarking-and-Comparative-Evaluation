"""
File: test_analytics_quality.py

Description:
Unit tests for category-weighted quality aggregation and coverage reporting.

Responsibilities:
- Verify equal-category and item-weighted scores differ as documented.
- Verify deterministic and judge scores remain separate.
- Verify failed responses affect only the named sensitivity score.
"""

from __future__ import annotations

import pytest

from benchmark.analytics.quality import quality_analysis

pytestmark = pytest.mark.unit


def test_quality_uses_equal_categories_and_reports_failure_sensitivity(
    make_row,
) -> None:
    rows = [
        make_row(
            question_id="gk-1",
            category="general_knowledge",
            deterministic=90.0,
            judge=80.0,
            composite=90.0,
        ),
        make_row(
            question_id="gk-2",
            category="general_knowledge",
            repetition=2,
            deterministic=70.0,
            judge=None,
            composite=70.0,
            judge_missing=True,
        ),
        make_row(
            question_id="gk-3",
            category="general_knowledge",
            status="error",
            evaluation_status=None,
            deterministic=None,
            judge=None,
            composite=None,
            judge_missing=True,
        ),
        make_row(
            question_id="code-1",
            category="coding",
            deterministic=50.0,
            judge=50.0,
            composite=50.0,
        ),
    ]

    quality = quality_analysis(rows, category_weights={"coding": 1.0})

    assert quality.measured_responses == 4
    assert quality.evaluated_responses == 3
    assert quality.quality_coverage == pytest.approx(0.75)
    assert quality.judge_evaluated_responses == 2
    assert quality.judge_coverage == pytest.approx(2 / 3)
    assert quality.composite_score == pytest.approx(65.0)
    assert quality.item_weighted_composite_score == pytest.approx(70.0)
    assert quality.failure_as_zero_composite_score == pytest.approx(52.5)
    assert quality.deterministic_score == pytest.approx(65.0)
    assert quality.judge_score == pytest.approx(65.0)

    categories = {item.category: item for item in quality.categories}
    assert categories["general_knowledge"].measured_responses == 3
    assert categories["general_knowledge"].evaluated_responses == 2
    assert categories["general_knowledge"].quality_coverage == pytest.approx(2 / 3)
    assert categories["general_knowledge"].composite_score == pytest.approx(80.0)
    assert categories["coding"].composite_score == pytest.approx(50.0)
    assert categories["coding"].category_weight == 1.0
    assert any("failed responses" in item for item in quality.limitations)


def test_quality_does_not_impute_missing_evaluations(make_row) -> None:
    rows = [
        make_row(
            status="error",
            evaluation_status=None,
            deterministic=None,
            judge=None,
            composite=None,
            judge_missing=True,
        )
    ]
    quality = quality_analysis(rows)
    assert quality.evaluated_responses == 0
    assert quality.quality_coverage == 0.0
    assert quality.composite_score is None
    assert quality.item_weighted_composite_score is None
    assert quality.failure_as_zero_composite_score == 0.0
    assert quality.limitations
