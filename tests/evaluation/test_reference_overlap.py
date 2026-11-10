"""
File: test_reference_overlap.py

Description:
Positive and negative tests for reference overlap and key-fact coverage.

Responsibilities:
- Verify all key facts produce a passing result.
- Verify missing facts fail even with some lexical overlap.
- Verify the documented component weights with a hand-calculated fixture.
"""

from __future__ import annotations

import pytest

from benchmark.dataset.models import DatasetItem
from benchmark.evaluation.deterministic.reference_overlap import ReferenceOverlapEvaluator

pytestmark = [pytest.mark.unit, pytest.mark.evaluation]


def test_reference_overlap_all_key_facts_pass(item_for) -> None:
    item = item_for("sum-001")
    result = ReferenceOverlapEvaluator().evaluate(
        item, " ".join(item.key_facts)
    )
    assert result.passed is True
    assert result.checks[2].score == 1.0


def test_reference_overlap_missing_fact_fails(item_for) -> None:
    item = item_for("sum-001")
    result = ReferenceOverlapEvaluator().evaluate(item, item.key_facts[0])
    assert result.passed is False
    assert result.checks[2].score < 1.0
    assert result.checks[3].passed is False


def test_reference_overlap_hand_calculated_weights() -> None:
    item = DatasetItem.model_validate(
        {
            "id": "sum-001",
            "category": "summarization",
            "difficulty": "easy",
            "prompt": "Summarize the short reference passage in your own words.",
            "reference_answer": "alpha beta gamma",
            "evaluation_type": "reference_overlap",
            "key_facts": ["alpha beta"],
            "judge_eligible": True,
        }
    )
    result = ReferenceOverlapEvaluator().evaluate(item, "alpha beta")
    assert result.score == pytest.approx(0.9)
    assert result.checks[0].score == pytest.approx(0.8)
    assert result.checks[1].score == pytest.approx(0.8)
    assert result.checks[2].score == pytest.approx(1.0)
