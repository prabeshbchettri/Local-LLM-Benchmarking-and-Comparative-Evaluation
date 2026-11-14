"""
File: test_numeric.py

Description:
Positive and negative tests for numeric answer extraction and tolerance.

Responsibilities:
- Verify integers, decimals, fractions, scientific notation, and units.
- Verify wrong values, missing units, and ambiguous missing numbers fail.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from benchmark.dataset.models import DatasetItem
from benchmark.evaluation.deterministic.numeric import NumericEvaluator

pytestmark = [pytest.mark.unit, pytest.mark.evaluation]


def _numeric_item(**overrides: object) -> DatasetItem:
    payload: dict[str, object] = {
        "id": "mr-001",
        "category": "mathematical_reasoning",
        "difficulty": "easy",
        "prompt": "What is twelve multiplied by eight?",
        "expected_answer": "96",
        "reference_answer": "96",
        "evaluation_type": "numeric",
        "numeric_tolerance": 0.0,
    }
    payload.update(overrides)
    return DatasetItem.model_validate(payload)


def test_numeric_extracts_final_number(item_for) -> None:
    result = NumericEvaluator().evaluate(item_for("mr-001"), "12 * 8 = 96.")
    assert result.passed is True
    assert result.checks[0].metadata["actual"] == 96.0


def test_numeric_rejects_wrong_value(item_for) -> None:
    result = NumericEvaluator().evaluate(item_for("mr-001"), "The answer is 95.")
    assert result.passed is False
    assert result.score == 0.0


def test_numeric_accepts_fraction() -> None:
    item = _numeric_item(
        prompt="What is three quarters as a decimal?",
        expected_answer="0.75",
        reference_answer="0.75",
    )
    result = NumericEvaluator().evaluate(item, "The answer is 3/4.")
    assert result.passed is True


def test_numeric_accepts_scientific_notation() -> None:
    item = _numeric_item(
        prompt="What is one thousand five hundred in scientific notation?",
        expected_answer="1500",
        reference_answer="1500",
    )
    result = NumericEvaluator().evaluate(item, "The answer is 1.5e3.")
    assert result.passed is True


def test_numeric_applies_configured_tolerance() -> None:
    item = _numeric_item(
        prompt="Estimate the value within tolerance.",
        expected_answer="10",
        reference_answer="10",
        numeric_tolerance=0.5,
    )
    assert NumericEvaluator().evaluate(item, "The answer is 10.4.").passed is True
    assert NumericEvaluator().evaluate(item, "The answer is 10.6.").passed is False


def test_numeric_requires_unit_when_configured(item_for) -> None:
    assert NumericEvaluator().evaluate(item_for("mr-002"), "150 km").passed is True
    result = NumericEvaluator().evaluate(item_for("mr-002"), "150")
    assert result.passed is False
    assert result.checks[-1].name == "expected_unit"


def test_numeric_rejects_missing_number(item_for) -> None:
    result = NumericEvaluator().evaluate(item_for("mr-001"), "I cannot calculate that.")
    assert result.passed is False


def test_numeric_item_requires_numeric_expected_answer() -> None:
    with pytest.raises(ValidationError):
        _numeric_item(expected_answer="ninety-six")


def test_numeric_answer_colon_marker_is_recognized(item_for) -> None:
    result = NumericEvaluator().evaluate(item_for("mr-001"), "The answer: 96")
    assert result.passed is True
    assert result.checks[0].metadata["actual"] == 96.0


def test_numeric_uses_last_number_in_first_sentence_after_marker(item_for) -> None:
    result = NumericEvaluator().evaluate(
        item_for("mr-001"),
        "We multiply 12 by 8. The final answer is 96. Both factors were used.",
    )
    assert result.passed is True
    assert result.checks[0].metadata["actual"] == 96.0
