"""
File: test_instruction_constraints.py

Description:
Positive and negative tests for instruction-following constraint checks.

Responsibilities:
- Verify sentence, word, line, bullet, phrase, prefix, suffix, case, and order checks.
- Verify partial credit is proportional to passed constraints.
"""

from __future__ import annotations

import pytest

from benchmark.dataset.models import DatasetItem
from benchmark.evaluation.deterministic.instruction_constraints import (
    InstructionConstraintEvaluator,
)

pytestmark = [pytest.mark.unit, pytest.mark.evaluation]


def test_instruction_reference_passes(item_for) -> None:
    item = item_for("if-001")
    result = InstructionConstraintEvaluator().evaluate(item, item.reference_answer or "")
    assert result.passed is True
    assert result.score == result.max_score


def test_instruction_wrong_sentence_count_fails(item_for) -> None:
    result = InstructionConstraintEvaluator().evaluate(
        item_for("if-001"),
        "Sleep helps the body repair itself. Sleep also supports memory. It improves mood.",
    )
    assert result.passed is False
    assert result.score < result.max_score


def test_instruction_forbidden_phrase_fails(item_for) -> None:
    result = InstructionConstraintEvaluator().evaluate(
        item_for("if-001"),
        "Sleep helps the body repair itself. Tired people need more rest.",
    )
    assert result.passed is False
    assert any(check.name.endswith("forbidden_phrase") and not check.passed for check in result.checks)


def test_instruction_lowercase_and_word_count(item_for) -> None:
    item = item_for("if-012")
    assert InstructionConstraintEvaluator().evaluate(item, "the sky is very blue").passed is True
    result = InstructionConstraintEvaluator().evaluate(item, "The sky is very blue")
    assert result.passed is False


def test_instruction_partial_credit() -> None:
    item = DatasetItem.model_validate(
        {
            "id": "if-001",
            "category": "instruction_following",
            "difficulty": "easy",
            "prompt": "Write one lowercase sentence containing the word water.",
            "reference_answer": "water is essential.",
            "evaluation_type": "instruction_constraints",
            "required_constraints": [
                {"type": "word_count", "minimum": 3, "maximum": 3},
                {"type": "lowercase"},
            ],
        }
    )
    result = InstructionConstraintEvaluator().evaluate(item, "Water is essential")
    assert result.score == 0.5
    assert result.passed is False


def test_instruction_line_checks(item_for) -> None:
    item = item_for("if-004")
    result = InstructionConstraintEvaluator().evaluate(
        item, "First line.\nSecond line.\nThird line."
    )
    assert result.passed is True
