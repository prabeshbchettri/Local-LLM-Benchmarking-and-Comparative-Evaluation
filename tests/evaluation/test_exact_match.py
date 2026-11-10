"""
File: test_exact_match.py

Description:
Positive and negative tests for deterministic exact-match evaluation.

Responsibilities:
- Verify normalization and accepted-answer matching.
- Verify wrong answers fail.
- Verify raw normalization details are reported.
"""

from __future__ import annotations

import pytest

from benchmark.evaluation.deterministic.exact_match import ExactMatchEvaluator

pytestmark = [pytest.mark.unit, pytest.mark.evaluation]


def test_exact_match_normalizes_case_whitespace_and_edge_punctuation(item_for) -> None:
    result = ExactMatchEvaluator().evaluate(item_for("gk-001"), "  PARIS.  ")
    assert result.passed is True
    assert result.score == result.max_score
    assert result.checks[0].metadata["normalized_response"] == "paris"


def test_exact_match_accepts_alternative_answer(item_for) -> None:
    result = ExactMatchEvaluator().evaluate(item_for("gk-001"), "Paris, France")
    assert result.passed is True


def test_exact_match_rejects_wrong_answer(item_for) -> None:
    result = ExactMatchEvaluator().evaluate(item_for("gk-001"), "Lyon")
    assert result.passed is False
    assert result.score == 0.0


def test_exact_match_rejects_verbose_sentence(item_for) -> None:
    result = ExactMatchEvaluator().evaluate(
        item_for("gk-001"), "The capital of France is Paris."
    )
    assert result.passed is False
