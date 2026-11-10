"""
File: test_refusal.py

Description:
Positive and negative tests for unanswerable-item refusal checks.

Responsibilities:
- Verify explicit uncertainty and fictional-place language pass.
- Verify fabricated claims fail.
- Verify non-refusal responses fail.
"""

from __future__ import annotations

import pytest

from benchmark.evaluation.deterministic.refusal import RefusalEvaluator

pytestmark = [pytest.mark.unit, pytest.mark.evaluation]


def test_refusal_reference_passes(item_for) -> None:
    item = item_for("gk-008")
    result = RefusalEvaluator().evaluate(item, item.reference_answer or "")
    assert result.passed is True


def test_refusal_uncertainty_phrase_passes(item_for) -> None:
    result = RefusalEvaluator().evaluate(
        item_for("gk-008"), "I don't know the population; the city is fictional."
    )
    assert result.passed is True


def test_refusal_fabricated_claim_fails(item_for) -> None:
    result = RefusalEvaluator().evaluate(
        item_for("gk-008"), "The population of Eldoria is 50,000."
    )
    assert result.passed is False
    assert result.score == 0.0
    assert result.checks[1].passed is False


def test_refusal_without_uncertainty_fails(item_for) -> None:
    result = RefusalEvaluator().evaluate(
        item_for("gk-008"), "Eldoria is a city in Valoria."
    )
    assert result.passed is False
    assert result.checks[0].passed is False


def test_refusal_insufficient_information_phrase_passes(item_for) -> None:
    result = RefusalEvaluator().evaluate(
        item_for("lr-014"), "The information is insufficient to decide."
    )
    assert result.passed is True
