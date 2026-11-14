"""
File: test_code_tests.py

Description:
Positive and negative tests for the restricted code runner.

Responsibilities:
- Verify correct reference code passes authored tests.
- Verify wrong code, syntax errors, and missing functions fail.
- Verify per-test timeouts and captured-output caps are enforced.
"""

from __future__ import annotations

import subprocess

import pytest

from benchmark.dataset.models import DatasetItem
from benchmark.evaluation.deterministic.code_tests import CodeTestsEvaluator
from benchmark.evaluation.sandbox import CodeSandbox, CodeTestCase

pytestmark = [pytest.mark.unit, pytest.mark.evaluation]


def test_correct_reference_code_passes(item_for) -> None:
    item = item_for("code-001")
    result = CodeTestsEvaluator().evaluate(item, item.reference_answer or "")
    assert result.passed is True
    assert result.score == result.max_score


def test_fenced_correct_code_passes(item_for) -> None:
    item = item_for("code-001")
    result = CodeTestsEvaluator().evaluate(
        item,
        "Here is the solution:\n```python\ndef add(a, b):\n    return a + b\n```",
    )
    assert result.passed is True


def test_wrong_code_fails(item_for) -> None:
    result = CodeTestsEvaluator().evaluate(
        item_for("code-001"), "def add(a, b):\n    return a - b"
    )
    assert result.passed is False
    assert result.score == pytest.approx(1 / 3)


def test_wrong_signature_fails(item_for) -> None:
    result = CodeTestsEvaluator().evaluate(
        item_for("code-001"), "def add(x, y):\n    return x + y"
    )
    assert result.passed is False
    signature_check = next(
        check for check in result.checks if check.name == "function_signature"
    )
    assert signature_check.passed is False


def test_syntax_error_fails(item_for) -> None:
    result = CodeTestsEvaluator().evaluate(
        item_for("code-001"), "def add(a, b)\n    return a + b"
    )
    assert result.passed is False
    assert result.checks[1].passed is False


def test_missing_function_fails(item_for) -> None:
    result = CodeTestsEvaluator().evaluate(
        item_for("code-001"), "def subtract(a, b):\n    return a - b"
    )
    assert result.passed is False
    assert result.checks[1].passed is False


def _timeout_item() -> DatasetItem:
    return DatasetItem.model_validate(
        {
            "id": "code-001",
            "category": "coding",
            "difficulty": "easy",
            "prompt": "Write a Python function named add that returns the sum of two numbers.",
            "reference_answer": "def add(a, b):\n    return a + b",
            "evaluation_type": "code_tests",
            "function_name": "add",
            "function_signature": "def add(a, b):",
            "test_cases": [{"name": "timeout", "args": [1, 2], "expected": 3}],
        }
    )


def test_timeout_is_recorded() -> None:
    sandbox = CodeSandbox(per_test_timeout_seconds=0.2, total_timeout_seconds=1.0)
    evaluator = CodeTestsEvaluator(sandbox)
    result = evaluator.evaluate(
        _timeout_item(), "def add(a, b):\n    while True:\n        pass"
    )
    assert result.passed is False
    test_check = next(check for check in result.checks if check.name == "test_timeout")
    assert test_check.passed is False
    assert test_check.metadata["timed_out"] is True


def test_timeout_never_counts_as_passed_even_with_stale_result() -> None:
    sandbox = CodeSandbox(per_test_timeout_seconds=0.2, total_timeout_seconds=1.0)
    completed = subprocess.CompletedProcess(
        args=["python"],
        returncode=None,
        stdout="",
        stderr="",
    )
    outcome = sandbox._outcome_from_result(
        CodeTestCase(name="timeout", args=[], expected=None),
        duration=0.2,
        completed=completed,
        result={"passed": True},
        timed_out=True,
    )
    assert outcome.passed is False
    assert outcome.timed_out is True


def test_output_is_capped() -> None:
    sandbox = CodeSandbox(output_limit=50)
    evaluator = CodeTestsEvaluator(sandbox)
    result = evaluator.evaluate(
        _timeout_item(), "def add(a, b):\n    print('x' * 1000)\n    return a + b"
    )
    test_check = next(check for check in result.checks if check.name == "test_timeout")
    assert test_check.passed is True
    assert test_check.metadata["output_truncated"] is True
    assert "output truncated" in test_check.metadata["stdout"]
