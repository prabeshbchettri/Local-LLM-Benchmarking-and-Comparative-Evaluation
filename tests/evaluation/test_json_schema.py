"""
File: test_json_schema.py

Description:
Positive and negative tests for JSON parsing, schema validation, and values.

Responsibilities:
- Verify valid fenced JSON and reference values pass.
- Verify malformed JSON, schema violations, and wrong values do not pass.
- Verify schema-valid but value-mismatched output receives documented partial credit.
"""

from __future__ import annotations

import pytest

from benchmark.evaluation.deterministic.json_schema import JsonSchemaEvaluator

pytestmark = [pytest.mark.unit, pytest.mark.evaluation]


def test_json_reference_passes(item_for) -> None:
    item = item_for("json-001")
    result = JsonSchemaEvaluator().evaluate(item, item.reference_answer or "")
    assert result.passed is True
    assert result.score == result.max_score


def test_json_fenced_output_passes(item_for) -> None:
    result = JsonSchemaEvaluator().evaluate(
        item_for("json-001"), '```json\n{"name": "Ada", "age": 36}\n```'
    )
    assert result.passed is True


def test_json_malformed_output_fails(item_for) -> None:
    result = JsonSchemaEvaluator().evaluate(item_for("json-001"), '{"name": "Ada",}')
    assert result.passed is False
    assert result.score == 0.0
    assert result.checks[0].passed is False


def test_json_schema_invalid_output_fails(item_for) -> None:
    result = JsonSchemaEvaluator().evaluate(
        item_for("json-001"), '{"name": "Ada", "age": 36, "extra": true}'
    )
    assert result.passed is False
    assert result.score == 0.0
    assert result.checks[1].passed is False


def test_json_wrong_values_get_partial_credit(item_for) -> None:
    result = JsonSchemaEvaluator().evaluate(
        item_for("json-001"), '{"name": "Grace", "age": 85}'
    )
    assert result.passed is False
    assert result.score == 0.5
    assert result.checks[2].passed is False
