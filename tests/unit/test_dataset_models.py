"""
File: test_dataset_models.py

Description:
Unit tests for dataset item validation, evaluation requirements, and hashing.

Responsibilities:
- Verify item IDs, evaluation fields, and method-specific constraints.
- Verify refusal items are consistently marked unanswerable.
- Verify canonical content hashing is deterministic.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from benchmark.dataset.models import DatasetItem, compute_dataset_content_hash

pytestmark = pytest.mark.unit


def _item(**overrides: object) -> DatasetItem:
    payload: dict[str, object] = {
        "id": "gk-001",
        "category": "general_knowledge",
        "difficulty": "easy",
        "prompt": "What is the capital city of France?",
        "expected_answer": "Paris",
        "reference_answer": "Paris",
        "evaluation_type": "exact_match",
    }
    payload.update(overrides)
    return DatasetItem.model_validate(payload)


def test_valid_exact_match_item() -> None:
    item = _item()
    assert item.id == "gk-001"
    assert item.evaluation_config()["evaluation_type"] == "exact_match"


def test_item_id_prefix_must_match_category() -> None:
    with pytest.raises(ValidationError, match="prefix"):
        _item(id="lr-001")


def test_exact_match_requires_expected_answer() -> None:
    with pytest.raises(ValidationError, match="expected_answer"):
        _item(expected_answer=None)


def test_numeric_item_requires_numeric_expected_answer() -> None:
    with pytest.raises(ValidationError, match="numeric expected_answer"):
        _item(
            id="mr-001",
            category="mathematical_reasoning",
            prompt="What is the value of the expression twelve plus eight?",
            expected_answer="not-a-number",
            evaluation_type="numeric",
            numeric_tolerance=0.0,
        )


def test_numeric_item_includes_tolerance_and_unit() -> None:
    item = _item(
        id="mr-001",
        category="mathematical_reasoning",
        prompt="What is the value of the expression twelve plus eight?",
        expected_answer="20",
        evaluation_type="numeric",
        numeric_tolerance=0.5,
        expected_unit="points",
    )
    config = item.evaluation_config()
    assert config["tolerance"] == 0.5
    assert config["expected_unit"] == "points"


def test_code_item_requires_test_cases() -> None:
    with pytest.raises(ValidationError, match="test case"):
        _item(
            id="code-001",
            category="coding",
            prompt="Write a Python function named add that returns the sum of two numbers.",
            expected_answer=None,
            reference_answer="def add(a, b):\n    return a + b\n",
            evaluation_type="code_tests",
            function_name="add",
            function_signature="def add(a, b):",
        )


def test_code_item_serializes_test_cases() -> None:
    item = _item(
        id="code-001",
        category="coding",
        prompt="Write a Python function named add that returns the sum of two numbers.",
        expected_answer=None,
        reference_answer="def add(a, b):\n    return a + b\n",
        evaluation_type="code_tests",
        function_name="add",
        function_signature="def add(a, b):",
        test_cases=[{"name": "basic", "args": [2, 3], "expected": 5}],
    )
    assert item.evaluation_config()["test_cases"][0]["name"] == "basic"


def test_json_item_requires_schema_and_reference() -> None:
    with pytest.raises(ValidationError, match="json_schema"):
        _item(
            id="json-001",
            category="structured_json",
            prompt="Return a JSON object containing the key name with the value Ada.",
            expected_answer=None,
            reference_answer='{"name": "Ada"}',
            evaluation_type="json_schema",
        )


def test_instruction_constraint_preserves_significant_whitespace() -> None:
    item = _item(
        id="if-001",
        category="instruction_following",
        prompt="List one benefit of drinking water and start the line with a hyphen and a space.",
        expected_answer=None,
        reference_answer="- Water supports digestion.",
        evaluation_type="instruction_constraints",
        required_constraints=[
            {"type": "each_line_starts_with", "text": "- "},
        ],
    )
    assert item.required_constraints[0].text == "- "


def test_instruction_item_requires_constraints() -> None:
    with pytest.raises(ValidationError, match="required_constraints"):
        _item(
            id="if-001",
            category="instruction_following",
            prompt="Write exactly one sentence about the value of clean drinking water.",
            expected_answer=None,
            reference_answer="Clean drinking water supports public health.",
            evaluation_type="instruction_constraints",
        )


def test_refusal_item_must_be_unanswerable() -> None:
    with pytest.raises(ValidationError, match="must set is_unanswerable"):
        _item(
            evaluation_type="refusal_check",
            reference_answer="The information is not available.",
            forbidden_claims=["The answer is"],
        )


def test_unanswerable_item_must_use_refusal_check() -> None:
    with pytest.raises(ValidationError, match="must use refusal_check"):
        _item(is_unanswerable=True)


def test_dataset_content_hash_is_order_independent() -> None:
    first = _item()
    second = _item(
        id="gk-002",
        prompt="How many continents are commonly recognized on Earth?",
        expected_answer="7",
        reference_answer="7",
    )
    assert compute_dataset_content_hash([first, second]) == compute_dataset_content_hash(
        [second, first]
    )
