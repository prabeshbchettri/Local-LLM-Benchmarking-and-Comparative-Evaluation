"""
File: json_schema.py

Description:
Deterministic JSON parsing, schema validation, and reference-value evaluator.

Responsibilities:
- Extract JSON from a response without repairing malformed syntax.
- Validate the parsed value against the item's JSON Schema.
- Compare the parsed value with the authored reference answer.
- Award partial credit only for schema-valid but reference-mismatched output.
"""

from __future__ import annotations

import json
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError

from benchmark.core.exceptions import EvaluationConfigurationError
from benchmark.dataset.models import DatasetItem, EvaluationType
from benchmark.evaluation.deterministic.base import build_result, make_check
from benchmark.evaluation.models import DeterministicResult
from benchmark.evaluation.parsing import JSONExtractionError, extract_json_value

SCHEMA_VALID_MISMATCH_CREDIT = 0.5


def _schema_error_message(error: ValidationError) -> str:
    path = ".".join(str(part) for part in error.absolute_path)
    location = f" at {path}" if path else ""
    return f"{error.message}{location}"


class JsonSchemaEvaluator:
    """Evaluate JSON syntax, schema conformance, and reference values."""

    name = "json_schema"
    version = "1.0.0"
    evaluation_type = EvaluationType.JSON_SCHEMA

    def evaluate(self, item: DatasetItem, response_text: str) -> DeterministicResult:
        if item.json_schema is None:
            raise EvaluationConfigurationError("json_schema item has no json_schema")
        if not item.reference_answer:
            raise EvaluationConfigurationError("json_schema item has no reference_answer")

        try:
            reference_value: Any = json.loads(item.reference_answer)
        except json.JSONDecodeError as exc:
            raise EvaluationConfigurationError(
                "json_schema item reference_answer is not valid JSON"
            ) from exc

        parsed_value: Any | None = None
        parse_error: str | None = None
        try:
            parsed_value = extract_json_value(response_text)
        except JSONExtractionError as exc:
            parse_error = str(exc)

        schema_errors: list[str] = []
        schema_valid = False
        if parse_error is None:
            try:
                validator = Draft202012Validator(item.json_schema)
                schema_errors = [
                    _schema_error_message(error)
                    for error in sorted(
                        validator.iter_errors(parsed_value),
                        key=lambda error: list(error.absolute_path),
                    )
                ]
                schema_valid = not schema_errors
            except SchemaError as exc:
                raise EvaluationConfigurationError(
                    f"item JSON Schema is invalid: {exc.message}"
                ) from exc

        reference_match = schema_valid and parsed_value == reference_value
        if parse_error is not None:
            score_fraction = 0.0
        elif not schema_valid:
            score_fraction = 0.0
        elif not reference_match:
            score_fraction = SCHEMA_VALID_MISMATCH_CREDIT
        else:
            score_fraction = 1.0

        checks = [
            make_check(
                "json_parse",
                parse_error is None,
                detail="valid JSON parsed" if parse_error is None else parse_error,
            ),
            make_check(
                "schema_valid",
                schema_valid,
                detail=(
                    "JSON conforms to the item schema"
                    if schema_valid
                    else "JSON does not conform to the item schema"
                ),
                metadata={"errors": schema_errors[:10]},
            ),
            make_check(
                "reference_match",
                reference_match,
                detail=(
                    "JSON values match the reference"
                    if reference_match
                    else "JSON values do not match the reference"
                ),
            ),
        ]
        return build_result(
            item,
            evaluator=self.name,
            evaluation_type=self.evaluation_type,
            score=item.max_score * score_fraction,
            checks=checks,
            passed=reference_match,
            version=self.version,
        )
