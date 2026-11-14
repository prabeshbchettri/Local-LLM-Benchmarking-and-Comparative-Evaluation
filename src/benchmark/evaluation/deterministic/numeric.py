"""
File: numeric.py

Description:
Deterministic numeric-answer evaluator with tolerance and unit checks.

Responsibilities:
- Extract the final integer, decimal, fraction, or scientific-notation answer.
- Apply item tolerance or the documented default tolerance.
- Require an expected unit when the item defines one.
"""

from __future__ import annotations

import re

from benchmark.core.exceptions import EvaluationConfigurationError
from benchmark.dataset.models import DatasetItem, EvaluationType
from benchmark.evaluation.deterministic.base import build_result, make_check
from benchmark.evaluation.models import DeterministicResult
from benchmark.evaluation.parsing import normalize_phrase

NUMBER_TOKEN = re.compile(
    r"(?<!\w)(?:"
    r"(?P<fraction>[-+]?\d+\s*/\s*\d+)"
    r"|"
    r"(?P<number>[-+]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?|\.\d+)"
    r"(?:[eE][-+]?\d+)?)"
    r")(?!\w)"
)
ANSWER_MARKER = re.compile(
    r"\b(?:final\s+answer|answer\s+is|result\s+is|therefore)\b|\banswer\s*:",
    re.IGNORECASE,
)
SENTENCE_END = re.compile(r"[.!?](?=\s|$)")
UNIT_ALIASES: dict[str, tuple[str, ...]] = {
    "dollars": ("dollars", "dollar", "usd", "$"),
    "kilometers": ("kilometers", "kilometer", "km"),
    "kilometers per hour": ("kilometers per hour", "kilometer per hour", "km/h", "kph"),
    "square centimeters": ("square centimeters", "square centimetres", "cm2", "cm²", "cm^2"),
    "square meters": ("square meters", "square metres", "m2", "m²", "m^2"),
    "cups": ("cups", "cup"),
    "marbles": ("marbles", "marble"),
}


def _parse_number(text: str) -> float | None:
    value = text.replace(",", "").strip()
    try:
        if "/" in value:
            numerator, denominator = value.split("/", maxsplit=1)
            denominator_value = float(denominator)
            if denominator_value == 0:
                return None
            return float(numerator) / denominator_value
        return float(value)
    except ValueError:
        return None


def extract_final_number(text: str) -> tuple[float | None, int]:
    """Return the final numeric value and the number of candidates found."""
    candidates: list[tuple[int, float]] = []
    for match in NUMBER_TOKEN.finditer(text):
        raw = match.group("fraction") or match.group("number")
        if raw is None:
            continue
        value = _parse_number(raw)
        if value is not None:
            candidates.append((match.start(), value))
    if not candidates:
        return None, 0

    marker_matches = list(ANSWER_MARKER.finditer(text))
    if marker_matches:
        marker_start = marker_matches[-1].end()
        after_marker = [value for start, value in candidates if start >= marker_start]
        if after_marker:
            sentence_end = SENTENCE_END.search(text, marker_start)
            if sentence_end is not None:
                in_sentence = [
                    value
                    for start, value in candidates
                    if marker_start <= start < sentence_end.start()
                ]
                if in_sentence:
                    return in_sentence[-1], len(candidates)
            return after_marker[0], len(candidates)
    return candidates[-1][1], len(candidates)


def _contains_unit(response_text: str, expected_unit: str) -> bool:
    normalized_response = normalize_phrase(response_text)
    aliases = UNIT_ALIASES.get(normalize_phrase(expected_unit), (normalize_phrase(expected_unit),))
    for alias in aliases:
        normalized_alias = normalize_phrase(alias)
        if not normalized_alias:
            continue
        if normalized_alias.isalpha() and len(normalized_alias) > 2:
            if re.search(rf"\b{re.escape(normalized_alias)}\b", normalized_response):
                return True
        elif normalized_alias in normalized_response:
            return True
    return False


class NumericEvaluator:
    """Evaluate the final numeric answer with tolerance and optional units."""

    name = "numeric"
    version = "1.0.0"
    evaluation_type = EvaluationType.NUMERIC

    def evaluate(self, item: DatasetItem, response_text: str) -> DeterministicResult:
        if item.expected_answer is None:
            raise EvaluationConfigurationError("numeric item has no expected_answer")
        try:
            expected_value = float(item.expected_answer)
        except ValueError as exc:
            raise EvaluationConfigurationError(
                "numeric item expected_answer is not a number"
            ) from exc

        actual_value, candidate_count = extract_final_number(response_text)
        tolerance = (
            item.numeric_tolerance
            if item.numeric_tolerance is not None
            else max(1e-6, abs(expected_value) * 0.005)
        )
        numeric_match = (
            actual_value is not None
            and abs(actual_value - expected_value) <= tolerance + 1e-12
        )
        checks = [
            make_check(
                "numeric_value",
                numeric_match,
                detail=(
                    "final numeric answer is within tolerance"
                    if numeric_match
                    else "final numeric answer is missing or outside tolerance"
                ),
                metadata={
                    "expected": expected_value,
                    "actual": actual_value,
                    "tolerance": tolerance,
                    "candidate_count": candidate_count,
                },
            )
        ]

        unit_match = True
        if item.expected_unit:
            unit_match = _contains_unit(response_text, item.expected_unit)
            checks.append(
                make_check(
                    "expected_unit",
                    unit_match,
                    detail=(
                        "expected unit is present"
                        if unit_match
                        else "expected unit is missing"
                    ),
                    metadata={"expected_unit": item.expected_unit},
                )
            )

        passed = numeric_match and unit_match
        return build_result(
            item,
            evaluator=self.name,
            evaluation_type=self.evaluation_type,
            score=item.max_score if passed else 0.0,
            checks=checks,
            passed=passed,
            version=self.version,
        )
