"""
File: instruction_constraints.py

Description:
Deterministic instruction-following constraint evaluator.

Responsibilities:
- Evaluate every authored machine-checkable constraint independently.
- Support word, sentence, bullet, line, phrase, prefix, suffix, case, and order checks.
- Award proportional credit from the number of passed constraints.
"""

from __future__ import annotations

from benchmark.core.exceptions import EvaluationConfigurationError
from benchmark.dataset.models import DatasetItem, EvaluationType, InstructionConstraint
from benchmark.evaluation.deterministic.base import build_result, make_check
from benchmark.evaluation.models import DeterministicResult
from benchmark.evaluation.parsing import (
    count_bullet_lines,
    count_occurrences,
    count_sentences,
    non_empty_lines,
    normalize_phrase,
)


def _check_constraint(
    constraint: InstructionConstraint,
    response_text: str,
) -> tuple[bool, str, dict[str, object]]:
    normalized_response = normalize_phrase(response_text)
    constraint_type = constraint.type

    if constraint_type == "word_count":
        actual = len(response_text.split())
        passed = actual >= constraint.minimum and (
            constraint.maximum is None or actual <= constraint.maximum
        )
        return passed, f"word count {actual}", {"actual": actual}

    if constraint_type == "sentence_count":
        actual = count_sentences(response_text)
        passed = actual >= constraint.minimum and (
            constraint.maximum is None or actual <= constraint.maximum
        )
        return passed, f"sentence count {actual}", {"actual": actual}

    if constraint_type == "bullet_count":
        actual = count_bullet_lines(response_text)
        return actual == constraint.count, f"bullet count {actual}", {"actual": actual}

    if constraint_type == "line_count":
        actual = len(non_empty_lines(response_text))
        return actual == constraint.count, f"non-empty line count {actual}", {"actual": actual}

    if constraint_type == "required_phrase":
        present = normalize_phrase(constraint.phrase) in normalized_response
        return present, f"required phrase {'present' if present else 'missing'}", {
            "phrase": constraint.phrase
        }

    if constraint_type == "forbidden_phrase":
        absent = normalize_phrase(constraint.phrase) not in normalized_response
        return absent, f"forbidden phrase {'absent' if absent else 'present'}", {
            "phrase": constraint.phrase
        }

    if constraint_type == "occurrence_count":
        actual = count_occurrences(response_text, constraint.phrase)
        return actual == constraint.count, f"phrase occurrences {actual}", {
            "phrase": constraint.phrase,
            "actual": actual,
        }

    if constraint_type == "starts_with":
        passed = response_text.startswith(constraint.text)
        return passed, f"prefix {'present' if passed else 'missing'}", {
            "text": constraint.text
        }

    if constraint_type == "ends_with":
        passed = response_text.rstrip().endswith(constraint.text)
        return passed, f"suffix {'present' if passed else 'missing'}", {
            "text": constraint.text
        }

    if constraint_type == "lowercase":
        passed = response_text == response_text.lower()
        return passed, f"response is {'lowercase' if passed else 'not lowercase'}", {}

    lines = non_empty_lines(response_text)
    if constraint_type == "each_line_starts_with":
        passed = bool(lines) and all(line.startswith(constraint.text) for line in lines)
        return passed, f"line prefix {'present' if passed else 'missing'}", {
            "text": constraint.text,
            "line_count": len(lines),
        }

    if constraint_type == "each_line_ends_with":
        passed = bool(lines) and all(line.endswith(constraint.text) for line in lines)
        return passed, f"line suffix {'present' if passed else 'missing'}", {
            "text": constraint.text,
            "line_count": len(lines),
        }

    if constraint_type == "each_line_contains":
        normalized_text = normalize_phrase(constraint.text)
        passed = bool(lines) and all(
            normalized_text in normalize_phrase(line) for line in lines
        )
        return passed, f"line content {'present' if passed else 'missing'}", {
            "text": constraint.text,
            "line_count": len(lines),
        }

    if constraint_type == "contains_all":
        missing = [
            term for term in constraint.terms if normalize_phrase(term) not in normalized_response
        ]
        passed = not missing
        return passed, f"required terms {'present' if passed else 'missing'}", {
            "missing_terms": missing
        }

    if constraint_type == "ordered_terms":
        position = 0
        missing: list[str] = []
        for term in constraint.terms:
            index = normalized_response.find(normalize_phrase(term), position)
            if index < 0:
                missing.append(term)
                continue
            position = index + len(normalize_phrase(term))
        passed = not missing
        return passed, f"ordered terms {'present' if passed else 'out of order or missing'}", {
            "missing_terms": missing
        }

    if constraint_type == "exact_text":
        passed = response_text == constraint.text
        return passed, f"exact text {'matched' if passed else 'did not match'}", {}

    raise EvaluationConfigurationError(f"unsupported instruction constraint: {constraint_type}")


class InstructionConstraintEvaluator:
    """Evaluate all machine-checkable instructions attached to an item."""

    name = "instruction_constraints"
    version = "1.0.0"
    evaluation_type = EvaluationType.INSTRUCTION_CONSTRAINTS

    def evaluate(self, item: DatasetItem, response_text: str) -> DeterministicResult:
        if not item.required_constraints:
            raise EvaluationConfigurationError(
                "instruction_constraints item has no required_constraints"
            )

        checks = []
        passed_count = 0
        for index, constraint in enumerate(item.required_constraints, start=1):
            passed, detail, metadata = _check_constraint(constraint, response_text)
            passed_count += int(passed)
            checks.append(
                make_check(
                    f"constraint_{index}_{constraint.type}",
                    passed,
                    detail=detail,
                    metadata=metadata,
                )
            )

        score = item.max_score * passed_count / len(item.required_constraints)
        return build_result(
            item,
            evaluator=self.name,
            evaluation_type=self.evaluation_type,
            score=score,
            checks=checks,
            passed=passed_count == len(item.required_constraints),
            version=self.version,
        )
