"""
File: exact_match.py

Description:
Deterministic exact-answer evaluator with documented normalization.

Responsibilities:
- Normalize candidate and accepted answers with NFKC and case folding.
- Collapse whitespace and remove edge punctuation.
- Award full credit only for an exact normalized match.
"""

from __future__ import annotations

from benchmark.core.exceptions import EvaluationConfigurationError
from benchmark.dataset.models import DatasetItem, EvaluationType
from benchmark.evaluation.deterministic.base import build_result, make_check
from benchmark.evaluation.models import DeterministicResult
from benchmark.evaluation.parsing import normalize_exact_text


class ExactMatchEvaluator:
    """Evaluate a response against the item's expected and accepted answers."""

    name = "exact_match"
    version = "1.0.0"
    evaluation_type = EvaluationType.EXACT_MATCH

    def evaluate(self, item: DatasetItem, response_text: str) -> DeterministicResult:
        if item.expected_answer is None:
            raise EvaluationConfigurationError("exact_match item has no expected_answer")

        accepted = [item.expected_answer, *item.accepted_answers]
        normalized_accepted = [normalize_exact_text(answer) for answer in accepted]
        normalized_response = normalize_exact_text(response_text)
        matched = normalized_response in normalized_accepted
        check = make_check(
            "normalized_exact_match",
            matched,
            detail=(
                "response matches an accepted answer after normalization"
                if matched
                else "response does not exactly match any accepted answer"
            ),
            metadata={
                "normalized_response": normalized_response,
                "normalized_accepted_answers": normalized_accepted,
            },
        )
        return build_result(
            item,
            evaluator=self.name,
            evaluation_type=self.evaluation_type,
            score=item.max_score if matched else 0.0,
            checks=[check],
            passed=matched,
            version=self.version,
        )
