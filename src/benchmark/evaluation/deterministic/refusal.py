"""
File: refusal.py

Description:
Deterministic refusal and uncertainty evaluator for unanswerable items.

Responsibilities:
- Detect explicit uncertainty, inability, or fictional-information language.
- Reject responses containing authored forbidden fabricated claims.
- Require both appropriate refusal and absence of fabricated claims for credit.
"""

from __future__ import annotations

import re

from benchmark.core.exceptions import EvaluationConfigurationError
from benchmark.dataset.models import DatasetItem, EvaluationType
from benchmark.evaluation.deterministic.base import build_result, make_check
from benchmark.evaluation.models import DeterministicResult
from benchmark.evaluation.parsing import normalize_phrase

REFUSAL_PATTERNS = (
    r"\bcannot\s+(?:be\s+)?(?:determine|determined|answer|answered|know|known)\b",
    r"\bcan't\s+(?:be\s+)?(?:determine|determined|answer|answered|know|known)\b",
    r"\bunable\s+to\s+(?:determine|answer|know|provide)\b",
    r"\binsufficient\b",
    r"\bnot\s+enough\s+(?:information|data|evidence)\b",
    r"\b(?:could|might)\s+be\s+either\b",
    r"\b(?:information|data)\s+(?:is\s+)?(?:not\s+provided|not\s+available|missing)\b",
    r"\bnot\s+(?:provided|specified|available|known|recorded)\b",
    r"\bno\s+(?:way|information|data|records?)\s+to\s+(?:determine|know)\b",
    r"\bthere\s+(?:is|are)\s+no\s+(?:records?|information|data)\b",
    r"\b(?:does|do)\s+not\s+exist\b",
    r"\bfictional\b",
    r"\bnot\s+(?:a\s+)?real\b",
    r"\bimpossible\s+to\s+determine\b",
    r"\bunknown\b",
    r"\bi\s+(?:do\s+not|don't)\s+know\b",
)
REFUSAL_REGEX = re.compile("|".join(REFUSAL_PATTERNS), re.IGNORECASE)


class RefusalEvaluator:
    """Check whether an unanswerable item was refused without fabrication."""

    name = "refusal_check"
    version = "1.0.0"
    evaluation_type = EvaluationType.REFUSAL_CHECK

    def evaluate(self, item: DatasetItem, response_text: str) -> DeterministicResult:
        if not item.reference_answer:
            raise EvaluationConfigurationError("refusal_check item has no reference_answer")
        if not item.forbidden_claims:
            raise EvaluationConfigurationError("refusal_check item has no forbidden_claims")

        refusal_match = REFUSAL_REGEX.search(response_text)
        refusal_detected = refusal_match is not None
        normalized_response = normalize_phrase(response_text)
        found_claims = [
            claim
            for claim in item.forbidden_claims
            if normalize_phrase(claim) in normalized_response
        ]
        no_forbidden_claims = not found_claims
        passed = refusal_detected and no_forbidden_claims

        checks = [
            make_check(
                "refusal_detected",
                refusal_detected,
                detail=(
                    "explicit uncertainty or refusal language detected"
                    if refusal_detected
                    else "no explicit refusal or uncertainty language detected"
                ),
                metadata={
                    "matched_text": refusal_match.group(0) if refusal_match else None
                },
            ),
            make_check(
                "no_forbidden_claims",
                no_forbidden_claims,
                detail=(
                    "no forbidden fabricated claims detected"
                    if no_forbidden_claims
                    else "forbidden fabricated claims detected"
                ),
                metadata={"found_claims": found_claims},
            ),
        ]
        return build_result(
            item,
            evaluator=self.name,
            evaluation_type=self.evaluation_type,
            score=item.max_score if passed else 0.0,
            checks=checks,
            passed=passed,
            version=self.version,
        )
