"""
File: reference_overlap.py

Description:
Lexical reference-overlap evaluator for summarization and similar items.

Responsibilities:
- Compute unigram precision, recall, and F1 against a reference answer.
- Compute a longest-common-subsequence F1 analogue.
- Compute configured key-fact coverage.
- Combine the components into a documented default score.
"""

from __future__ import annotations

from collections import Counter

from benchmark.core.exceptions import EvaluationConfigurationError
from benchmark.dataset.models import DatasetItem, EvaluationType
from benchmark.evaluation.deterministic.base import build_result, make_check
from benchmark.evaluation.models import DeterministicResult
from benchmark.evaluation.parsing import normalize_phrase, tokenize_words

UNIGRAM_WEIGHT = 0.25
LCS_WEIGHT = 0.25
KEY_FACT_WEIGHT = 0.50


def _f1(overlap: int, candidate_count: int, reference_count: int) -> tuple[float, float, float]:
    if candidate_count == 0 or reference_count == 0:
        return 0.0, 0.0, 0.0
    precision = overlap / candidate_count
    recall = overlap / reference_count
    if precision + recall == 0:
        return precision, recall, 0.0
    return precision, recall, 2 * precision * recall / (precision + recall)


def _lcs_length(left: list[str], right: list[str]) -> int:
    if not left or not right:
        return 0
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for index, right_token in enumerate(right, start=1):
            if left_token == right_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[index - 1]))
        previous = current
    return previous[-1]


def _contains_sequence(haystack: list[str], needle: list[str]) -> bool:
    if not needle:
        return True
    width = len(needle)
    return any(
        haystack[index : index + width] == needle
        for index in range(len(haystack) - width + 1)
    )


def _key_fact_covered(fact: str, response_text: str, response_tokens: list[str]) -> bool:
    if normalize_phrase(fact) in normalize_phrase(response_text):
        return True
    fact_tokens = tokenize_words(fact)
    return _contains_sequence(response_tokens, fact_tokens)


class ReferenceOverlapEvaluator:
    """Score lexical overlap and key-fact coverage against a reference answer."""

    name = "reference_overlap"
    version = "1.0.0"
    evaluation_type = EvaluationType.REFERENCE_OVERLAP

    def evaluate(self, item: DatasetItem, response_text: str) -> DeterministicResult:
        if not item.reference_answer:
            raise EvaluationConfigurationError(
                "reference_overlap item has no reference_answer"
            )
        if not item.key_facts:
            raise EvaluationConfigurationError("reference_overlap item has no key_facts")

        reference_tokens = tokenize_words(item.reference_answer)
        response_tokens = tokenize_words(response_text)
        reference_counts = Counter(reference_tokens)
        response_counts = Counter(response_tokens)
        overlap = sum(
            min(count, reference_counts[token])
            for token, count in response_counts.items()
        )
        precision, recall, unigram_f1 = _f1(
            overlap, len(response_tokens), len(reference_tokens)
        )

        lcs = _lcs_length(response_tokens, reference_tokens)
        lcs_precision, lcs_recall, lcs_f1 = _f1(
            lcs, len(response_tokens), len(reference_tokens)
        )

        covered = [
            fact
            for fact in item.key_facts
            if _key_fact_covered(fact, response_text, response_tokens)
        ]
        coverage = len(covered) / len(item.key_facts)
        composite = (
            UNIGRAM_WEIGHT * unigram_f1
            + LCS_WEIGHT * lcs_f1
            + KEY_FACT_WEIGHT * coverage
        )
        passed = coverage >= 1.0

        checks = [
            make_check(
                "unigram_f1",
                unigram_f1 >= 0.999999,
                score=unigram_f1,
                detail=f"unigram precision={precision:.4f} recall={recall:.4f}",
                metadata={
                    "precision": precision,
                    "recall": recall,
                    "overlap": overlap,
                },
            ),
            make_check(
                "lcs_f1",
                lcs_f1 >= 0.999999,
                score=lcs_f1,
                detail=f"LCS precision={lcs_precision:.4f} recall={lcs_recall:.4f}",
                metadata={
                    "precision": lcs_precision,
                    "recall": lcs_recall,
                    "lcs_tokens": lcs,
                },
            ),
            make_check(
                "key_fact_coverage",
                coverage >= 1.0,
                score=coverage,
                detail=f"{len(covered)} of {len(item.key_facts)} key facts covered",
                metadata={"covered_facts": covered, "missing_facts": [
                    fact for fact in item.key_facts if fact not in covered
                ]},
            ),
            make_check(
                "all_key_facts",
                passed,
                detail="all required key facts are present" if passed else "key facts are missing",
            ),
        ]
        return build_result(
            item,
            evaluator=self.name,
            evaluation_type=self.evaluation_type,
            score=item.max_score * composite,
            checks=checks,
            passed=passed,
            version=self.version,
        )
