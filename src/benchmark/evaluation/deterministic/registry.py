"""
File: registry.py

Description:
Evaluation-type dispatcher for deterministic evaluators.

Responsibilities:
- Register one evaluator per supported evaluation type.
- Resolve the evaluator selected by a dataset item.
- Provide a default registry with all authored deterministic methods.
"""

from __future__ import annotations

from benchmark.core.exceptions import EvaluationConfigurationError
from benchmark.dataset.models import DatasetItem, EvaluationType
from benchmark.evaluation.deterministic.base import Evaluator
from benchmark.evaluation.deterministic.code_tests import CodeTestsEvaluator
from benchmark.evaluation.deterministic.exact_match import ExactMatchEvaluator
from benchmark.evaluation.deterministic.instruction_constraints import (
    InstructionConstraintEvaluator,
)
from benchmark.evaluation.deterministic.json_schema import JsonSchemaEvaluator
from benchmark.evaluation.deterministic.numeric import NumericEvaluator
from benchmark.evaluation.deterministic.reference_overlap import ReferenceOverlapEvaluator
from benchmark.evaluation.deterministic.refusal import RefusalEvaluator
from benchmark.evaluation.models import DeterministicResult


class EvaluatorRegistry:
    """Dispatch dataset items to the evaluator registered for their method."""

    def __init__(self) -> None:
        self._evaluators: dict[EvaluationType, Evaluator] = {}

    def register(self, evaluator: Evaluator) -> None:
        """Register an evaluator, rejecting duplicate evaluation types."""
        if evaluator.evaluation_type in self._evaluators:
            raise ValueError(
                f"evaluator already registered for {evaluator.evaluation_type.value}"
            )
        self._evaluators[evaluator.evaluation_type] = evaluator

    def get(self, evaluation_type: EvaluationType) -> Evaluator:
        """Return the evaluator for a method or raise a configuration error."""
        try:
            return self._evaluators[evaluation_type]
        except KeyError as exc:
            raise EvaluationConfigurationError(
                f"no deterministic evaluator registered for {evaluation_type.value}"
            ) from exc

    def evaluate(self, item: DatasetItem, response_text: str) -> DeterministicResult:
        """Dispatch an item to its registered evaluator."""
        return self.get(item.evaluation_type).evaluate(item, response_text)


def build_default_registry() -> EvaluatorRegistry:
    """Build the registry used by the evaluation engine."""
    registry = EvaluatorRegistry()
    registry.register(ExactMatchEvaluator())
    registry.register(NumericEvaluator())
    registry.register(CodeTestsEvaluator())
    registry.register(JsonSchemaEvaluator())
    registry.register(InstructionConstraintEvaluator())
    registry.register(ReferenceOverlapEvaluator())
    registry.register(RefusalEvaluator())
    return registry
