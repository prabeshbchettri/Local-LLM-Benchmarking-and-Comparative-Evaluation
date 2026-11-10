"""
File: __init__.py

Description:
Deterministic evaluator implementations and the evaluation-type dispatcher.

Responsibilities:
- Export the public deterministic evaluation API.
"""

from benchmark.evaluation.deterministic.base import Evaluator, build_result, make_check
from benchmark.evaluation.deterministic.code_tests import CodeTestsEvaluator
from benchmark.evaluation.deterministic.exact_match import ExactMatchEvaluator
from benchmark.evaluation.deterministic.instruction_constraints import (
    InstructionConstraintEvaluator,
)
from benchmark.evaluation.deterministic.json_schema import JsonSchemaEvaluator
from benchmark.evaluation.deterministic.numeric import NumericEvaluator
from benchmark.evaluation.deterministic.reference_overlap import ReferenceOverlapEvaluator
from benchmark.evaluation.deterministic.refusal import RefusalEvaluator
from benchmark.evaluation.deterministic.registry import EvaluatorRegistry, build_default_registry

__all__ = [
    "CodeTestsEvaluator",
    "Evaluator",
    "EvaluatorRegistry",
    "ExactMatchEvaluator",
    "InstructionConstraintEvaluator",
    "JsonSchemaEvaluator",
    "NumericEvaluator",
    "ReferenceOverlapEvaluator",
    "RefusalEvaluator",
    "build_default_registry",
    "build_result",
    "make_check",
]
