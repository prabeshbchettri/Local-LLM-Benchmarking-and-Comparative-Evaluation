"""
File: __init__.py

Description:
Deterministic evaluation, blind local judge, scoring, and reliability API.

Responsibilities:
- Export the public evaluation engine API.
"""

from benchmark.evaluation.deterministic import (
    CodeTestsEvaluator,
    Evaluator,
    EvaluatorRegistry,
    ExactMatchEvaluator,
    InstructionConstraintEvaluator,
    JsonSchemaEvaluator,
    NumericEvaluator,
    ReferenceOverlapEvaluator,
    RefusalEvaluator,
    build_default_registry,
    build_result,
    make_check,
)
from benchmark.evaluation.engine import EvaluationEngine, EvaluationRequest
from benchmark.evaluation.judge import (
    Judge,
    OllamaJudge,
    build_ollama_judge,
    parse_judge_output,
)
from benchmark.evaluation.models import (
    EVALUATION_SCHEMA_VERSION,
    EVALUATOR_VERSION,
    CategoryScoreSummary,
    CheckResult,
    DeterministicResult,
    EvaluationError,
    EvaluationResult,
    EvaluationStatus,
    EvaluationSummary,
    FamilyBiasSummary,
    JudgeConfig,
    JudgeDimensionScore,
    JudgeRawOutput,
    JudgeReliabilitySummary,
    JudgeRequest,
    JudgeResult,
    JudgeStatus,
    JudgeVerification,
    ManualAgreementSummary,
    ManualJudgeScore,
)
from benchmark.evaluation.rubric import (
    JUDGE_DIMENSIONS,
    JUDGE_PROMPT_VERSION,
    RUBRIC_VERSION,
    build_judge_messages,
)
from benchmark.evaluation.sandbox import CodeSandbox, SandboxTestOutcome
from benchmark.evaluation.scoring import (
    DEFAULT_CATEGORY_JUDGE_WEIGHTS,
    compare_manual_scores,
    score_results,
    summarize_family_bias,
    summarize_judge_reliability,
)

__all__ = [
    "CodeSandbox",
    "CodeTestsEvaluator",
    "CategoryScoreSummary",
    "CheckResult",
    "DEFAULT_CATEGORY_JUDGE_WEIGHTS",
    "DeterministicResult",
    "EVALUATION_SCHEMA_VERSION",
    "EVALUATOR_VERSION",
    "EvaluationEngine",
    "EvaluationError",
    "EvaluationRequest",
    "EvaluationResult",
    "EvaluationStatus",
    "EvaluationSummary",
    "Evaluator",
    "EvaluatorRegistry",
    "ExactMatchEvaluator",
    "FamilyBiasSummary",
    "InstructionConstraintEvaluator",
    "JUDGE_DIMENSIONS",
    "JUDGE_PROMPT_VERSION",
    "Judge",
    "JudgeConfig",
    "JudgeDimensionScore",
    "JudgeRawOutput",
    "JudgeReliabilitySummary",
    "JudgeRequest",
    "JudgeResult",
    "JudgeStatus",
    "JudgeVerification",
    "JsonSchemaEvaluator",
    "ManualAgreementSummary",
    "ManualJudgeScore",
    "NumericEvaluator",
    "OllamaJudge",
    "RUBRIC_VERSION",
    "ReferenceOverlapEvaluator",
    "RefusalEvaluator",
    "SandboxTestOutcome",
    "build_default_registry",
    "build_judge_messages",
    "build_ollama_judge",
    "build_result",
    "compare_manual_scores",
    "make_check",
    "parse_judge_output",
    "score_results",
    "summarize_family_bias",
    "summarize_judge_reliability",
]
