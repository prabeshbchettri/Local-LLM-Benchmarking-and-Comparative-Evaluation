"""
File: code_tests.py

Description:
Deterministic coding evaluator using authored tests and a restricted runner.

Responsibilities:
- Extract the first complete Python code block or full response.
- Verify the required top-level function exists before execution.
- Run every authored test in the sandbox and award pass@1-style proportional credit.
"""

from __future__ import annotations

import ast

from benchmark.core.exceptions import EvaluationConfigurationError
from benchmark.dataset.models import DatasetItem, EvaluationType
from benchmark.evaluation.deterministic.base import build_result, make_check
from benchmark.evaluation.models import DeterministicResult
from benchmark.evaluation.parsing import extract_code_block
from benchmark.evaluation.sandbox import CodeSandbox


def _find_function(code: str, function_name: str) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    tree = ast.parse(code)
    for node in tree.body:
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == function_name
        ):
            return node
    return None


def _function_defined(code: str, function_name: str) -> bool:
    return _find_function(code, function_name) is not None


def _signature_arguments(signature: str) -> list[str] | None:
    try:
        tree = ast.parse(f"{signature}\n    pass")
    except SyntaxError:
        return None
    if len(tree.body) != 1:
        return None
    node = tree.body[0]
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return None
    return [argument.arg for argument in node.args.posonlyargs + node.args.args]


def _signature_matches(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    expected_signature: str,
) -> bool:
    expected = _signature_arguments(expected_signature)
    if expected is None:
        return False
    actual = [argument.arg for argument in function.args.posonlyargs + function.args.args]
    return actual == expected


class CodeTestsEvaluator:
    """Run authored unit tests against extracted generated code."""

    name = "code_tests"
    version = "1.0.0"
    evaluation_type = EvaluationType.CODE_TESTS

    def __init__(self, sandbox: CodeSandbox | None = None) -> None:
        self.sandbox = sandbox or CodeSandbox()

    def evaluate(self, item: DatasetItem, response_text: str) -> DeterministicResult:
        if not item.function_name:
            raise EvaluationConfigurationError("code_tests item has no function_name")
        if not item.test_cases:
            raise EvaluationConfigurationError("code_tests item has no test cases")

        code = extract_code_block(response_text)
        checks = [
            make_check(
                "code_extracted",
                bool(code),
                detail="non-empty code extracted" if code else "no code found",
            )
        ]
        if not code:
            return build_result(
                item,
                evaluator=self.name,
                evaluation_type=self.evaluation_type,
                score=0.0,
                checks=checks,
                passed=False,
                version=self.version,
            )

        try:
            function = _find_function(code, item.function_name)
            defined = function is not None
        except SyntaxError as exc:
            checks.append(
                make_check(
                    "function_defined",
                    False,
                    detail=f"code is not valid Python: {exc.msg}",
                )
            )
            return build_result(
                item,
                evaluator=self.name,
                evaluation_type=self.evaluation_type,
                score=0.0,
                checks=checks,
                passed=False,
                version=self.version,
            )

        checks.append(
            make_check(
                "function_defined",
                defined,
                detail=(
                    f"top-level function {item.function_name!r} found"
                    if defined
                    else f"top-level function {item.function_name!r} not found"
                ),
                metadata={"function_signature": item.function_signature},
            )
        )
        if not defined:
            return build_result(
                item,
                evaluator=self.name,
                evaluation_type=self.evaluation_type,
                score=0.0,
                checks=checks,
                passed=False,
                version=self.version,
            )

        signature_ok = True
        if item.function_signature:
            signature_ok = function is not None and _signature_matches(
                function, item.function_signature
            )
            checks.append(
                make_check(
                    "function_signature",
                    signature_ok,
                    detail=(
                        "function signature matches the required signature"
                        if signature_ok
                        else "function arguments do not match the required signature"
                    ),
                    metadata={"expected": item.function_signature},
                )
            )
        if not signature_ok:
            return build_result(
                item,
                evaluator=self.name,
                evaluation_type=self.evaluation_type,
                score=0.0,
                checks=checks,
                passed=False,
                version=self.version,
            )

        outcomes = self.sandbox.run_tests(code, item.function_name, item.test_cases)
        for outcome in outcomes:
            checks.append(
                make_check(
                    f"test_{outcome.name}",
                    outcome.passed,
                    detail=(
                        "test passed"
                        if outcome.passed
                        else (outcome.error or "test returned an incorrect value")
                    ),
                    metadata={
                        "timed_out": outcome.timed_out,
                        "exit_code": outcome.exit_code,
                        "duration_seconds": outcome.duration_seconds,
                        "stdout": outcome.stdout,
                        "stderr": outcome.stderr,
                        "output_truncated": outcome.output_truncated,
                    },
                )
            )

        passed_count = sum(1 for outcome in outcomes if outcome.passed)
        passed = passed_count == len(outcomes) and len(outcomes) > 0
        score = item.max_score * passed_count / len(outcomes) if outcomes else 0.0
        return build_result(
            item,
            evaluator=self.name,
            evaluation_type=self.evaluation_type,
            score=score,
            checks=checks,
            passed=passed,
            version=self.version,
        )
