"""
File: sandbox.py

Description:
Best-effort restricted subprocess runner for authored coding tests.

Responsibilities:
- Run each authored test in a fresh Python isolated-mode subprocess.
- Use a temporary working directory, minimal environment, and no shell.
- Enforce per-test and total timeouts plus captured-output limits.
- Return structured outcomes without executing generated code in-process.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from pydantic import Field

from benchmark.dataset.models import CodeTestCase
from benchmark.models.base import BaseSchema
from benchmark.evaluation.parsing import truncate_output

DEFAULT_PER_TEST_TIMEOUT_SECONDS = 5.0
DEFAULT_TOTAL_TIMEOUT_SECONDS = 20.0
DEFAULT_OUTPUT_LIMIT = 10_000


class SandboxTestOutcome(BaseSchema):
    """Result of running one authored test in the restricted runner."""

    name: str = Field(min_length=1)
    passed: bool
    timed_out: bool = False
    exit_code: int | None = None
    duration_seconds: float = Field(ge=0.0)
    stdout: str = ""
    stderr: str = ""
    output_truncated: bool = False
    error: str | None = None


HARNESS_SOURCE = r'''
import importlib.util
import json
import math
import sys
import traceback
from pathlib import Path


def values_equal(actual, expected):
    if isinstance(expected, bool) or isinstance(actual, bool):
        return type(actual) is type(expected) and actual == expected
    if isinstance(expected, float) or isinstance(actual, float):
        try:
            return math.isclose(float(actual), float(expected), rel_tol=1e-9, abs_tol=1e-12)
        except (TypeError, ValueError):
            return False
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            return False
        return all(values_equal(a, e) for a, e in zip(actual, expected))
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            return False
        return all(values_equal(actual[key], expected[key]) for key in expected)
    return actual == expected


def write_result(path, payload):
    Path(path).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def main():
    solution_path, test_path, result_path, function_name = sys.argv[1:5]
    payload = {"passed": False, "actual_repr": None, "error": None}
    try:
        test = json.loads(Path(test_path).read_text(encoding="utf-8"))
        spec = importlib.util.spec_from_file_location("candidate_solution", solution_path)
        if spec is None or spec.loader is None:
            raise RuntimeError("could not load candidate module")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        function = getattr(module, function_name, None)
        if not callable(function):
            raise AttributeError(f"callable {function_name!r} not found")
        actual = function(*test.get("args", []), **test.get("kwargs", {}))
        payload["actual_repr"] = repr(actual)[:2000]
        payload["passed"] = bool(values_equal(actual, test.get("expected")))
    except BaseException as exc:
        payload["error"] = f"{type(exc).__name__}: {exc}"
        payload["traceback"] = traceback.format_exc(limit=5)[-4000:]
    finally:
        try:
            write_result(result_path, payload)
        except BaseException:
            pass


if __name__ == "__main__":
    main()
'''


class CodeSandbox:
    """Run authored code tests in fresh, isolated subprocesses."""

    def __init__(
        self,
        *,
        python_executable: str | None = None,
        per_test_timeout_seconds: float = DEFAULT_PER_TEST_TIMEOUT_SECONDS,
        total_timeout_seconds: float = DEFAULT_TOTAL_TIMEOUT_SECONDS,
        output_limit: int = DEFAULT_OUTPUT_LIMIT,
    ) -> None:
        if per_test_timeout_seconds <= 0:
            raise ValueError("per_test_timeout_seconds must be positive")
        if total_timeout_seconds <= 0:
            raise ValueError("total_timeout_seconds must be positive")
        if output_limit < 1:
            raise ValueError("output_limit must be positive")
        self.python_executable = python_executable or sys.executable
        self.per_test_timeout_seconds = per_test_timeout_seconds
        self.total_timeout_seconds = total_timeout_seconds
        self.output_limit = output_limit

    def _environment(self, workdir: Path) -> dict[str, str]:
        environment = {
            "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),
            "WINDIR": os.environ.get("WINDIR", ""),
            "TEMP": str(workdir),
            "TMP": str(workdir),
        }
        return {key: value for key, value in environment.items() if value}

    def _outcome_from_result(
        self,
        case: CodeTestCase,
        *,
        duration: float,
        completed: subprocess.CompletedProcess[str] | None,
        result: dict[str, object] | None,
        timed_out: bool,
    ) -> SandboxTestOutcome:
        stdout = completed.stdout if completed is not None else ""
        stderr = completed.stderr if completed is not None else ""
        output_truncated = len(stdout) > self.output_limit or len(stderr) > self.output_limit
        passed = not timed_out and bool(result and result.get("passed") is True)
        error = None
        if timed_out:
            error = "test timed out"
        elif result is None:
            error = "runner did not produce a result"
            if completed is not None and completed.returncode != 0:
                error = f"runner exited with code {completed.returncode}"
        elif result.get("error"):
            error = str(result["error"])
        return SandboxTestOutcome(
            name=case.name,
            passed=passed,
            timed_out=timed_out,
            exit_code=completed.returncode if completed is not None else None,
            duration_seconds=duration,
            stdout=truncate_output(stdout, self.output_limit),
            stderr=truncate_output(stderr, self.output_limit),
            output_truncated=output_truncated,
            error=error,
        )

    def run_tests(
        self,
        code: str,
        function_name: str,
        test_cases: list[CodeTestCase],
    ) -> list[SandboxTestOutcome]:
        """Run all test cases and return one outcome per case."""
        outcomes: list[SandboxTestOutcome] = []
        deadline = time.monotonic() + self.total_timeout_seconds
        with tempfile.TemporaryDirectory(prefix="benchmark-code-") as directory:
            workdir = Path(directory)
            solution_path = workdir / "solution.py"
            harness_path = workdir / "harness.py"
            solution_path.write_text(code, encoding="utf-8")
            harness_path.write_text(HARNESS_SOURCE, encoding="utf-8")

            for index, case in enumerate(test_cases):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    outcomes.append(
                        SandboxTestOutcome(
                            name=case.name,
                            passed=False,
                            timed_out=True,
                            duration_seconds=0.0,
                            error="total timeout exceeded before test started",
                        )
                    )
                    continue

                test_path = workdir / f"test_{index}.json"
                result_path = workdir / f"result_{index}.json"
                test_path.write_text(
                    json.dumps(
                        {
                            "args": case.args,
                            "kwargs": case.kwargs,
                            "expected": case.expected,
                        },
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
                result_path.unlink(missing_ok=True)
                timeout = min(self.per_test_timeout_seconds, remaining)
                start = time.monotonic()
                completed: subprocess.CompletedProcess[str] | None = None
                timed_out = False
                try:
                    completed = subprocess.run(
                        [
                            self.python_executable,
                            "-I",
                            str(harness_path),
                            str(solution_path),
                            str(test_path),
                            str(result_path),
                            function_name,
                        ],
                        cwd=workdir,
                        env=self._environment(workdir),
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        timeout=timeout,
                        check=False,
                    )
                except subprocess.TimeoutExpired as exc:
                    timed_out = True
                    completed = subprocess.CompletedProcess(
                        args=exc.cmd,
                        returncode=None,
                        stdout=exc.stdout or "",
                        stderr=exc.stderr or "",
                    )
                duration = time.monotonic() - start
                result: dict[str, object] | None = None
                if result_path.is_file():
                    try:
                        loaded = json.loads(result_path.read_text(encoding="utf-8"))
                        if isinstance(loaded, dict):
                            result = loaded
                    except (OSError, json.JSONDecodeError):
                        result = None
                outcomes.append(
                    self._outcome_from_result(
                        case,
                        duration=duration,
                        completed=completed,
                        result=result,
                        timed_out=timed_out,
                    )
                )
        return outcomes
