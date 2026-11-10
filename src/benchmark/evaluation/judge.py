"""
File: judge.py

Description:
Blind local LLM judge client, verification, strict parser, and one retry.

Responsibilities:
- Verify the configured judge tag, digest, and quantization before scoring.
- Send a blind fixed-rubric prompt with untrusted candidate data delimited as JSON.
- Parse strict JSON scores and rationales.
- Retry invalid output once, and record failures without substituting zero.
"""

from __future__ import annotations

import json
from typing import Protocol

from benchmark.core.config import EvaluationSettings, ModelSpec
from benchmark.core.exceptions import (
    JudgeParseError,
    OllamaConnectionError,
    OllamaError,
    OllamaHTTPError,
    OllamaTimeoutError,
)
from benchmark.evaluation.models import (
    JudgeConfig,
    JudgeRawOutput,
    JudgeRequest,
    JudgeResult,
    JudgeStatus,
    JudgeVerification,
)
from benchmark.evaluation.parsing import JSON_FENCE_PATTERN
from benchmark.evaluation.rubric import (
    JUDGE_DIMENSIONS,
    JUDGE_PROMPT_VERSION,
    RUBRIC_VERSION,
    build_judge_messages,
)
from benchmark.execution.ollama_client import OllamaClient


class Judge(Protocol):
    """Protocol for a blind subjective judge."""

    config: JudgeConfig

    def judge(self, request: JudgeRequest, **kwargs: object) -> JudgeResult:
        """Judge one candidate response."""


def _normalized_digest(value: str | None) -> str | None:
    if value is None:
        return None
    return value.lower().removeprefix("sha256:")


def parse_judge_output(raw_output: str) -> JudgeRawOutput:
    """Parse strict judge JSON and validate all scores and rationales."""
    stripped = raw_output.strip()
    match = JSON_FENCE_PATTERN.fullmatch(stripped)
    candidate = match.group(1).strip() if match is not None else stripped
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise JudgeParseError(f"judge output is not valid JSON: {exc.msg}") from exc
    try:
        return JudgeRawOutput.model_validate(payload)
    except Exception as exc:
        raise JudgeParseError(f"judge output failed schema validation: {exc}") from exc


class OllamaJudge:
    """Judge candidate responses with the configured local Ollama model."""

    def __init__(
        self,
        client: OllamaClient,
        model: ModelSpec,
        evaluation: EvaluationSettings,
        *,
        keep_alive: str = "30m",
        verify_model: bool = True,
    ) -> None:
        self.client = client
        self.model = model
        self.evaluation = evaluation
        self.keep_alive = keep_alive
        self.verify_model = verify_model
        self.config = JudgeConfig(
            name=model.name,
            tag=model.tag,
            digest=None,
            family=None,
            rubric_version=RUBRIC_VERSION,
            prompt_version=JUDGE_PROMPT_VERSION,
            temperature=evaluation.judge_temperature,
            seed=evaluation.judge_seed,
            top_p=1.0,
            top_k=40,
            repeat_penalty=1.1,
            num_ctx=evaluation.judge_context_length,
            num_predict=evaluation.judge_max_output_tokens,
        )
        self.verification: JudgeVerification | None = None
        self._verified = False

    def verify(self) -> JudgeVerification:
        """Verify the installed judge artifact against configured identity."""
        model = self.client.find_model(self.model.tag)
        if model is None:
            verification = JudgeVerification(
                tag=self.model.tag,
                installed=False,
                expected_digest=_normalized_digest(self.model.expected_digest),
                expected_quantization=self.model.expected_quantization,
                problems=["judge model is not installed"],
            )
            self.verification = verification
            self._verified = False
            return verification

        digest = _normalized_digest(model.digest)
        expected_digest = _normalized_digest(self.model.expected_digest)
        quantization = model.details.quantization_level or None
        expected_quantization = self.model.expected_quantization
        problems: list[str] = []
        digest_match = None
        if expected_digest is not None:
            digest_match = digest == expected_digest
            if not digest_match:
                problems.append("judge digest does not match configuration")
        quantization_match = None
        if expected_quantization is not None:
            if quantization is None:
                problems.append("judge quantization is unavailable")
            else:
                quantization_match = quantization.upper() == expected_quantization.upper()
                if not quantization_match:
                    problems.append("judge quantization does not match configuration")

        verification = JudgeVerification(
            tag=self.model.tag,
            installed=True,
            digest=digest,
            expected_digest=expected_digest,
            digest_match=digest_match,
            quantization=quantization,
            expected_quantization=expected_quantization,
            quantization_match=quantization_match,
            problems=problems,
        )
        self.verification = verification
        self._verified = verification.ok
        if verification.ok:
            self.config = self.config.model_copy(
                update={
                    "digest": digest,
                    "family": model.details.family or None,
                }
            )
        return verification

    def _result_for_error(
        self,
        status: JudgeStatus,
        error: str,
        *,
        config: JudgeConfig,
        attempts: int,
        raw_output: str | None = None,
    ) -> JudgeResult:
        return JudgeResult(
            status=status,
            config=config,
            verification=self.verification,
            attempts=attempts,
            retry_count=max(0, attempts - 1),
            raw_output=raw_output,
            error=error,
        )

    def judge(
        self,
        request: JudgeRequest,
        *,
        temperature: float | None = None,
        seed: int | None = None,
        reliability: bool = False,
    ) -> JudgeResult:
        """Judge one response, retrying malformed output once."""
        if self.verify_model and not self._verified:
            verification = self.verify()
            if not verification.ok:
                return self._result_for_error(
                    JudgeStatus.VERIFICATION_ERROR,
                    "; ".join(verification.problems) or "judge verification failed",
                    config=self.config,
                    attempts=0,
                )

        actual_temperature = (
            self.config.temperature if temperature is None else temperature
        )
        actual_seed = self.config.seed if seed is None else seed
        call_config = self.config.model_copy(
            update={"temperature": actual_temperature, "seed": actual_seed}
        )
        options = {
            "temperature": call_config.temperature,
            "seed": call_config.seed,
            "top_p": call_config.top_p,
            "top_k": call_config.top_k,
            "repeat_penalty": call_config.repeat_penalty,
            "num_ctx": call_config.num_ctx,
            "num_predict": call_config.num_predict,
        }

        last_raw_output: str | None = None
        for attempt in (1, 2):
            messages = build_judge_messages(request, stricter=attempt == 2)
            try:
                response = self.client.chat(
                    self.model.tag,
                    messages,
                    options=options,
                    keep_alive=self.keep_alive,
                )
            except OllamaTimeoutError as exc:
                return self._result_for_error(
                    JudgeStatus.TIMEOUT,
                    str(exc),
                    config=call_config,
                    attempts=attempt,
                    raw_output=last_raw_output,
                )
            except OllamaConnectionError as exc:
                return self._result_for_error(
                    JudgeStatus.CONNECTION_ERROR,
                    str(exc),
                    config=call_config,
                    attempts=attempt,
                    raw_output=last_raw_output,
                )
            except OllamaHTTPError as exc:
                return self._result_for_error(
                    JudgeStatus.HTTP_ERROR,
                    str(exc),
                    config=call_config,
                    attempts=attempt,
                    raw_output=last_raw_output,
                )
            except OllamaError as exc:
                return self._result_for_error(
                    JudgeStatus.ERROR,
                    str(exc),
                    config=call_config,
                    attempts=attempt,
                    raw_output=last_raw_output,
                )

            last_raw_output = response.message.content
            try:
                parsed = parse_judge_output(last_raw_output)
            except JudgeParseError as exc:
                if attempt == 1:
                    continue
                return self._result_for_error(
                    JudgeStatus.INVALID_OUTPUT,
                    str(exc),
                    config=call_config,
                    attempts=attempt,
                    raw_output=last_raw_output,
                )

            dimensions = {
                name: getattr(parsed, name) for name in JUDGE_DIMENSIONS
            }
            raw_score = sum(item.score for item in dimensions.values()) / len(dimensions)
            normalized = (raw_score - 1.0) / 4.0 * 100.0
            return JudgeResult(
                status=JudgeStatus.SUCCESS,
                config=call_config,
                verification=self.verification,
                dimensions=dimensions,
                raw_score=raw_score,
                normalized_score=normalized,
                attempts=attempt,
                retry_count=attempt - 1,
                raw_output=last_raw_output,
                reliability=reliability,
                reliability_seed=actual_seed if reliability else None,
            )

        return self._result_for_error(
            JudgeStatus.ERROR,
            "judge call ended without a result",
            config=call_config,
            attempts=2,
            raw_output=last_raw_output,
        )


def build_ollama_judge(
    client: OllamaClient,
    settings: EvaluationSettings,
    model: ModelSpec,
    *,
    keep_alive: str = "30m",
    verify_model: bool = True,
) -> OllamaJudge:
    """Build the configured primary judge."""
    return OllamaJudge(
        client,
        model,
        settings,
        keep_alive=keep_alive,
        verify_model=verify_model,
    )
