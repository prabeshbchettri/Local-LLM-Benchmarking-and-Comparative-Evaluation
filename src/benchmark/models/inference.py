"""
File: inference.py

Description:
Typed request, timing, token, response, and run-summary schemas for inference.

Responsibilities:
- Preserve raw prompt and response text without whitespace normalization.
- Represent successful, empty, timed-out, failed, and cancelled responses.
- Store client timing separately from Ollama-reported durations.
- Provide stable records for the interim JSONL response log.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

RESPONSE_SCHEMA_VERSION = "1.0.0"


class InferenceSchema(BaseModel):
    """Base schema that preserves raw prompt and response whitespace."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=False,
    )


class ResponseStatus(StrEnum):
    """Final status of one inference request."""

    SUCCESS = "success"
    EMPTY = "empty"
    TIMEOUT = "timeout"
    ERROR = "error"
    CANCELLED = "cancelled"


class RunStatus(StrEnum):
    """Final status of an inference batch."""

    COMPLETED = "completed"
    CANCELLED = "cancelled"


class InferenceProfile(StrEnum):
    """Purpose of a group of inference requests."""

    QUALITY = "quality"
    PERFORMANCE = "performance"
    CONSISTENCY = "consistency"
    COLD_START = "cold_start"
    SMOKE = "smoke"


class PromptSpec(InferenceSchema):
    """One prompt and the output limit that should be applied to it."""

    item_id: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    system_instruction: str | None = None
    maximum_output_tokens: int = Field(default=512, ge=1, le=4096)
    category: str | None = None


class GenerationOptions(InferenceSchema):
    """Fixed generation options sent to Ollama."""

    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    top_k: int = Field(default=40, ge=1)
    seed: int = 42
    repeat_penalty: float = Field(default=1.1, gt=0.0)
    num_ctx: int = Field(default=4096, ge=512)
    num_predict: int = Field(default=512, ge=1)

    def to_payload(self) -> dict[str, Any]:
        """Return the options mapping expected by the Ollama API."""
        return self.model_dump(mode="json")


class TimingMetrics(InferenceSchema):
    """Client and Ollama timing measurements for one request."""

    ttft_seconds: float | None = Field(default=None, ge=0.0)
    total_latency_seconds: float = Field(ge=0.0)
    load_duration_seconds: float | None = Field(default=None, ge=0.0)
    prompt_eval_duration_seconds: float | None = Field(default=None, ge=0.0)
    eval_duration_seconds: float | None = Field(default=None, ge=0.0)


class TokenMetrics(InferenceSchema):
    """Token counts and rates derived from Ollama runtime metadata."""

    prompt_eval_count: int | None = Field(default=None, ge=0)
    eval_count: int | None = Field(default=None, ge=0)
    output_tokens_per_second: float | None = Field(default=None, ge=0.0)
    prompt_tokens_per_second: float | None = Field(default=None, ge=0.0)


class ResponseRecord(InferenceSchema):
    """Structured record for one inference request."""

    schema_version: str = RESPONSE_SCHEMA_VERSION
    experiment_id: str = Field(min_length=1)
    response_id: str = Field(default_factory=lambda: uuid4().hex)
    model_name: str | None = None
    model_tag: str = Field(min_length=1)
    model_digest: str | None = None
    item_id: str | None = None
    category: str | None = None
    profile: InferenceProfile
    repetition: int = Field(ge=1)
    is_warmup: bool = False
    cold_start: bool = False
    prompt: str
    system_prompt: str | None = None
    response_text: str = ""
    finish_reason: str | None = None
    status: ResponseStatus
    error_type: str | None = None
    error_message: str | None = None
    started_at: datetime
    completed_at: datetime
    generation_options: GenerationOptions
    timing: TimingMetrics
    tokens: TokenMetrics
    ollama_metadata: dict[str, Any] = Field(default_factory=dict)
    stream_chunk_count: int = Field(default=0, ge=0)
    config_hash: str | None = None

    @field_validator("started_at", "completed_at")
    @classmethod
    def validate_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("response timestamps must include a timezone")
        return value


class RunSummary(InferenceSchema):
    """Summary of one inference batch, excluding warm-up from measured counts."""

    experiment_id: str = Field(min_length=1)
    profile: InferenceProfile
    status: RunStatus
    started_at: datetime
    completed_at: datetime
    model_tags: list[str] = Field(min_length=1)
    question_count: int = Field(ge=1)
    repetitions: int = Field(ge=1)
    warmup_requests: int = Field(ge=0)
    cold_start: bool = False
    total_requests: int = Field(ge=0)
    measured_requests: int = Field(ge=0)
    warmup_count: int = Field(ge=0)
    success_count: int = Field(ge=0)
    empty_count: int = Field(ge=0)
    timeout_count: int = Field(ge=0)
    error_count: int = Field(ge=0)
    cancelled_count: int = Field(ge=0)
    output_path: str | None = None

    @field_validator("started_at", "completed_at")
    @classmethod
    def validate_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("run timestamps must include a timezone")
        return value
