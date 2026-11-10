"""
File: engine.py

Description:
Streaming inference engine, cancellation control, batch execution, and JSONL logging.

Responsibilities:
- Build fixed Ollama chat requests and stream responses.
- Measure TTFT and total latency with a monotonic clock.
- Capture Ollama token counts, durations, and final raw metadata.
- Record empty, timed-out, failed, and cancelled requests without retrying.
- Execute selected models and prompts sequentially with warm-up and cold-start controls.
- Append response records to an interim JSONL log until Phase 07 storage exists.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmark.core.config import AppSettings
from benchmark.core.exceptions import (
    EmptyResponseError,
    InferenceCancelledError,
    InferenceError,
    InferenceTimeoutError,
    ModelNotFoundError,
    OllamaConnectionError,
    OllamaError,
    OllamaHTTPError,
    OllamaResponseError,
    OllamaTimeoutError,
    StreamParseError,
)
from benchmark.core.logging import get_logger
from benchmark.execution.ollama_client import OllamaClient
from benchmark.execution.registry import ModelManager
from benchmark.execution.timing import (
    elapsed_seconds,
    nanoseconds_to_seconds,
    tokens_per_second,
    ttft_seconds,
)
from benchmark.models.hardware import HardwareSummary
from benchmark.models.inference import (
    GenerationOptions,
    InferenceProfile,
    PromptSpec,
    ResponseRecord,
    ResponseStatus,
    RunStatus,
    RunSummary,
    TimingMetrics,
    TokenMetrics,
)
from benchmark.models.ollama import OllamaChatChunk
from benchmark.monitoring.sampler import HardwareSampler

LOGGER = get_logger("execution.engine")


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


class CancellationToken:
    """Thread-safe cooperative cancellation flag."""

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        """Request cancellation."""
        self._event.set()

    def reset(self) -> None:
        """Clear a previous cancellation request."""
        self._event.clear()

    @property
    def is_cancelled(self) -> bool:
        """Return whether cancellation has been requested."""
        return self._event.is_set()


class JsonlResponseWriter:
    """Append response records to a durable newline-delimited JSON file."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path).expanduser().resolve()

    def append(self, record: ResponseRecord) -> None:
        """Append one record and flush it to disk."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(
            record.model_dump(mode="json"),
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        try:
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(line + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        except OSError as exc:
            raise InferenceError(f"Could not write response log {self.path}: {exc}") from exc


class InferenceEngine:
    """Execute one streaming chat request and return a structured record."""

    def __init__(
        self,
        settings: AppSettings,
        client: OllamaClient,
        *,
        clock_ns: Callable[[], int] = time.perf_counter_ns,
        model_identity: Callable[[str], tuple[str, str | None]] | None = None,
        hardware_factory: Callable[[], HardwareSampler | None] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.settings = settings
        self.client = client
        self._clock_ns = clock_ns
        self._model_identity = model_identity or self._resolve_model_identity
        self._identity_cache: dict[str, tuple[str, str | None]] = {}
        self._interrupted = False
        self._sleep = sleep
        self._hardware_factory = (
            hardware_factory
            if hardware_factory is not None
            else self._default_hardware_factory
        )

    def _default_hardware_factory(self) -> HardwareSampler | None:
        benchmark = self.settings.benchmark
        if not benchmark.hardware_monitoring_enabled:
            return None
        return HardwareSampler(
            interval_seconds=benchmark.hardware_sample_interval_ms / 1000,
            baseline_seconds=benchmark.hardware_baseline_seconds,
        )

    def _attach_model_vram(
        self,
        hardware: HardwareSummary,
        model_tag: str,
    ) -> None:
        loader = getattr(self.client, "loaded_models", None)
        if not callable(loader):
            return
        try:
            loaded = loader()
        except Exception as exc:
            hardware.ollama_model_vram_error = f"Could not read /api/ps: {exc}"
            return
        match = next(
            (
                model
                for model in loaded
                if model_tag in {model.name, model.model, model.tag}
            ),
            None,
        )
        if match is None:
            hardware.ollama_model_vram_error = (
                f"Model is not listed by /api/ps: {model_tag}"
            )
            return
        hardware.ollama_model_name = match.tag
        hardware.ollama_model_size_bytes = match.size
        hardware.ollama_model_size_vram_bytes = match.size_vram
        hardware.ollama_model_vram_ratio = (
            match.size_vram / match.size if match.size > 0 else None
        )
        hardware.ollama_model_vram_available = True

    @property
    def interrupted(self) -> bool:
        """Return whether a keyboard interruption was captured."""
        return self._interrupted

    def _configured_name(self, tag: str) -> str | None:
        for spec in [*self.settings.models.candidates, self.settings.models.judge]:
            if spec.tag == tag:
                return spec.name
        return None

    def _resolve_model_identity(self, tag: str) -> tuple[str, str | None]:
        cached = self._identity_cache.get(tag)
        if cached is not None:
            return cached
        summary = self.client.find_model(tag)
        if summary is None:
            raise ModelNotFoundError(f"Model is not installed: {tag}")
        identity = (self._configured_name(tag) or summary.name, summary.digest or None)
        self._identity_cache[tag] = identity
        return identity

    def _options(self, max_tokens: int | None) -> GenerationOptions:
        benchmark = self.settings.benchmark
        limit = benchmark.num_predict
        if max_tokens is not None:
            limit = min(max_tokens, benchmark.num_predict)
        return GenerationOptions(
            temperature=benchmark.temperature,
            top_p=benchmark.top_p,
            top_k=benchmark.top_k,
            seed=benchmark.random_seed,
            repeat_penalty=benchmark.repeat_penalty,
            num_ctx=benchmark.num_ctx,
            num_predict=limit,
        )

    def _record(
        self,
        *,
        experiment_id: str,
        model_tag: str,
        model_name: str | None,
        model_digest: str | None,
        item_id: str | None,
        category: str | None,
        profile: InferenceProfile,
        repetition: int,
        is_warmup: bool,
        cold_start: bool,
        prompt: str,
        system_prompt: str | None,
        response_text: str,
        finish_reason: str | None,
        status: ResponseStatus,
        error_type: str | None,
        error_message: str | None,
        started_at: datetime,
        completed_at: datetime,
        options: GenerationOptions,
        start_ns: int,
        end_ns: int,
        first_token_ns: int | None,
        final_chunk: OllamaChatChunk | None,
        metadata_chunk: OllamaChatChunk | None,
        chunk_count: int,
        hardware: HardwareSummary | None = None,
        hardware_error: str | None = None,
    ) -> ResponseRecord:
        final = final_chunk
        timing = TimingMetrics(
            ttft_seconds=ttft_seconds(start_ns, first_token_ns),
            total_latency_seconds=elapsed_seconds(start_ns, end_ns),
            load_duration_seconds=nanoseconds_to_seconds(
                final.load_duration if final else None
            ),
            prompt_eval_duration_seconds=nanoseconds_to_seconds(
                final.prompt_eval_duration if final else None
            ),
            eval_duration_seconds=nanoseconds_to_seconds(
                final.eval_duration if final else None
            ),
        )
        tokens = TokenMetrics(
            prompt_eval_count=final.prompt_eval_count if final else None,
            eval_count=final.eval_count if final else None,
            output_tokens_per_second=tokens_per_second(
                final.eval_count if final else None,
                final.eval_duration if final else None,
            ),
            prompt_tokens_per_second=tokens_per_second(
                final.prompt_eval_count if final else None,
                final.prompt_eval_duration if final else None,
            ),
        )
        metadata_source = final or metadata_chunk
        return ResponseRecord(
            experiment_id=experiment_id,
            model_name=model_name,
            model_tag=model_tag,
            model_digest=model_digest,
            item_id=item_id,
            category=category,
            profile=profile,
            repetition=repetition,
            is_warmup=is_warmup,
            cold_start=cold_start,
            prompt=prompt,
            system_prompt=system_prompt,
            response_text=response_text,
            finish_reason=finish_reason,
            status=status,
            error_type=error_type,
            error_message=error_message,
            started_at=started_at,
            completed_at=completed_at,
            generation_options=options,
            timing=timing,
            tokens=tokens,
            ollama_metadata=metadata_source.raw if metadata_source else {},
            stream_chunk_count=chunk_count,
            hardware=hardware,
            hardware_error=hardware_error,
            config_hash=self.settings.config_hash,
        )

    def run_request(
        self,
        model_tag: str,
        prompt: str,
        *,
        experiment_id: str,
        item_id: str | None = None,
        category: str | None = None,
        repetition: int = 1,
        profile: InferenceProfile | str = InferenceProfile.SMOKE,
        system_instruction: str | None = None,
        max_tokens: int | None = None,
        is_warmup: bool = False,
        cold_start: bool = False,
        keep_alive: str | int | None = None,
        cancellation: CancellationToken | None = None,
    ) -> ResponseRecord:
        """Run one request, recording failures instead of retrying them."""
        if repetition < 1:
            raise ValueError("repetition must be at least 1")
        profile_value = InferenceProfile(profile)
        options = self._options(max_tokens)
        model_name = self._configured_name(model_tag)

        if cancellation is not None and cancellation.is_cancelled:
            started_at = utc_now()
            start_ns = self._clock_ns()
            return self._record(
                experiment_id=experiment_id,
                model_tag=model_tag,
                model_name=model_name,
                model_digest=None,
                item_id=item_id,
                category=category,
                profile=profile_value,
                repetition=repetition,
                is_warmup=is_warmup,
                cold_start=cold_start,
                prompt=prompt,
                system_prompt=system_instruction,
                response_text="",
                finish_reason=None,
                status=ResponseStatus.CANCELLED,
                error_type=InferenceCancelledError.__name__,
                error_message="Cancellation requested before the request started",
                started_at=started_at,
                completed_at=utc_now(),
                options=options,
                start_ns=start_ns,
                end_ns=start_ns,
                first_token_ns=None,
                final_chunk=None,
                metadata_chunk=None,
                chunk_count=0,
            )

        model_name, model_digest = self._model_identity(model_tag)
        messages: list[dict[str, str]] = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})
        effective_keep_alive = (
            keep_alive
            if keep_alive is not None
            else 0
            if cold_start
            else self.settings.models.keep_alive
        )

        monitor: HardwareSampler | None = None
        hardware: HardwareSummary | None = None
        hardware_error: str | None = None
        if self._hardware_factory is not None:
            try:
                monitor = self._hardware_factory()
                if monitor is not None:
                    monitor.start()
                    baseline = getattr(
                        monitor,
                        "baseline_seconds",
                        self.settings.benchmark.hardware_baseline_seconds,
                    )
                    if baseline > 0:
                        self._sleep(baseline)
                    monitor.mark_request_start()
            except BaseException as exc:
                if monitor is not None:
                    try:
                        monitor.stop()
                    except Exception:
                        pass
                monitor = None
                if isinstance(exc, Exception):
                    hardware_error = f"Hardware monitoring failed: {exc}"
                else:
                    raise

        parts: list[str] = []
        first_token_ns: int | None = None
        final_chunk: OllamaChatChunk | None = None
        metadata_chunk: OllamaChatChunk | None = None
        end_ns: int | None = None
        chunk_count = 0
        status = ResponseStatus.SUCCESS
        error_type: str | None = None
        error_message: str | None = None
        response_text = ""
        finish_reason: str | None = None
        started_at = utc_now()
        start_ns = self._clock_ns()

        try:
            for chunk in self.client.stream_chat(
                model_tag,
                messages,
                options=options.to_payload(),
                keep_alive=effective_keep_alive,
            ):
                now_ns = self._clock_ns()
                chunk_count += 1
                metadata_chunk = chunk
                content = chunk.message.content if chunk.message is not None else ""
                if content:
                    if first_token_ns is None:
                        first_token_ns = now_ns
                    parts.append(content)
                if chunk.done:
                    final_chunk = chunk
                    end_ns = now_ns
                    break
                if cancellation is not None and cancellation.is_cancelled:
                    raise InferenceCancelledError("Cancellation requested during streaming")
            if final_chunk is None:
                raise StreamParseError("Chat stream ended without a final done chunk")
            response_text = "".join(parts)
            finish_reason = final_chunk.done_reason
            if not response_text:
                raise EmptyResponseError("Model returned an empty response")
        except KeyboardInterrupt:
            self._interrupted = True
            status = ResponseStatus.CANCELLED
            error_type = InferenceCancelledError.__name__
            error_message = "Interrupted by user"
        except InferenceCancelledError as exc:
            status = ResponseStatus.CANCELLED
            error_type = type(exc).__name__
            error_message = str(exc)
        except OllamaTimeoutError as exc:
            status = ResponseStatus.TIMEOUT
            error_type = InferenceTimeoutError.__name__
            error_message = str(exc)
        except OllamaConnectionError as exc:
            status = ResponseStatus.ERROR
            error_type = type(exc).__name__
            error_message = str(exc)
        except OllamaHTTPError as exc:
            status = ResponseStatus.ERROR
            error_type = type(exc).__name__
            error_message = str(exc)
        except OllamaResponseError as exc:
            status = ResponseStatus.ERROR
            error_type = type(exc).__name__
            error_message = str(exc)
        except EmptyResponseError as exc:
            status = ResponseStatus.EMPTY
            error_type = type(exc).__name__
            error_message = str(exc)
            response_text = "".join(parts)
        except StreamParseError as exc:
            status = ResponseStatus.ERROR
            error_type = type(exc).__name__
            error_message = str(exc)
            response_text = "".join(parts)
        except OllamaError as exc:
            status = ResponseStatus.ERROR
            error_type = type(exc).__name__
            error_message = str(exc)
            response_text = "".join(parts)
        finally:
            if monitor is not None:
                try:
                    hardware = monitor.stop()
                except Exception as exc:
                    hardware_error = f"Hardware monitoring stop failed: {exc}"
                else:
                    try:
                        self._attach_model_vram(hardware, model_tag)
                    except Exception as exc:
                        hardware.ollama_model_vram_error = (
                            f"Could not attach /api/ps data: {exc}"
                        )

        if not response_text:
            response_text = "".join(parts)
        if end_ns is None:
            end_ns = self._clock_ns()
        return self._record(
            experiment_id=experiment_id,
            model_tag=model_tag,
            model_name=model_name,
            model_digest=model_digest,
            item_id=item_id,
            category=category,
            profile=profile_value,
            repetition=repetition,
            is_warmup=is_warmup,
            cold_start=cold_start,
            prompt=prompt,
            system_prompt=system_instruction,
            response_text=response_text,
            finish_reason=finish_reason,
            status=status,
            error_type=error_type,
            error_message=error_message,
            started_at=started_at,
            completed_at=utc_now(),
            options=options,
            start_ns=start_ns,
            end_ns=end_ns,
            first_token_ns=first_token_ns,
            final_chunk=final_chunk,
            metadata_chunk=metadata_chunk,
            chunk_count=chunk_count,
            hardware=hardware,
            hardware_error=hardware_error,
        )


@dataclass(frozen=True)
class InferenceBatchResult:
    """Records and summary produced by one inference batch."""

    records: tuple[ResponseRecord, ...]
    summary: RunSummary


class InferenceRunner:
    """Execute selected models and prompts sequentially with lifecycle controls."""

    def __init__(
        self,
        engine: InferenceEngine,
        writer: JsonlResponseWriter | None = None,
        *,
        manager: ModelManager | None = None,
        cancellation: CancellationToken | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.engine = engine
        self.writer = writer
        self.manager = manager
        self.cancellation = cancellation or CancellationToken()
        self._sleep = sleep

    def _append(
        self,
        records: list[ResponseRecord],
        record: ResponseRecord,
        progress: Callable[[ResponseRecord], None] | None,
    ) -> None:
        records.append(record)
        if self.writer is not None:
            self.writer.append(record)
        if progress is not None:
            progress(record)

    def run(
        self,
        models: Sequence[str],
        prompts: Sequence[PromptSpec],
        *,
        repetitions: int = 1,
        warmup_requests: int = 0,
        profile: InferenceProfile | str = InferenceProfile.SMOKE,
        experiment_id: str | None = None,
        cold_start: bool = False,
        unload_between_models: bool = True,
        cooldown_seconds: float = 0.0,
        progress: Callable[[ResponseRecord], None] | None = None,
    ) -> InferenceBatchResult:
        """Run all selected models and prompts in deterministic sequential order."""
        if not models:
            raise ValueError("at least one model is required")
        if not prompts:
            raise ValueError("at least one prompt is required")
        if repetitions < 1:
            raise ValueError("repetitions must be at least 1")
        if warmup_requests < 0:
            raise ValueError("warmup_requests must not be negative")
        profile_value = InferenceProfile(profile)
        run_id = experiment_id or (
            f"{utc_now():%Y%m%dT%H%M%SZ}-{os.urandom(4).hex()}"
        )
        started_at = utc_now()
        effective_warmup = 0 if cold_start else warmup_requests
        records: list[ResponseRecord] = []
        cancelled = False

        for model_index, model_tag in enumerate(models):
            if cancelled or self.cancellation.is_cancelled or self.engine.interrupted:
                cancelled = True
                break
            if self.manager is not None and not cold_start:
                self.manager.load(model_tag)
            try:
                for warmup_index in range(effective_warmup):
                    if self.cancellation.is_cancelled or self.engine.interrupted:
                        cancelled = True
                        break
                    prompt = prompts[warmup_index % len(prompts)]
                    record = self.engine.run_request(
                        model_tag,
                        prompt.prompt,
                        experiment_id=run_id,
                        item_id=prompt.item_id,
                        category=prompt.category,
                        repetition=1,
                        profile=profile_value,
                        system_instruction=prompt.system_instruction,
                        max_tokens=prompt.maximum_output_tokens,
                        is_warmup=True,
                        cold_start=False,
                        cancellation=self.cancellation,
                    )
                    self._append(records, record, progress)
                    if record.status is ResponseStatus.CANCELLED or self.engine.interrupted:
                        cancelled = True
                        break
                if not cancelled:
                    for prompt in prompts:
                        if cancelled:
                            break
                        for repetition in range(1, repetitions + 1):
                            if self.cancellation.is_cancelled or self.engine.interrupted:
                                cancelled = True
                                break
                            if self.manager is not None and cold_start:
                                self.manager.unload(model_tag)
                            record = self.engine.run_request(
                                model_tag,
                                prompt.prompt,
                                experiment_id=run_id,
                                item_id=prompt.item_id,
                                category=prompt.category,
                                repetition=repetition,
                                profile=profile_value,
                                system_instruction=prompt.system_instruction,
                                max_tokens=prompt.maximum_output_tokens,
                                is_warmup=False,
                                cold_start=cold_start,
                                cancellation=self.cancellation,
                            )
                            self._append(records, record, progress)
                            if (
                                record.status is ResponseStatus.CANCELLED
                                or self.engine.interrupted
                            ):
                                cancelled = True
                                break
            finally:
                if self.manager is not None and unload_between_models:
                    self.manager.unload(model_tag)
            if cancelled:
                break
            if cooldown_seconds > 0 and model_index < len(models) - 1:
                self._sleep(cooldown_seconds)

        measured = [record for record in records if not record.is_warmup]
        counts = {status: 0 for status in ResponseStatus}
        for record in measured:
            counts[record.status] += 1
        summary = RunSummary(
            experiment_id=run_id,
            profile=profile_value,
            status=RunStatus.CANCELLED if cancelled else RunStatus.COMPLETED,
            started_at=started_at,
            completed_at=utc_now(),
            model_tags=list(models),
            question_count=len(prompts),
            repetitions=repetitions,
            warmup_requests=effective_warmup,
            cold_start=cold_start,
            total_requests=len(records),
            measured_requests=len(measured),
            warmup_count=len(records) - len(measured),
            success_count=counts[ResponseStatus.SUCCESS],
            empty_count=counts[ResponseStatus.EMPTY],
            timeout_count=counts[ResponseStatus.TIMEOUT],
            error_count=counts[ResponseStatus.ERROR],
            cancelled_count=counts[ResponseStatus.CANCELLED],
            output_path=str(self.writer.path) if self.writer is not None else None,
        )
        return InferenceBatchResult(records=tuple(records), summary=summary)
