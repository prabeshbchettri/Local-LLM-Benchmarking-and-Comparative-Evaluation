"""
File: sampler.py

Description:
Background hardware sampling with baseline and inference phases.

Responsibilities:
- Sample CPU, RAM, Ollama process, and GPU metrics on a background thread.
- Provide explicit start, request-start marker, and stop controls.
- Keep samples isolated per sampler instance to prevent request mixing.
- Measure sampler overhead and stop safely on errors.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from typing import Callable

from benchmark.models.hardware import (
    HardwareSample,
    HardwareSummary,
    GpuSource,
    SamplePhase,
)
from benchmark.monitoring.gpu import GpuReading, GpuSampler
from benchmark.monitoring.summaries import build_hardware_summary
from benchmark.monitoring.system import SystemReading, SystemSampler

Clock = Callable[[], float]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class HardwareSampler:
    """Collect hardware samples for one inference request."""

    def __init__(
        self,
        *,
        interval_seconds: float = 1.0,
        baseline_seconds: float = 0.0,
        system_sampler: SystemSampler | None = None,
        gpu_sampler: GpuSampler | None = None,
        clock: Clock = time.perf_counter,
        wall_clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        if baseline_seconds < 0:
            raise ValueError("baseline_seconds must not be negative")
        self.interval_seconds = interval_seconds
        self.baseline_seconds = baseline_seconds
        self._system = system_sampler or SystemSampler()
        self._gpu = gpu_sampler or GpuSampler()
        self._clock = clock
        self._wall_clock = wall_clock
        self._lock = threading.Lock()
        self._samples: list[HardwareSample] = []
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._started = False
        self._started_at: float | None = None
        self._request_started_at: float | None = None
        self._sampler_error: str | None = None
        self._closing = False
        self._gpu_closed = False

    @property
    def is_running(self) -> bool:
        return self._started

    @property
    def samples(self) -> tuple[HardwareSample, ...]:
        with self._lock:
            return tuple(self._samples)

    def _note_error(self, message: str) -> None:
        if self._sampler_error and message in self._sampler_error:
            return
        self._sampler_error = (
            f"{self._sampler_error}; {message}" if self._sampler_error else message
        )

    def start(self) -> None:
        """Prime sensors, take an initial sample, and start the worker thread."""
        if self._started or self._samples:
            raise RuntimeError("hardware sampler instances cannot be restarted")
        self._started = True
        self._request_started_at = None
        self._stop_event.clear()
        try:
            self._system.prime()
        except Exception as exc:
            self._note_error(f"System sensor priming failed: {exc}")
        try:
            self._gpu.initialize()
        except Exception as exc:
            self._note_error(f"GPU sensor initialization failed: {exc}")
        self._started_at = self._clock()
        self._capture()
        self._thread = threading.Thread(
            target=self._run,
            name="benchmark-hardware-sampler",
            daemon=True,
        )
        self._thread.start()

    def mark_request_start(self) -> None:
        """Mark the transition from baseline to inference sampling."""
        if not self._started or self._started_at is None:
            raise RuntimeError("hardware sampler is not running")
        self._request_started_at = self._clock()

    def stop(self) -> HardwareSummary:
        """Stop the worker, take a final sample, and return the summary."""
        if not self._started or self._started_at is None:
            raise RuntimeError("hardware sampler is not running")
        self._closing = True
        self._stop_event.set()
        thread = self._thread
        alive = False
        if thread is not None:
            thread.join(timeout=max(5.0, self.interval_seconds * 3))
            alive = thread.is_alive()
            if alive:
                self._note_error(
                    "Hardware sampler thread did not stop within the timeout"
                )
            else:
                self._capture()
        if not alive:
            self._close_gpu()
        with self._lock:
            samples = list(self._samples)
        self._started = False
        self._thread = thread if alive else None
        return build_hardware_summary(
            samples,
            sampling_interval_ms=round(self.interval_seconds * 1000),
            baseline_seconds=self.baseline_seconds,
            gpu_error=self._gpu.error,
            sampler_error=self._sampler_error,
        )

    def _close_gpu(self) -> None:
        if self._gpu_closed:
            return
        self._gpu_closed = True
        try:
            self._gpu.close()
        except Exception as exc:
            self._note_error(f"GPU sensor close failed: {exc}")

    def _run(self) -> None:
        try:
            while not self._stop_event.wait(self.interval_seconds):
                self._capture()
        finally:
            if self._closing:
                self._close_gpu()

    def _capture(self) -> None:
        sample_started = self._clock()
        try:
            system = self._system.sample()
        except Exception as exc:
            system = SystemReading(None, None, None, None, None, ())
            self._note_error(f"System sample failed: {exc}")
        try:
            gpu = self._gpu.sample()
        except Exception as exc:
            gpu = GpuReading(
                GpuSource.UNAVAILABLE,
                error=f"GPU sample failed: {exc}",
            )
            self._note_error(f"GPU sample failed: {exc}")
        captured_at = self._clock()
        phase = (
            SamplePhase.INFERENCE
            if self._request_started_at is not None
            else SamplePhase.BASELINE
        )
        sample = HardwareSample(
            timestamp=self._wall_clock(),
            elapsed_seconds=max(0.0, captured_at - (self._started_at or captured_at)),
            phase=phase,
            cpu_percent=system.cpu_percent,
            ram_total_bytes=system.ram_total_bytes,
            ram_available_bytes=system.ram_available_bytes,
            ram_used_bytes=system.ram_used_bytes,
            ram_percent=system.ram_percent,
            ollama_process_count=system.ollama_process_count,
            ollama_process_names=system.ollama_process_names,
            ollama_rss_bytes=system.ollama_rss_bytes,
            gpu_index=gpu.index,
            gpu_name=gpu.name,
            gpu_utilization_percent=gpu.utilization_percent,
            gpu_memory_utilization_percent=gpu.memory_utilization_percent,
            vram_total_bytes=gpu.vram_total_bytes,
            vram_used_bytes=gpu.vram_used_bytes,
            vram_free_bytes=gpu.vram_free_bytes,
            gpu_temperature_c=gpu.temperature_c,
            gpu_source=gpu.source,
            sample_overhead_ms=max(0.0, (captured_at - sample_started) * 1000),
        )
        with self._lock:
            self._samples.append(sample)
