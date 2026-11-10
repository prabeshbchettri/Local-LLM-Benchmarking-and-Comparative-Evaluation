"""
File: summaries.py

Description:
Average and peak calculations for hardware samples.

Responsibilities:
- Separate baseline, inference, and overall collection windows.
- Calculate averages and peaks only from available values.
- Preserve null metrics instead of substituting zeroes.
- Aggregate GPU identity, availability flags, and process names.
"""

from __future__ import annotations

from collections.abc import Iterable

from benchmark.models.hardware import (
    HardwareSample,
    HardwareSummary,
    SamplePhase,
    WindowSummary,
)


def _floats(values: Iterable[float | None]) -> list[float]:
    return [value for value in values if value is not None]


def _ints(values: Iterable[int | None]) -> list[int]:
    return [value for value in values if value is not None]


def _avg_float(values: Iterable[float | None]) -> float | None:
    present = _floats(values)
    return sum(present) / len(present) if present else None


def _peak_float(values: Iterable[float | None]) -> float | None:
    present = _floats(values)
    return max(present) if present else None


def _avg_int(values: Iterable[int | None]) -> int | None:
    present = _ints(values)
    return round(sum(present) / len(present)) if present else None


def _peak_int(values: Iterable[int | None]) -> int | None:
    present = _ints(values)
    return max(present) if present else None


def _window(samples: list[HardwareSample]) -> WindowSummary:
    if not samples:
        return WindowSummary(low_sample_count=True)
    elapsed = [sample.elapsed_seconds for sample in samples]
    return WindowSummary(
        sample_count=len(samples),
        low_sample_count=len(samples) < 2,
        duration_seconds=max(elapsed) - min(elapsed),
        cpu_avg_percent=_avg_float(sample.cpu_percent for sample in samples),
        cpu_peak_percent=_peak_float(sample.cpu_percent for sample in samples),
        ram_avg_used_bytes=_avg_int(sample.ram_used_bytes for sample in samples),
        ram_peak_used_bytes=_peak_int(sample.ram_used_bytes for sample in samples),
        ram_avg_percent=_avg_float(sample.ram_percent for sample in samples),
        ram_peak_percent=_peak_float(sample.ram_percent for sample in samples),
        ollama_rss_avg_bytes=_avg_int(sample.ollama_rss_bytes for sample in samples),
        ollama_rss_peak_bytes=_peak_int(sample.ollama_rss_bytes for sample in samples),
        gpu_utilization_avg_percent=_avg_float(
            sample.gpu_utilization_percent for sample in samples
        ),
        gpu_utilization_peak_percent=_peak_float(
            sample.gpu_utilization_percent for sample in samples
        ),
        gpu_memory_utilization_avg_percent=_avg_float(
            sample.gpu_memory_utilization_percent for sample in samples
        ),
        gpu_memory_utilization_peak_percent=_peak_float(
            sample.gpu_memory_utilization_percent for sample in samples
        ),
        vram_avg_used_bytes=_avg_int(sample.vram_used_bytes for sample in samples),
        vram_peak_used_bytes=_peak_int(sample.vram_used_bytes for sample in samples),
        gpu_temperature_avg_c=_avg_float(
            sample.gpu_temperature_c for sample in samples
        ),
        gpu_temperature_peak_c=_peak_float(
            sample.gpu_temperature_c for sample in samples
        ),
    )


def build_hardware_summary(
    samples: Iterable[HardwareSample],
    *,
    sampling_interval_ms: int,
    baseline_seconds: float,
    gpu_error: str | None = None,
    sampler_error: str | None = None,
) -> HardwareSummary:
    """Build baseline, inference, and overall summaries from collected samples."""
    ordered = list(samples)
    baseline = _window(
        [sample for sample in ordered if sample.phase is SamplePhase.BASELINE]
    )
    inference = _window(
        [sample for sample in ordered if sample.phase is SamplePhase.INFERENCE]
    )
    overall = _window(ordered)
    last_gpu = next(
        (
            sample
            for sample in reversed(ordered)
            if sample.gpu_source is not None
        ),
        None,
    )
    last_ram = next(
        (sample for sample in reversed(ordered) if sample.ram_total_bytes is not None),
        None,
    )
    last_vram = next(
        (
            sample
            for sample in reversed(ordered)
            if sample.vram_total_bytes is not None
        ),
        None,
    )
    process_names = sorted(
        {
            name
            for sample in ordered
            for name in sample.ollama_process_names
        }
    )
    overheads = [
        sample.sample_overhead_ms
        for sample in ordered
        if sample.sample_overhead_ms is not None
    ]
    gpu_source = last_gpu.gpu_source if last_gpu is not None else None
    return HardwareSummary(
        sampling_interval_ms=sampling_interval_ms,
        baseline_seconds=baseline_seconds,
        sample_count=len(ordered),
        baseline=baseline,
        inference=inference,
        overall=overall,
        ram_total_bytes=last_ram.ram_total_bytes if last_ram is not None else None,
        vram_total_bytes=last_vram.vram_total_bytes if last_vram is not None else None,
        gpu_name=last_gpu.gpu_name if last_gpu is not None else None,
        gpu_source=gpu_source,
        gpu_available=gpu_source is not None
        and gpu_source.value != "unavailable",
        gpu_utilization_available=any(
            sample.gpu_utilization_percent is not None for sample in ordered
        ),
        gpu_memory_available=any(
            sample.vram_total_bytes is not None and sample.gpu_source is not None
            for sample in ordered
        ),
        gpu_temperature_available=any(
            sample.gpu_temperature_c is not None for sample in ordered
        ),
        gpu_error=gpu_error,
        sampler_error=sampler_error,
        ollama_process_available=any(
            sample.ollama_process_count > 0 for sample in ordered
        ),
        ollama_process_names=process_names,
        sampler_overhead_avg_ms=(
            sum(overheads) / len(overheads) if overheads else None
        ),
        sampler_overhead_peak_ms=max(overheads) if overheads else None,
        samples=ordered,
    )
