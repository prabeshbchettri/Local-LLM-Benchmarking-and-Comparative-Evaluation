"""
File: test_monitoring_summaries.py

Description:
Unit tests for baseline, inference, and overall hardware summaries.

Responsibilities:
- Verify average and peak calculations.
- Verify null metrics stay null.
- Verify GPU and process availability flags.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from benchmark.models.hardware import (
    GpuSource,
    HardwareSample,
    SamplePhase,
)
from benchmark.monitoring.summaries import build_hardware_summary

pytestmark = pytest.mark.unit


def _sample(
    elapsed: float,
    phase: SamplePhase,
    *,
    cpu: float | None,
    ram: int | None,
    ram_percent: float | None,
    rss: int | None,
    gpu: float | None,
    gpu_memory: float | None,
    vram: int | None,
    temperature: float | None,
    source: GpuSource | None = GpuSource.NVML,
    names: list[str] | None = None,
) -> HardwareSample:
    return HardwareSample(
        timestamp=datetime.now(timezone.utc),
        elapsed_seconds=elapsed,
        phase=phase,
        cpu_percent=cpu,
        ram_total_bytes=16 * 1024**3,
        ram_used_bytes=ram,
        ram_percent=ram_percent,
        ollama_process_count=len(names or []),
        ollama_process_names=names or [],
        ollama_rss_bytes=rss,
        gpu_name="Fake GPU" if source is not None else None,
        gpu_utilization_percent=gpu,
        gpu_memory_utilization_percent=gpu_memory,
        vram_total_bytes=4 * 1024**3 if source is not None else None,
        vram_used_bytes=vram,
        gpu_temperature_c=temperature,
        gpu_source=source,
        sample_overhead_ms=2.0,
    )


def test_builds_baseline_inference_and_overall_summaries() -> None:
    samples = [
        _sample(
            0.0,
            SamplePhase.BASELINE,
            cpu=10,
            ram=1000,
            ram_percent=10,
            rss=100,
            gpu=20,
            gpu_memory=30,
            vram=500,
            temperature=40,
            names=["ollama.exe"],
        ),
        _sample(
            1.0,
            SamplePhase.BASELINE,
            cpu=20,
            ram=2000,
            ram_percent=20,
            rss=200,
            gpu=40,
            gpu_memory=50,
            vram=1000,
            temperature=42,
            names=["ollama.exe", "ollama app.exe"],
        ),
        _sample(
            2.0,
            SamplePhase.INFERENCE,
            cpu=60,
            ram=3000,
            ram_percent=30,
            rss=300,
            gpu=80,
            gpu_memory=70,
            vram=1500,
            temperature=50,
            names=["ollama.exe"],
        ),
        _sample(
            3.0,
            SamplePhase.INFERENCE,
            cpu=70,
            ram=4000,
            ram_percent=40,
            rss=400,
            gpu=90,
            gpu_memory=80,
            vram=2500,
            temperature=55,
            names=["ollama.exe"],
        ),
    ]
    summary = build_hardware_summary(
        samples,
        sampling_interval_ms=1000,
        baseline_seconds=2.0,
    )
    assert summary.sample_count == 4
    assert summary.baseline.sample_count == 2
    assert summary.baseline.low_sample_count is False
    assert summary.baseline.duration_seconds == pytest.approx(1.0)
    assert summary.baseline.cpu_avg_percent == pytest.approx(15.0)
    assert summary.baseline.cpu_peak_percent == pytest.approx(20.0)
    assert summary.inference.sample_count == 2
    assert summary.inference.cpu_avg_percent == pytest.approx(65.0)
    assert summary.inference.cpu_peak_percent == pytest.approx(70.0)
    assert summary.inference.ram_avg_used_bytes == 3500
    assert summary.inference.ram_peak_used_bytes == 4000
    assert summary.inference.ollama_rss_avg_bytes == 350
    assert summary.inference.ollama_rss_peak_bytes == 400
    assert summary.inference.gpu_utilization_avg_percent == pytest.approx(85.0)
    assert summary.inference.gpu_utilization_peak_percent == pytest.approx(90.0)
    assert summary.inference.gpu_memory_utilization_avg_percent == pytest.approx(75.0)
    assert summary.inference.vram_avg_used_bytes == 2000
    assert summary.inference.vram_peak_used_bytes == 2500
    assert summary.inference.gpu_temperature_avg_c == pytest.approx(52.5)
    assert summary.inference.gpu_temperature_peak_c == pytest.approx(55.0)
    assert summary.overall.cpu_avg_percent == pytest.approx(40.0)
    assert summary.gpu_name == "Fake GPU"
    assert summary.gpu_source is GpuSource.NVML
    assert summary.gpu_available is True
    assert summary.gpu_utilization_available is True
    assert summary.gpu_memory_available is True
    assert summary.gpu_temperature_available is True
    assert summary.ollama_process_available is True
    assert summary.ollama_process_names == ["ollama app.exe", "ollama.exe"]
    assert summary.sampler_overhead_avg_ms == pytest.approx(2.0)
    assert summary.sampler_overhead_peak_ms == pytest.approx(2.0)
    assert summary.ram_total_bytes == 16 * 1024**3
    assert summary.vram_total_bytes == 4 * 1024**3
    assert len(summary.samples) == 4


def test_unavailable_metrics_remain_null() -> None:
    sample = _sample(
        0.0,
        SamplePhase.INFERENCE,
        cpu=None,
        ram=None,
        ram_percent=None,
        rss=None,
        gpu=None,
        gpu_memory=None,
        vram=None,
        temperature=None,
        source=None,
        names=[],
    )
    summary = build_hardware_summary(
        [sample],
        sampling_interval_ms=500,
        baseline_seconds=0.0,
        gpu_error="no GPU",
    )
    assert summary.inference.low_sample_count is True
    assert summary.inference.cpu_avg_percent is None
    assert summary.inference.ram_avg_used_bytes is None
    assert summary.inference.gpu_utilization_avg_percent is None
    assert summary.inference.vram_avg_used_bytes is None
    assert summary.inference.gpu_temperature_avg_c is None
    assert summary.gpu_available is False
    assert summary.gpu_utilization_available is False
    assert summary.gpu_memory_available is False
    assert summary.gpu_temperature_available is False
    assert summary.ollama_process_available is False
    assert summary.gpu_error == "no GPU"


def test_empty_samples_produce_zero_count_without_errors() -> None:
    summary = build_hardware_summary(
        [],
        sampling_interval_ms=1000,
        baseline_seconds=2.0,
    )
    assert summary.sample_count == 0
    assert summary.baseline.sample_count == 0
    assert summary.inference.sample_count == 0
    assert summary.inference.low_sample_count is True
    assert summary.inference.cpu_avg_percent is None
    assert summary.samples == []
