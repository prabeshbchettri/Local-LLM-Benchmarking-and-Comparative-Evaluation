"""
File: test_monitoring_sampler.py

Description:
Unit tests for hardware sampler start/stop, phases, and safe failure handling.

Responsibilities:
- Verify baseline and inference samples are separated.
- Verify start and stop controls.
- Verify sensor failures produce nulls and sampler errors.
"""

from __future__ import annotations

import pytest

from benchmark.models.hardware import GpuSource, SamplePhase
from benchmark.monitoring.gpu import GpuReading
from benchmark.monitoring.sampler import HardwareSampler
from benchmark.monitoring.system import SystemReading

pytestmark = pytest.mark.unit


class FakeSystemSampler:
    def __init__(self, readings: list[SystemReading], *, fail: bool = False) -> None:
        self.readings = iter(readings)
        self.fail = fail
        self.primed = False

    def prime(self) -> None:
        self.primed = True

    def sample(self) -> SystemReading:
        if self.fail:
            raise RuntimeError("sensor failed")
        return next(self.readings)


class FakeGpuSampler:
    def __init__(self, readings: list[GpuReading], *, fail: bool = False) -> None:
        self.readings = iter(readings)
        self.fail = fail
        self.initialized = False
        self.closed = False
        self.error: str | None = None

    def initialize(self) -> None:
        self.initialized = True

    def sample(self) -> GpuReading:
        if self.fail:
            raise RuntimeError("gpu failed")
        return next(self.readings)

    def close(self) -> None:
        self.closed = True


def _system(cpu: float, ram: int, rss: int) -> SystemReading:
    return SystemReading(
        cpu_percent=cpu,
        ram_total_bytes=16 * 1024**3,
        ram_available_bytes=8 * 1024**3,
        ram_used_bytes=ram,
        ram_percent=50.0,
        ollama_processes=(),
    )


def _gpu(utilization: float, vram: int) -> GpuReading:
    return GpuReading(
        source=GpuSource.NVML,
        index=0,
        name="Fake GPU",
        utilization_percent=utilization,
        memory_utilization_percent=utilization,
        vram_total_bytes=4 * 1024**3,
        vram_used_bytes=vram,
        vram_free_bytes=4 * 1024**3 - vram,
        temperature_c=40.0,
    )


def test_sampler_separates_baseline_and_inference() -> None:
    system = FakeSystemSampler([_system(10.0, 1000, 100), _system(50.0, 3000, 300)])
    gpu = FakeGpuSampler([_gpu(20.0, 500), _gpu(80.0, 1500)])
    sampler = HardwareSampler(
        interval_seconds=0.5,
        baseline_seconds=2.0,
        system_sampler=system,
        gpu_sampler=gpu,
    )
    sampler.start()
    assert sampler.is_running is True
    assert system.primed is True
    assert gpu.initialized is True
    sampler.mark_request_start()
    summary = sampler.stop()
    assert sampler.is_running is False
    assert gpu.closed is True
    assert summary.sampling_interval_ms == 500
    assert summary.baseline_seconds == pytest.approx(2.0)
    assert summary.baseline.sample_count == 1
    assert summary.inference.sample_count == 1
    assert summary.baseline.cpu_avg_percent == pytest.approx(10.0)
    assert summary.inference.cpu_avg_percent == pytest.approx(50.0)
    assert summary.inference.ram_avg_used_bytes == 3000
    assert summary.inference.gpu_utilization_avg_percent == pytest.approx(80.0)
    assert summary.inference.vram_avg_used_bytes == 1500
    assert summary.samples[0].phase is SamplePhase.BASELINE
    assert summary.samples[1].phase is SamplePhase.INFERENCE
    assert summary.sampler_overhead_avg_ms is not None


def test_sampler_controls_reject_invalid_state() -> None:
    sampler = HardwareSampler(
        interval_seconds=1.0,
        system_sampler=FakeSystemSampler([_system(10.0, 1000, 100)]),
        gpu_sampler=FakeGpuSampler([_gpu(20.0, 500)]),
    )
    with pytest.raises(RuntimeError, match="not running"):
        sampler.stop()
    sampler.start()
    with pytest.raises(RuntimeError, match="cannot be restarted"):
        sampler.start()
    sampler.mark_request_start()
    sampler.stop()


def test_sensor_failures_are_recorded_without_raising() -> None:
    sampler = HardwareSampler(
        interval_seconds=1.0,
        system_sampler=FakeSystemSampler([], fail=True),
        gpu_sampler=FakeGpuSampler([], fail=True),
    )
    sampler.start()
    sampler.mark_request_start()
    summary = sampler.stop()
    assert summary.sample_count == 2
    assert summary.inference.cpu_avg_percent is None
    assert summary.inference.gpu_utilization_avg_percent is None
    assert summary.sampler_error is not None
    assert "System sample failed" in summary.sampler_error
    assert "GPU sample failed" in summary.sampler_error
