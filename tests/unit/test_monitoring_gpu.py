"""
File: test_monitoring_gpu.py

Description:
Unit tests for NVML GPU monitoring and the nvidia-smi fallback.

Responsibilities:
- Verify NVML is preferred when usable.
- Verify nvidia-smi fallback parsing.
- Verify missing libraries, unavailable utilities, and N/A fields.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import benchmark.monitoring.gpu as gpu_module
from benchmark.models.hardware import GpuSource
from benchmark.monitoring.gpu import GpuSampler

pytestmark = pytest.mark.unit

MIB = 1024**2


class FakeNvml:
    NVML_TEMPERATURE_GPU = 0

    def __init__(
        self,
        *,
        fail_init: bool = False,
        fail_utilization: bool = False,
    ) -> None:
        self.fail_init = fail_init
        self.fail_utilization = fail_utilization
        self.shutdown_called = False

    def nvmlInit(self) -> None:
        if self.fail_init:
            raise RuntimeError("driver unavailable")

    def nvmlDeviceGetHandleByIndex(self, index: int):
        return SimpleNamespace(index=index)

    def nvmlDeviceGetName(self, handle):
        return b"Fake GPU"

    def nvmlDeviceGetUtilizationRates(self, handle):
        if self.fail_utilization:
            raise RuntimeError("WDDM utilization unavailable")
        return SimpleNamespace(gpu=55, memory=25)

    def nvmlDeviceGetMemoryInfo(self, handle):
        return SimpleNamespace(total=4 * 1024**3, used=1024**3, free=3 * 1024**3)

    def nvmlDeviceGetTemperature(self, handle, sensor):
        return 42

    def nvmlShutdown(self) -> None:
        self.shutdown_called = True


def test_nvml_is_preferred() -> None:
    nvml = FakeNvml()
    sampler = GpuSampler(nvml_module=nvml)
    sampler.initialize()
    reading = sampler.sample()
    assert reading.source is GpuSource.NVML
    assert reading.name == "Fake GPU"
    assert reading.utilization_percent == pytest.approx(55.0)
    assert reading.memory_utilization_percent == pytest.approx(25.0)
    assert reading.vram_total_bytes == 4 * 1024**3
    assert reading.vram_used_bytes == 1024**3
    assert reading.temperature_c == pytest.approx(42.0)
    sampler.close()
    assert nvml.shutdown_called is True


def test_wddm_utilization_gap_preserves_memory_and_temperature() -> None:
    sampler = GpuSampler(nvml_module=FakeNvml(fail_utilization=True))
    reading = sampler.sample()
    assert reading.source is GpuSource.NVML
    assert reading.utilization_percent is None
    assert reading.memory_utilization_percent is None
    assert reading.vram_used_bytes == 1024**3
    assert reading.temperature_c == pytest.approx(42.0)


def test_nvml_failure_falls_back_to_nvidia_smi() -> None:
    sampler = GpuSampler(
        nvml_module=FakeNvml(fail_init=True),
        smi_runner=lambda: "0, Fake SMI GPU, 40, 20, 4096, 1024, 3072, 45\n",
    )
    reading = sampler.sample()
    assert reading.source is GpuSource.NVIDIA_SMI
    assert reading.name == "Fake SMI GPU"
    assert reading.utilization_percent == pytest.approx(40.0)
    assert reading.memory_utilization_percent == pytest.approx(20.0)
    assert reading.vram_total_bytes == 4096 * MIB
    assert reading.vram_used_bytes == 1024 * MIB
    assert reading.vram_free_bytes == 3072 * MIB
    assert reading.temperature_c == pytest.approx(45.0)


def test_missing_pynvml_uses_nvidia_smi(monkeypatch) -> None:
    monkeypatch.setattr(gpu_module, "_PYNVML", None)
    sampler = GpuSampler(
        smi_runner=lambda: "0, Fallback GPU, 10, 5, 2048, 512, 1536, 38\n"
    )
    reading = sampler.sample()
    assert reading.source is GpuSource.NVIDIA_SMI
    assert reading.name == "Fallback GPU"
    assert sampler.error is not None
    assert "pynvml" in sampler.error


def test_nvidia_smi_na_values_become_nulls() -> None:
    sampler = GpuSampler(
        nvml_module=FakeNvml(fail_init=True),
        smi_runner=lambda: "0, Fake GPU, N/A, N/A, 4096, N/A, N/A, N/A\n",
    )
    reading = sampler.sample()
    assert reading.source is GpuSource.NVIDIA_SMI
    assert reading.utilization_percent is None
    assert reading.memory_utilization_percent is None
    assert reading.vram_total_bytes == 4096 * MIB
    assert reading.vram_used_bytes is None
    assert reading.temperature_c is None


def test_unavailable_nvidia_smi_is_explicit() -> None:
    def fail() -> str:
        raise RuntimeError("not found")

    sampler = GpuSampler(
        nvml_module=FakeNvml(fail_init=True),
        smi_runner=fail,
    )
    reading = sampler.sample()
    assert reading.source is GpuSource.UNAVAILABLE
    assert reading.error is not None
    assert reading.utilization_percent is None
    assert reading.vram_total_bytes is None
