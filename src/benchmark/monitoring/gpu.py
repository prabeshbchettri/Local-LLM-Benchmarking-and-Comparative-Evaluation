"""
File: gpu.py

Description:
GPU monitoring through NVML with an nvidia-smi fallback.

Responsibilities:
- Prefer NVML for utilization, VRAM, and temperature measurements.
- Fall back to nvidia-smi when NVML is missing or unusable.
- Preserve unavailable metrics as nulls, including Windows WDDM gaps.
- Report the backend, GPU identity, and backend error without raising into inference.
"""

from __future__ import annotations

import csv
import shutil
import subprocess
from dataclasses import dataclass
from types import ModuleType
from typing import Callable

from benchmark.models.hardware import GpuSource

try:
    import pynvml as _PYNVML
except ImportError:
    _PYNVML = None

SMI_FIELDS = (
    "index",
    "name",
    "utilization.gpu",
    "utilization.memory",
    "memory.total",
    "memory.used",
    "memory.free",
    "temperature.gpu",
)
SMI_COMMAND = (
    "nvidia-smi",
    "--query-gpu=" + ",".join(SMI_FIELDS),
    "--format=csv,noheader,nounits",
)
MIB = 1024**2


@dataclass(frozen=True)
class GpuReading:
    """One GPU measurement from NVML, nvidia-smi, or neither."""

    source: GpuSource
    index: int | None = None
    name: str | None = None
    utilization_percent: float | None = None
    memory_utilization_percent: float | None = None
    vram_total_bytes: int | None = None
    vram_used_bytes: int | None = None
    vram_free_bytes: int | None = None
    temperature_c: float | None = None
    error: str | None = None


def _safe_float(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.upper() == "N/A":
        return None
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _safe_int(value: object) -> int | None:
    number = _safe_float(value)
    return None if number is None else int(number)


def _safe_text(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    text = str(value).strip()
    return text or None


def _run_nvidia_smi() -> str:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        raise RuntimeError("nvidia-smi is not available on PATH")
    completed = subprocess.run(
        [executable, *SMI_COMMAND[1:]],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or f"exit code {completed.returncode}"
        raise RuntimeError(f"nvidia-smi failed: {detail}")
    return completed.stdout


SmiRunner = Callable[[], str]


class GpuSampler:
    """Collect GPU metrics from NVML, falling back to nvidia-smi."""

    def __init__(
        self,
        *,
        device_index: int = 0,
        nvml_module: ModuleType | None = None,
        smi_runner: SmiRunner | None = None,
    ) -> None:
        self.device_index = device_index
        self._nvml = nvml_module if nvml_module is not None else _PYNVML
        self._smi_runner = smi_runner or _run_nvidia_smi
        self._handle: object | None = None
        self._initialized = False
        self._source = GpuSource.UNAVAILABLE
        self._error: str | None = None
        self._name: str | None = None

    @property
    def source(self) -> GpuSource:
        return self._source

    @property
    def error(self) -> str | None:
        return self._error

    @property
    def name(self) -> str | None:
        return self._name

    def initialize(self) -> None:
        """Initialize NVML or select the nvidia-smi fallback."""
        if self._initialized:
            return
        self._initialized = True
        if self._nvml is not None:
            try:
                self._nvml.nvmlInit()
                self._handle = self._nvml.nvmlDeviceGetHandleByIndex(self.device_index)
                self._name = _safe_text(
                    self._nvml.nvmlDeviceGetName(self._handle)
                )
                self._source = GpuSource.NVML
                self._error = None
                return
            except Exception as exc:
                self._error = f"NVML unavailable: {exc}"
                try:
                    self._nvml.nvmlShutdown()
                except Exception:
                    pass
                self._handle = None
        else:
            self._error = "pynvml is not installed"
        self._source = GpuSource.NVIDIA_SMI
        try:
            reading = self._sample_nvidia_smi()
        except Exception as exc:
            self._error = f"{self._error}; nvidia-smi unavailable: {exc}"
            self._source = GpuSource.UNAVAILABLE
            return
        self._name = reading.name

    def sample(self) -> GpuReading:
        """Return one GPU measurement without raising into the sampler thread."""
        if not self._initialized:
            self.initialize()
        if self._source is GpuSource.NVML:
            try:
                reading = self._sample_nvml()
            except Exception as exc:
                self._error = f"NVML sample failed: {exc}"
                reading = None
            if reading is not None and self._has_measurement(reading):
                return reading
            self._source = GpuSource.NVIDIA_SMI
        if self._source is GpuSource.NVIDIA_SMI:
            try:
                reading = self._sample_nvidia_smi()
            except Exception as exc:
                self._error = f"nvidia-smi sample failed: {exc}"
                self._source = GpuSource.UNAVAILABLE
                return GpuReading(GpuSource.UNAVAILABLE, error=self._error)
            self._name = reading.name
            return reading
        return GpuReading(GpuSource.UNAVAILABLE, error=self._error)

    def close(self) -> None:
        """Release NVML resources."""
        if self._handle is not None and self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass
        self._handle = None
        self._initialized = False

    @staticmethod
    def _has_measurement(reading: GpuReading) -> bool:
        return any(
            value is not None
            for value in (
                reading.utilization_percent,
                reading.memory_utilization_percent,
                reading.vram_total_bytes,
                reading.temperature_c,
            )
        )

    def _sample_nvml(self) -> GpuReading:
        assert self._nvml is not None and self._handle is not None
        name = self._name or _safe_text(
            self._nvml.nvmlDeviceGetName(self._handle)
        )
        utilization = None
        memory_utilization = None
        try:
            rates = self._nvml.nvmlDeviceGetUtilizationRates(self._handle)
            utilization = _safe_float(getattr(rates, "gpu", None))
            memory_utilization = _safe_float(getattr(rates, "memory", None))
        except Exception:
            pass
        total = used = free = None
        try:
            memory = self._nvml.nvmlDeviceGetMemoryInfo(self._handle)
            total = _safe_int(getattr(memory, "total", None))
            used = _safe_int(getattr(memory, "used", None))
            free = _safe_int(getattr(memory, "free", None))
        except Exception:
            pass
        temperature = None
        try:
            temperature = _safe_float(
                self._nvml.nvmlDeviceGetTemperature(
                    self._handle,
                    getattr(self._nvml, "NVML_TEMPERATURE_GPU", 0),
                )
            )
        except Exception:
            pass
        return GpuReading(
            source=GpuSource.NVML,
            index=self.device_index,
            name=name,
            utilization_percent=utilization,
            memory_utilization_percent=memory_utilization,
            vram_total_bytes=total,
            vram_used_bytes=used,
            vram_free_bytes=free,
            temperature_c=temperature,
            error=self._error,
        )

    def _sample_nvidia_smi(self) -> GpuReading:
        output = self._smi_runner()
        line = next((item for item in output.splitlines() if item.strip()), None)
        if line is None:
            raise RuntimeError("nvidia-smi returned no GPU rows")
        row = next(csv.reader([line]))
        if len(row) < len(SMI_FIELDS):
            raise RuntimeError(f"nvidia-smi returned {len(row)} fields, expected {len(SMI_FIELDS)}")
        return GpuReading(
            source=GpuSource.NVIDIA_SMI,
            index=_safe_int(row[0]),
            name=_safe_text(row[1]),
            utilization_percent=_safe_float(row[2]),
            memory_utilization_percent=_safe_float(row[3]),
            vram_total_bytes=(
                None if (total := _safe_int(row[4])) is None else total * MIB
            ),
            vram_used_bytes=(
                None if (used := _safe_int(row[5])) is None else used * MIB
            ),
            vram_free_bytes=(
                None if (free := _safe_int(row[6])) is None else free * MIB
            ),
            temperature_c=_safe_float(row[7]),
            error=self._error,
        )
