"""
File: hardware.py

Description:
Typed hardware samples and resource summaries for inference monitoring.

Responsibilities:
- Preserve timestamped system-wide and Ollama process-specific measurements.
- Represent unavailable CPU, RAM, GPU, VRAM, and temperature metrics as nulls.
- Distinguish baseline samples from samples collected during a request.
- Provide average and peak summaries without claiming exclusive model attribution.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator

HARDWARE_SCHEMA_VERSION = "1.0.0"
ATTRIBUTION_NOTE = (
    "System-wide CPU, RAM, and GPU measurements include the operating system and "
    "other processes; they are not exclusively attributable to the model. Ollama "
    "RSS is process-specific only when a matching Ollama process is found."
)
WDDM_NOTE = (
    "Windows WDDM may report GPU utilization, VRAM, or temperature as unavailable "
    "or zero; unavailable values are stored as null instead of being inferred."
)


class HardwareSchema(BaseModel):
    """Base schema for hardware measurements and summaries."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=False,
    )


class SamplePhase(StrEnum):
    """Collection window for one hardware sample."""

    BASELINE = "baseline"
    INFERENCE = "inference"


class GpuSource(StrEnum):
    """Backend that produced a GPU measurement."""

    NVML = "nvml"
    NVIDIA_SMI = "nvidia-smi"
    UNAVAILABLE = "unavailable"


class HardwareSample(HardwareSchema):
    """One timestamped hardware measurement."""

    timestamp: datetime
    elapsed_seconds: float = Field(ge=0.0)
    phase: SamplePhase
    cpu_percent: float | None = Field(default=None, ge=0.0)
    ram_total_bytes: int | None = Field(default=None, ge=0)
    ram_available_bytes: int | None = Field(default=None, ge=0)
    ram_used_bytes: int | None = Field(default=None, ge=0)
    ram_percent: float | None = Field(default=None, ge=0.0, le=100.0)
    ollama_process_count: int = Field(default=0, ge=0)
    ollama_process_names: list[str] = Field(default_factory=list)
    ollama_rss_bytes: int | None = Field(default=None, ge=0)
    gpu_index: int | None = Field(default=None, ge=0)
    gpu_name: str | None = None
    gpu_utilization_percent: float | None = Field(default=None, ge=0.0, le=100.0)
    gpu_memory_utilization_percent: float | None = Field(
        default=None,
        ge=0.0,
        le=100.0,
    )
    vram_total_bytes: int | None = Field(default=None, ge=0)
    vram_used_bytes: int | None = Field(default=None, ge=0)
    vram_free_bytes: int | None = Field(default=None, ge=0)
    gpu_temperature_c: float | None = Field(default=None, ge=0.0, le=150.0)
    gpu_source: GpuSource | None = None
    sample_overhead_ms: float | None = Field(default=None, ge=0.0)

    @field_validator("timestamp")
    @classmethod
    def validate_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("hardware sample timestamps must include a timezone")
        return value


class WindowSummary(HardwareSchema):
    """Average and peak values for one collection window."""

    sample_count: int = Field(default=0, ge=0)
    low_sample_count: bool = False
    duration_seconds: float = Field(default=0.0, ge=0.0)
    cpu_avg_percent: float | None = Field(default=None, ge=0.0)
    cpu_peak_percent: float | None = Field(default=None, ge=0.0)
    ram_avg_used_bytes: int | None = Field(default=None, ge=0)
    ram_peak_used_bytes: int | None = Field(default=None, ge=0)
    ram_avg_percent: float | None = Field(default=None, ge=0.0, le=100.0)
    ram_peak_percent: float | None = Field(default=None, ge=0.0, le=100.0)
    ollama_rss_avg_bytes: int | None = Field(default=None, ge=0)
    ollama_rss_peak_bytes: int | None = Field(default=None, ge=0)
    gpu_utilization_avg_percent: float | None = Field(default=None, ge=0.0)
    gpu_utilization_peak_percent: float | None = Field(default=None, ge=0.0)
    gpu_memory_utilization_avg_percent: float | None = Field(default=None, ge=0.0)
    gpu_memory_utilization_peak_percent: float | None = Field(default=None, ge=0.0)
    vram_avg_used_bytes: int | None = Field(default=None, ge=0)
    vram_peak_used_bytes: int | None = Field(default=None, ge=0)
    gpu_temperature_avg_c: float | None = Field(default=None, ge=0.0)
    gpu_temperature_peak_c: float | None = Field(default=None, ge=0.0)


class HardwareSummary(HardwareSchema):
    """Hardware summary associated with one inference request."""

    schema_version: str = HARDWARE_SCHEMA_VERSION
    sampling_interval_ms: int = Field(ge=1)
    baseline_seconds: float = Field(default=0.0, ge=0.0)
    sample_count: int = Field(default=0, ge=0)
    baseline: WindowSummary = Field(default_factory=WindowSummary)
    inference: WindowSummary = Field(default_factory=WindowSummary)
    overall: WindowSummary = Field(default_factory=WindowSummary)
    ram_total_bytes: int | None = Field(default=None, ge=0)
    vram_total_bytes: int | None = Field(default=None, ge=0)
    gpu_name: str | None = None
    gpu_source: GpuSource | None = None
    gpu_available: bool = False
    gpu_utilization_available: bool = False
    gpu_memory_available: bool = False
    gpu_temperature_available: bool = False
    gpu_error: str | None = None
    sampler_error: str | None = None
    gpu_note: str | None = WDDM_NOTE
    ollama_process_available: bool = False
    ollama_process_names: list[str] = Field(default_factory=list)
    ollama_model_vram_available: bool = False
    ollama_model_name: str | None = None
    ollama_model_size_bytes: int | None = Field(default=None, ge=0)
    ollama_model_size_vram_bytes: int | None = Field(default=None, ge=0)
    ollama_model_vram_ratio: float | None = Field(default=None, ge=0.0, le=1.0)
    ollama_model_vram_error: str | None = None
    sampler_overhead_avg_ms: float | None = Field(default=None, ge=0.0)
    sampler_overhead_peak_ms: float | None = Field(default=None, ge=0.0)
    attribution_note: str = ATTRIBUTION_NOTE
    samples: list[HardwareSample] = Field(default_factory=list)
