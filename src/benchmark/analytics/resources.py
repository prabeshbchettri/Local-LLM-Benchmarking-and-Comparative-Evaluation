"""
File: resources.py

Description:
Aggregation of stored CPU, RAM, GPU, VRAM, and model-specific VRAM measurements.

Why this file exists:
Hardware values are system-wide, frequently unavailable on Windows, and easy to
misread. This module averages only measured values and states what is missing,
including whether VRAM was attributed specifically to the Ollama model.

Responsibilities:
- Average CPU, RAM, GPU utilization, system VRAM, and model VRAM.
- Convert byte measurements to MiB.
- Compute quality per CPU point, RAM GiB, VRAM GiB, and model VRAM GiB.
- Record which measurements were unavailable without treating them as zero.
"""

from __future__ import annotations

from collections.abc import Sequence

from benchmark.analytics.models import ResourceUsage
from benchmark.database.repositories import AnalyticsRow

BYTES_PER_MIB = 1024 * 1024
MIB_PER_GIB = 1024


def _mean(values: Sequence[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None
    return float(sum(clean) / len(clean))


def _per_unit(quality: float | None, amount: float | None) -> float | None:
    if quality is None or amount is None or amount <= 0:
        return None
    return quality / amount


def resource_usage(
    rows: Sequence[AnalyticsRow],
    quality_score: float | None,
) -> ResourceUsage:
    """Average measured resource usage and derive quality per resource unit."""
    hardware = [row for row in rows if row.hardware_present]
    cpu = _mean([row.cpu_avg_percent for row in hardware])
    ram_mib = _mean(
        [
            row.ram_avg_used_bytes / BYTES_PER_MIB
            if row.ram_avg_used_bytes is not None
            else None
            for row in hardware
        ]
    )
    gpu = _mean([row.gpu_utilization_avg_percent for row in hardware])
    vram_mib = _mean(
        [
            row.vram_avg_used_bytes / BYTES_PER_MIB
            if row.vram_avg_used_bytes is not None
            else None
            for row in hardware
        ]
    )
    model_vram_mib = _mean(
        [
            row.ollama_model_vram_bytes / BYTES_PER_MIB
            if row.ollama_model_vram_bytes is not None
            else None
            for row in hardware
        ]
    )
    limitations: list[str] = []
    if not hardware:
        limitations.append("no hardware measurements were stored")
    elif cpu is None:
        limitations.append("CPU usage was not measured")
    if hardware and ram_mib is None:
        limitations.append("RAM usage was not measured")
    if hardware and gpu is None:
        limitations.append("GPU utilization was not measured")
    if hardware and vram_mib is None:
        limitations.append("system VRAM usage was not measured")
    if hardware and model_vram_mib is None:
        limitations.append("model-specific VRAM usage was not measured")
    return ResourceUsage(
        hardware_samples=len(hardware),
        cpu_avg_percent=cpu,
        ram_avg_mib=ram_mib,
        gpu_utilization_avg_percent=gpu,
        vram_avg_mib=vram_mib,
        model_vram_avg_mib=model_vram_mib,
        cpu_available=cpu is not None,
        ram_available=ram_mib is not None,
        gpu_available=gpu is not None,
        vram_available=vram_mib is not None,
        model_vram_available=model_vram_mib is not None,
        quality_per_cpu_point=_per_unit(quality_score, cpu),
        quality_per_ram_gib=_per_unit(
            quality_score,
            ram_mib / MIB_PER_GIB if ram_mib is not None else None,
        ),
        quality_per_vram_gib=_per_unit(
            quality_score,
            vram_mib / MIB_PER_GIB if vram_mib is not None else None,
        ),
        quality_per_model_vram_gib=_per_unit(
            quality_score,
            model_vram_mib / MIB_PER_GIB if model_vram_mib is not None else None,
        ),
        limitations=limitations,
    )
