"""
File: presentation.py

Description:
Turn a report bundle into one formatted, sectioned view for HTML and PDF output.

Why this file exists:
HTML and PDF must show the same numbers, units, and wording. Formatting here once
means a value cannot appear as a percentage in one format and a fraction in the
other, and missing measurements stay visibly missing instead of becoming zero.

Responsibilities:
- Format numbers with explicit units and em dashes for unavailable values.
- Build the ten required report sections.
- Provide table column definitions and rows for both renderers.
- Provide chart data and self-contained SVG charts.
- Aggregate hardware rows into average and peak values per model.
"""

from __future__ import annotations

from datetime import datetime, timezone
from html import escape
from typing import Any

from benchmark.reporting.bundle import REPORT_SCHEMA_VERSION, ReportBundle

BYTES_PER_MIB = 1024 * 1024
MISSING = "—"


def _text(value: Any, digits: int = 2, suffix: str = "") -> str:
    """Format a value for display, keeping missing values visible."""
    if value is None:
        return MISSING
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return f"{value:,}{suffix}"
    return f"{value:,.{digits}f}{suffix}"


def _percent(fraction: float | None, digits: int = 1) -> str:
    """Format a 0-1 fraction as a percentage."""
    if fraction is None:
        return MISSING
    return f"{fraction * 100:.{digits}f}%"


def _score(value: float | None, digits: int = 1) -> str:
    """Format a 0-100 score."""
    if value is None:
        return MISSING
    return f"{value:.{digits}f}"


def _mib(value: int | float | None, digits: int = 0) -> str:
    """Format a byte count as MiB."""
    if value is None:
        return MISSING
    return f"{value / BYTES_PER_MIB:,.{digits}f} MiB"


def _bytes(value: int | None) -> str:
    """Format a byte count with binary units."""
    if value is None:
        return MISSING
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB":
            return f"{size:,.0f} {unit}" if unit == "B" else f"{size:,.1f} {unit}"
        size /= 1024
    return MISSING


def _timestamp(value: datetime | None) -> str:
    """Format a timestamp in UTC for reproducible reports."""
    if value is None:
        return MISSING
    return value.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _columns(*items: tuple[str, str, str]) -> list[dict[str, str]]:
    """Build table column definitions from (key, label, align) tuples."""
    return [{"key": key, "label": label, "align": align} for key, label, align in items]


def _table(
    columns: list[dict[str, str]], rows: list[dict[str, str]]
) -> dict[str, Any]:
    """Build a table block with explicit columns so empty tables keep headers."""
    return {"columns": columns, "rows": rows}


def _flatten(prefix: str, data: dict[str, Any]) -> list[dict[str, str]]:
    """Flatten a one-level metadata mapping into environment rows."""
    rows: list[dict[str, str]] = []
    for key in sorted(data):
        value = data[key]
        if isinstance(value, (dict, list)):
            rendered = ", ".join(str(item) for item in value) if isinstance(value, list) else str(value)
        else:
            rendered = str(value)
        rows.append({"label": f"{prefix} · {key}" if prefix else key, "value": rendered})
    return rows


def _mean(values: list[float | None]) -> float | None:
    clean = [value for value in values if value is not None]
    return sum(clean) / len(clean) if clean else None


def _max(values: list[float | None]) -> float | None:
    clean = [value for value in values if value is not None]
    return max(clean) if clean else None


def _charts(analysis: Any) -> list[dict[str, Any]]:
    """Build chart specifications from models that have the measured value."""
    definitions = (
        ("composite", "Composite score by model", "score (0–100)", lambda model: model.composite_score, 1),
        ("quality", "Quality score by model", "score (0–100)", lambda model: model.quality_score, 1),
        (
            "latency_p95",
            "P95 total latency by model",
            "seconds (lower is better)",
            lambda model: model.statistics.latency.p95,
            2,
        ),
        (
            "throughput",
            "Mean output rate by model",
            "tokens per second (higher is better)",
            lambda model: model.statistics.output_tokens_per_second.mean,
            1,
        ),
    )
    charts: list[dict[str, Any]] = []
    for key, title, unit, getter, digits in definitions:
        categories: list[str] = []
        values: list[float] = []
        for model in analysis.models:
            value = getter(model)
            if value is None:
                continue
            categories.append(model.model_tag)
            values.append(round(float(value), digits))
        if not values:
            continue
        charts.append(
            {
                "key": key,
                "title": title,
                "unit": unit,
                "categories": categories,
                "values": values,
                "digits": digits,
            }
        )
    return charts


def bar_chart_svg(chart: dict[str, Any], width: int = 700, height: int = 220) -> str:
    """Render one chart as a self-contained inline SVG bar chart."""
    categories = chart["categories"]
    values = chart["values"]
    digits = chart.get("digits", 2)
    if not categories:
        return ""
    left, right, top, bottom = 190, 60, 16, 26
    plot_width = max(width - left - right, 80)
    plot_height = max(height - top - bottom, 40)
    peak = max(values) or 1.0
    row_height = plot_height / max(len(categories), 1)
    bar_height = min(row_height * 0.55, 22.0)
    parts = [
        f'<svg class="chart" viewBox="0 0 {width} {height}" width="100%" '
        f'role="img" aria-label="{escape(chart["title"])}">'
    ]
    for index, (category, value) in enumerate(zip(categories, values)):
        y = top + index * row_height + (row_height - bar_height) / 2
        bar_width = max((value / peak) * plot_width, 2.0)
        parts.append(
            f'<text x="{left - 8}" y="{y + bar_height * 0.75:.1f}" '
            f'text-anchor="end" class="chart-label">{escape(str(category))}</text>'
        )
        parts.append(
            f'<rect x="{left}" y="{y:.1f}" width="{bar_width:.1f}" '
            f'height="{bar_height:.1f}" rx="3" fill="#2563eb" />'
        )
        parts.append(
            f'<text x="{left + bar_width + 6:.1f}" y="{y + bar_height * 0.75:.1f}" '
            f'class="chart-value">{value:.{digits}f}</text>'
        )
    parts.append(
        f'<text x="{left}" y="{height - 6}" class="chart-axis">'
        f"{escape(chart['unit'])}</text>"
    )
    parts.append("</svg>")
    return "".join(parts)


def _hardware_summaries(bundle: ReportBundle) -> dict[str, dict[str, Any]]:
    """Aggregate stored hardware rows per model, falling back to analytics averages."""
    grouped: dict[str, list[Any]] = {}
    for row in bundle.hardware_rows:
        grouped.setdefault(row.model_tag, []).append(row)
    summaries: dict[str, dict[str, Any]] = {}
    for model in bundle.analysis.models:
        rows = grouped.get(model.model_tag, [])
        notes = list(
            dict.fromkeys(
                note
                for row in rows
                for note in (row.gpu_error, row.sampler_error, row.attribution_note)
                if note
            )
        )
        summaries[model.model_tag] = {
            "responses": len(rows),
            "samples": sum(row.inference_sample_count for row in rows),
            "cpu_avg_percent": _mean([row.cpu_avg_percent for row in rows]),
            "cpu_peak_percent": _max([row.cpu_peak_percent for row in rows]),
            "ram_avg_bytes": _mean([row.ram_avg_used_bytes for row in rows]),
            "ram_peak_bytes": _max([row.ram_peak_used_bytes for row in rows]),
            "gpu_avg_percent": _mean([row.gpu_utilization_avg_percent for row in rows]),
            "gpu_peak_percent": _max([row.gpu_utilization_peak_percent for row in rows]),
            "vram_avg_bytes": _mean([row.vram_avg_used_bytes for row in rows]),
            "vram_peak_bytes": _max([row.vram_peak_used_bytes for row in rows]),
            "temperature_avg_c": _mean([row.gpu_temperature_avg_c for row in rows]),
            "temperature_peak_c": _max([row.gpu_temperature_peak_c for row in rows]),
            "gpu_name": next((row.gpu_name for row in rows if row.gpu_name), None),
            "notes": notes,
        }
    return summaries


def _limitations(bundle: ReportBundle) -> list[str]:
    """Collect and deduplicate every limitation that qualifies the report."""
    items: list[str] = list(bundle.limitations)
    items.extend(bundle.analysis.limitations)
    for model in bundle.analysis.models:
        items.extend(f"{model.model_tag}: {item}" for item in model.limitations)
    for hardware in _hardware_summaries(bundle).values():
        items.extend(hardware["notes"])
    if bundle.errors:
        items.append(
            f"{len(bundle.errors)} structured error record(s) are stored for this experiment"
        )
    if not bundle.hardware_rows:
        items.append("no hardware measurements are stored for the reported responses")
    return list(dict.fromkeys(item for item in items if item))


def build_view(bundle: ReportBundle) -> dict[str, Any]:
    """Build the formatted, sectioned report view shared by HTML and PDF."""
    analysis = bundle.analysis
    experiment = bundle.experiment
    dataset = {
        "name": experiment.dataset_name,
        "version": experiment.dataset_version,
        "hash": experiment.dataset_hash,
        "schema_version": bundle.detail.dataset_schema_version,
        "item_count": bundle.detail.dataset_item_count,
        "random_seed": bundle.detail.dataset_random_seed,
        "source_path": bundle.detail.dataset_source_path,
    }
    ranked = sorted(
        (model for model in analysis.models if model.composite_score is not None),
        key=lambda model: model.composite_score or 0.0,
        reverse=True,
    )
    fastest = min(
        (model for model in analysis.models if model.statistics.latency.p95 is not None),
        key=lambda model: model.statistics.latency.p95 or 0.0,
        default=None,
    )
    throughput_best = max(
        (
            model
            for model in analysis.models
            if model.statistics.output_tokens_per_second.mean is not None
        ),
        key=lambda model: model.statistics.output_tokens_per_second.mean or 0.0,
        default=None,
    )
    top = ranked[0] if ranked else None
    narrative = (
        f"{top.model_tag} ranks first with a composite score of "
        f"{_score(top.composite_score)} out of 100 under the {analysis.profile} profile. "
        f"The experiment stored {analysis.measured_responses} measured responses across "
        f"{len(analysis.models)} model(s); {analysis.evaluated_responses} were evaluated."
        if top
        else "No model has a composite score for the selected profile and filters."
    )
    hardware = _hardware_summaries(bundle)

    models_table = _table(
        _columns(
            ("tag", "Model tag", "left"),
            ("name", "Name", "left"),
            ("role", "Role", "left"),
            ("family", "Family", "left"),
            ("parameters", "Parameters", "left"),
            ("quantization", "Quantization", "left"),
            ("size", "Artifact size", "right"),
            ("context", "Context", "right"),
            ("digest", "Digest", "left"),
            ("installed", "Installed", "left"),
        ),
        [
            {
                "tag": model.tag,
                "name": model.name,
                "role": model.role,
                "family": model.family or MISSING,
                "parameters": model.parameter_size
                or (_text(model.parameter_count) if model.parameter_count else MISSING),
                "quantization": model.quantization or MISSING,
                "size": _bytes(model.size_bytes),
                "context": _text(model.context_length) if model.context_length else MISSING,
                "digest": (model.digest[:12] + "…") if model.digest else MISSING,
                "installed": "yes" if model.installed else "no",
            }
            for model in bundle.detail.models
        ],
    )

    performance_table = _table(
        _columns(
            ("model", "Model", "left"),
            ("measured", "Measured", "right"),
            ("successful", "Success", "right"),
            ("failed", "Failed", "right"),
            ("success_rate", "Success %", "right"),
            ("latency_mean", "Mean latency (s)", "right"),
            ("latency_median", "Median latency (s)", "right"),
            ("latency_p95", "P95 latency (s)", "right"),
            ("ttft", "Mean TTFT (s)", "right"),
            ("output_tps", "Output tok/s", "right"),
            ("prompt_tps", "Prompt tok/s", "right"),
        ),
        [
            {
                "model": model.model_tag,
                "measured": _text(model.statistics.measured_responses),
                "successful": _text(model.statistics.successful_responses),
                "failed": _text(model.statistics.failed_responses),
                "success_rate": _percent(model.statistics.success_rate),
                "latency_mean": _text(model.statistics.latency.mean),
                "latency_median": _text(model.statistics.latency.median),
                "latency_p95": _text(model.statistics.latency.p95),
                "ttft": _text(model.statistics.ttft.mean),
                "output_tps": _text(model.statistics.output_tokens_per_second.mean, 1),
                "prompt_tps": _text(model.statistics.prompt_tokens_per_second.mean, 1),
            }
            for model in analysis.models
        ],
    )

    quality_table = _table(
        _columns(
            ("model", "Model", "left"),
            ("evaluated", "Evaluated", "right"),
            ("measured", "Measured", "right"),
            ("coverage", "Quality coverage", "right"),
            ("judge_coverage", "Judge coverage", "right"),
            ("deterministic", "Deterministic (0–100)", "right"),
            ("judge", "LLM judge (0–100)", "right"),
            ("composite", "Composite (0–100)", "right"),
            ("failure_zero", "Failures as zero", "right"),
        ),
        [
            {
                "model": model.model_tag,
                "evaluated": _text(model.quality.evaluated_responses),
                "measured": _text(model.quality.measured_responses),
                "coverage": _percent(model.quality.quality_coverage),
                "judge_coverage": _percent(model.quality.judge_coverage),
                "deterministic": _score(model.quality.deterministic_score),
                "judge": _score(model.quality.judge_score),
                "composite": _score(model.quality.composite_score),
                "failure_zero": _score(model.quality.failure_as_zero_composite_score),
            }
            for model in analysis.models
        ],
    )

    category_rows: list[dict[str, str]] = []
    for model in analysis.models:
        for category in model.quality.categories:
            category_rows.append(
                {
                    "model": model.model_tag,
                    "category": category.category.replace("_", " ").title(),
                    "measured": _text(category.measured_responses),
                    "evaluated": _text(category.evaluated_responses),
                    "deterministic": _score(category.deterministic_score),
                    "judge": _score(category.judge_score),
                    "composite": _score(category.composite_score),
                    "coverage": _percent(category.quality_coverage),
                    "weight": _percent(category.category_weight),
                }
            )
    quality_category_table = _table(
        _columns(
            ("model", "Model", "left"),
            ("category", "Category", "left"),
            ("measured", "Measured", "right"),
            ("evaluated", "Evaluated", "right"),
            ("deterministic", "Deterministic (0–100)", "right"),
            ("judge", "LLM judge (0–100)", "right"),
            ("composite", "Composite (0–100)", "right"),
            ("coverage", "Coverage", "right"),
            ("weight", "Category weight", "right"),
        ),
        category_rows,
    )

    hardware_cpu_ram_table = _table(
        _columns(
            ("model", "Model", "left"),
            ("responses", "Responses", "right"),
            ("samples", "Samples", "right"),
            ("cpu_avg", "CPU avg %", "right"),
            ("cpu_peak", "CPU peak %", "right"),
            ("ram_avg", "RAM avg", "right"),
            ("ram_peak", "RAM peak", "right"),
        ),
        [
            {
                "model": model.model_tag,
                "responses": _text(hardware[model.model_tag]["responses"]),
                "samples": _text(hardware[model.model_tag]["samples"]),
                "cpu_avg": _text(hardware[model.model_tag]["cpu_avg_percent"], 1),
                "cpu_peak": _text(hardware[model.model_tag]["cpu_peak_percent"], 1),
                "ram_avg": _mib(hardware[model.model_tag]["ram_avg_bytes"]),
                "ram_peak": _mib(hardware[model.model_tag]["ram_peak_bytes"]),
            }
            for model in analysis.models
        ],
    )

    hardware_gpu_table = _table(
        _columns(
            ("model", "Model", "left"),
            ("gpu_avg", "GPU avg %", "right"),
            ("gpu_peak", "GPU peak %", "right"),
            ("vram_avg", "VRAM avg", "right"),
            ("vram_peak", "VRAM peak", "right"),
            ("temp_avg", "Temp avg °C", "right"),
            ("temp_peak", "Temp peak °C", "right"),
            ("gpu", "GPU", "left"),
        ),
        [
            {
                "model": model.model_tag,
                "gpu_avg": _text(hardware[model.model_tag]["gpu_avg_percent"], 1),
                "gpu_peak": _text(hardware[model.model_tag]["gpu_peak_percent"], 1),
                "vram_avg": _mib(hardware[model.model_tag]["vram_avg_bytes"]),
                "vram_peak": _mib(hardware[model.model_tag]["vram_peak_bytes"]),
                "temp_avg": _text(hardware[model.model_tag]["temperature_avg_c"], 1),
                "temp_peak": _text(hardware[model.model_tag]["temperature_peak_c"], 1),
                "gpu": hardware[model.model_tag]["gpu_name"] or MISSING,
            }
            for model in analysis.models
        ],
    )

    reliability_table = _table(
        _columns(
            ("model", "Model", "left"),
            ("measured", "Measured", "right"),
            ("successful", "Success", "right"),
            ("failed", "Failed", "right"),
            ("timeout", "Timeout", "right"),
            ("cancelled", "Cancelled", "right"),
            ("success_rate", "Success %", "right"),
            ("failure_rate", "Failure %", "right"),
            ("timeout_rate", "Timeout %", "right"),
            ("consistency", "Consistency (0–1)", "right"),
        ),
        [
            {
                "model": model.model_tag,
                "measured": _text(model.statistics.measured_responses),
                "successful": _text(model.statistics.successful_responses),
                "failed": _text(model.statistics.failed_responses),
                "timeout": _text(model.statistics.timed_out_responses),
                "cancelled": _text(model.statistics.cancelled_responses),
                "success_rate": _percent(model.statistics.success_rate),
                "failure_rate": _percent(model.statistics.failure_rate),
                "timeout_rate": _percent(model.statistics.timeout_rate),
                "consistency": _text(model.statistics.response_consistency, 3),
            }
            for model in analysis.models
        ],
    )

    weights_table = _table(
        _columns(
            ("dimension", "Dimension", "left"),
            ("weight", "Weight", "right"),
        ),
        [
            {"dimension": name.replace("_", " ").title(), "weight": f"{weight:g}%"}
            for name, weight in analysis.weights.items()
        ],
    )

    errors_table = _table(
        _columns(
            ("occurred_at", "Occurred (UTC)", "left"),
            ("stage", "Stage", "left"),
            ("error_type", "Type", "left"),
            ("message", "Message", "left"),
        ),
        [
            {
                "occurred_at": _timestamp(error.get("occurred_at")),
                "stage": str(error.get("stage") or ""),
                "error_type": str(error.get("error_type") or ""),
                "message": str(error.get("message") or ""),
            }
            for error in bundle.errors
        ],
    )

    environment_rows = [
        {"label": "Experiment ID", "value": experiment.experiment_id},
        {"label": "Experiment name", "value": experiment.name or MISSING},
        {"label": "Status", "value": experiment.status},
        {"label": "Created", "value": _timestamp(experiment.created_at)},
        {"label": "Started", "value": _timestamp(experiment.started_at)},
        {"label": "Completed", "value": _timestamp(experiment.completed_at)},
        {"label": "Configuration hash", "value": experiment.config_hash},
        {"label": "Dataset", "value": f"{dataset['name']} {dataset['version']}"},
        {"label": "Dataset hash", "value": dataset["hash"] or MISSING},
        {"label": "Dataset schema", "value": dataset["schema_version"] or MISSING},
        {"label": "Dataset items", "value": _text(dataset["item_count"])},
        {"label": "Dataset seed", "value": _text(dataset["random_seed"])},
        {"label": "Scoring profile", "value": f"{analysis.profile} — {analysis.profile_description}"},
        {"label": "Report generated", "value": _timestamp(bundle.generated_at)},
        {"label": "Report schema version", "value": REPORT_SCHEMA_VERSION},
    ]
    environment_rows.extend(_flatten("Host", bundle.detail.host))
    environment_rows.extend(_flatten("Software", bundle.detail.software))
    environment_rows.extend(_flatten("Configuration", bundle.detail.config))

    methodology = [
        "All values in this report are computed from rows stored in the benchmark "
        "SQLite database. No metric is hardcoded and no missing measurement is imputed.",
        "Performance metrics use successful measured responses only. Warm-up requests are "
        "excluded, and smoke and cold-start runs are excluded unless explicitly included.",
        "Latency is client-observed total latency in seconds; TTFT is time to first token "
        "in seconds; throughput is Ollama-reported output tokens per second.",
        "Quality combines deterministic checks and the blind LLM judge using the configured "
        "per-category judge weights. Deterministic failures are never converted into passes.",
        "Reliability rates use attempted requests excluding cancellations from the success, "
        "failure, and timeout denominators. A failure rate above 20% triggers the "
        "insufficient-reliability rule in recommendations.",
        "Hardware values are system-wide samples taken during each request and are not "
        "exclusively attributable to the model. Unavailable sensors are reported as missing.",
        "Composite scores use the configured workload profile weights and require at least "
        "two models for relative normalization of latency, throughput, and efficiency.",
    ]

    recommendations: list[dict[str, Any]] = []
    for result in bundle.recommendations:
        entries = [
            {
                "rank": item.rank,
                "model": item.model_tag,
                "score": _score(item.composite_score),
                "confidence": item.confidence,
                "evidence": _text(item.evidence_responses),
                "coverage": _percent(item.quality_coverage),
                "rationale": item.rationale,
                "strengths": item.strengths,
                "weaknesses": item.weaknesses,
                "tied_with": item.tied_with,
                "limitations": item.limitations,
            }
            for item in result.recommendations
        ]
        recommendations.append(
            {
                "workload": result.workload,
                "available": result.available,
                "reason": result.reason,
                "methodology_version": result.methodology_version,
                "evaluated_models": result.evaluated_models,
                "comparable_models": result.comparable_models,
                "entries": entries,
                "limitations": result.limitations,
            }
        )

    excerpts = []
    for inspection in bundle.inspections:
        if inspection.response_status != "success":
            continue
        text = inspection.response_text or ""
        excerpts.append(
            {
                "model": inspection.model_tag,
                "question": inspection.question_id,
                "text": text[:600] + ("…" if len(text) > 600 else ""),
            }
        )
        if len(excerpts) >= 2:
            break

    charts = _charts(analysis)
    for chart in charts:
        chart["svg"] = bar_chart_svg(chart)

    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "generated_at": _timestamp(bundle.generated_at),
        "preliminary": bundle.preliminary,
        "title": f"Benchmark Report — {experiment.name or experiment.experiment_id}",
        "subtitle": (
            f"Experiment {experiment.experiment_id} · dataset {dataset['name']} "
            f"{dataset['version']} · profile {analysis.profile}"
        ),
        "experiment": {
            "id": experiment.experiment_id,
            "name": experiment.name or MISSING,
            "status": experiment.status,
            "created_at": _timestamp(experiment.created_at),
            "started_at": _timestamp(experiment.started_at),
            "completed_at": _timestamp(experiment.completed_at),
            "config_hash": experiment.config_hash,
            "notes": bundle.detail.notes or MISSING,
            "dataset": dataset,
        },
        "summary": {
            "profile": analysis.profile,
            "profile_description": analysis.profile_description,
            "measured_responses": analysis.measured_responses,
            "evaluated_responses": analysis.evaluated_responses,
            "model_count": len(analysis.models),
            "run_count": len(bundle.detail.runs),
            "top_model": top.model_tag if top else MISSING,
            "top_score": _score(top.composite_score) if top else MISSING,
            "fastest_model": fastest.model_tag if fastest else MISSING,
            "fastest_p95": _text(fastest.statistics.latency.p95) if fastest else MISSING,
            "throughput_model": throughput_best.model_tag if throughput_best else MISSING,
            "throughput_rate": (
                _text(throughput_best.statistics.output_tokens_per_second.mean, 1)
                if throughput_best
                else MISSING
            ),
            "narrative": narrative,
            "key_metrics": [
                {
                    "label": "Measured responses",
                    "value": _text(analysis.measured_responses),
                    "detail": "Successful and failed measured responses after filtering.",
                },
                {
                    "label": "Evaluated responses",
                    "value": _text(analysis.evaluated_responses),
                    "detail": "Responses with a stored deterministic or judge evaluation.",
                },
                {
                    "label": "Models",
                    "value": _text(len(analysis.models)),
                    "detail": "Models with at least one measured response.",
                },
                {
                    "label": "Top composite",
                    "value": f"{top.model_tag} · {_score(top.composite_score)}" if top else MISSING,
                    "detail": f"Highest composite score under {analysis.profile}.",
                },
                {
                    "label": "Lowest P95 latency",
                    "value": (
                        f"{fastest.model_tag} · {_text(fastest.statistics.latency.p95)} s"
                        if fastest
                        else MISSING
                    ),
                    "detail": "Lower is better; P95 needs a large sample to be stable.",
                },
                {
                    "label": "Highest output rate",
                    "value": (
                        f"{throughput_best.model_tag} · "
                        f"{_text(throughput_best.statistics.output_tokens_per_second.mean, 1)} tok/s"
                        if throughput_best
                        else MISSING
                    ),
                    "detail": "Ollama-reported output tokens per second.",
                },
            ],
        },
        "environment": environment_rows,
        "methodology": {
            "paragraphs": methodology,
            "weights": weights_table,
            "filters": {
                "run_profile": bundle.run_profile or "all stored run profiles",
                "include_cold_start": "yes" if bundle.include_cold_start else "no",
                "include_smoke": "yes" if bundle.include_smoke else "no",
            },
        },
        "tables": {
            "models": models_table,
            "performance": performance_table,
            "quality": quality_table,
            "quality_categories": quality_category_table,
            "hardware_cpu_ram": hardware_cpu_ram_table,
            "hardware_gpu": hardware_gpu_table,
            "reliability": reliability_table,
            "weights": weights_table,
            "errors": errors_table,
        },
        "charts": charts,
        "recommendations": recommendations,
        "excerpts": excerpts,
        "limitations": _limitations(bundle),
    }


__all__ = ["MISSING", "bar_chart_svg", "build_view"]
