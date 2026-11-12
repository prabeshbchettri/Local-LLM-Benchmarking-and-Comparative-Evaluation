"""
Hardware monitoring page.

Responsibilities:
- Show CPU, RAM, GPU, VRAM, and GPU temperature usage.
- Show average and peak consumption per model.
- Show per-response resource timelines and availability notes.
"""

from __future__ import annotations

import streamlit as st

from benchmark.core.exceptions import BenchmarkError
from benchmark.dashboard import charts, data, ui

SERIES_OPTIONS = {
    "CPU utilization": ("cpu_avg_percent", "CPU utilization", "%", 1.0),
    "RAM used": ("ram_avg_used_bytes", "RAM used", "GiB", 1024 * 1024 * 1024),
    "GPU utilization": (
        "gpu_utilization_avg_percent",
        "GPU utilization",
        "%",
        1.0,
    ),
    "VRAM used": ("vram_avg_used_bytes", "VRAM used", "GiB", 1024 * 1024 * 1024),
    "GPU temperature": (
        "gpu_temperature_avg_c",
        "GPU temperature",
        "°C",
        1.0,
    ),
}


def render(context: data.DashboardContext) -> None:
    """Render the hardware monitoring page."""
    ui.page_header(
        "Hardware Monitoring",
        "System-wide resource usage sampled during each stored inference request.",
    )
    view = ui.select_experiment(context, "hardware_experiment")
    if view is None:
        return
    include_cold = st.toggle(
        "Include cold-start runs",
        value=False,
        key="hardware_cold",
        help="Cold-start runs include model load time and are excluded by default.",
    )
    try:
        with st.spinner("Loading hardware samples…"):
            rows = data.hardware_rows(
                context,
                view.experiment_id,
                include_cold_start=include_cold,
            )
    except BenchmarkError as exc:
        st.warning(f"Hardware measurements are unavailable: {exc}")
        return
    if not rows:
        ui.empty_state(
            "No hardware measurements",
            "This experiment has no stored CPU, RAM, GPU, VRAM, or temperature "
            "samples for the requested filters.",
        )
        return

    summaries = data.hardware_summary_rows(rows)
    gpu_name = next(
        (row.gpu_name for row in rows if row.gpu_name),
        "GPU not identified",
    )
    peak_cpu = max(
        (
            row.cpu_peak_percent
            for row in rows
            if row.cpu_peak_percent is not None
        ),
        default=None,
    )
    peak_temperature = max(
        (
            row.gpu_temperature_peak_c
            for row in rows
            if row.gpu_temperature_peak_c is not None
        ),
        default=None,
    )
    ui.metric_row(
        [
            (
                "Monitored responses",
                ui.format_count(len(rows)),
                "Stored responses with a hardware summary.",
            ),
            (
                "Inference samples",
                ui.format_count(sum(row.inference_sample_count for row in rows)),
                "Samples collected during inference windows.",
            ),
            (
                "Peak CPU",
                f"{peak_cpu:.1f}%" if peak_cpu is not None else "—",
                "Highest per-response CPU peak.",
            ),
            (
                "Peak GPU temperature",
                (
                    f"{peak_temperature:.1f} °C"
                    if peak_temperature is not None
                    else "—"
                ),
                "Highest per-response GPU temperature peak.",
            ),
        ]
    )
    st.caption(f"GPU: {gpu_name} · sampling interval varies per stored summary.")
    st.dataframe(
        ui.clean_frame(
            [
                {
                    "Model": row["model_tag"],
                    "Responses": row["responses"],
                    "Samples": row["samples"],
                    "CPU avg %": row["cpu_avg_percent"],
                    "CPU peak %": row["cpu_peak_percent"],
                    "RAM avg MiB": row["ram_avg_mib"],
                    "RAM peak MiB": row["ram_peak_mib"],
                    "GPU avg %": row["gpu_avg_percent"],
                    "GPU peak %": row["gpu_peak_percent"],
                    "VRAM avg MiB": row["vram_avg_mib"],
                    "VRAM peak MiB": row["vram_peak_mib"],
                    "Temp avg °C": row["temperature_avg_c"],
                    "Temp peak °C": row["temperature_peak_c"],
                }
                for row in summaries
            ]
        ),
        width="stretch",
        hide_index=True,
        column_config={
            "CPU avg %": st.column_config.NumberColumn(format="%.1f"),
            "CPU peak %": st.column_config.NumberColumn(format="%.1f"),
            "RAM avg MiB": st.column_config.NumberColumn(format="%.0f"),
            "RAM peak MiB": st.column_config.NumberColumn(format="%.0f"),
            "GPU avg %": st.column_config.NumberColumn(format="%.1f"),
            "GPU peak %": st.column_config.NumberColumn(format="%.1f"),
            "VRAM avg MiB": st.column_config.NumberColumn(format="%.0f"),
            "VRAM peak MiB": st.column_config.NumberColumn(format="%.0f"),
            "Temp avg °C": st.column_config.NumberColumn(format="%.1f"),
            "Temp peak °C": st.column_config.NumberColumn(format="%.1f"),
        },
    )

    st.write("")
    ui.section(
        "Average and peak resource consumption",
        "Averages are the mean of per-response averages; peaks are the highest "
        "per-response peak.",
    )
    left, right = st.columns(2)
    with left:
        chart = charts.hardware_avg_peak(
            summaries, "cpu_avg_percent", "cpu_peak_percent", "CPU utilization", "%"
        )
        st.markdown("**CPU utilization**")
        if chart is None:
            st.info("CPU usage was not measured.")
        else:
            st.plotly_chart(
                chart, width="stretch", config={"displayModeBar": False}
            )
    with right:
        chart = charts.hardware_avg_peak(
            summaries,
            "ram_avg_mib",
            "ram_peak_mib",
            "RAM used",
            "GiB",
            divisor=1024.0,
        )
        st.markdown("**RAM usage**")
        if chart is None:
            st.info("RAM usage was not measured.")
        else:
            st.plotly_chart(
                chart, width="stretch", config={"displayModeBar": False}
            )
    left, right = st.columns(2)
    with left:
        chart = charts.hardware_avg_peak(
            summaries,
            "gpu_avg_percent",
            "gpu_peak_percent",
            "GPU utilization",
            "%",
        )
        st.markdown("**GPU utilization**")
        if chart is None:
            st.info("GPU utilization was not measured.")
        else:
            st.plotly_chart(
                chart, width="stretch", config={"displayModeBar": False}
            )
    with right:
        chart = charts.hardware_avg_peak(
            summaries,
            "vram_avg_mib",
            "vram_peak_mib",
            "VRAM used",
            "GiB",
            divisor=1024.0,
        )
        st.markdown("**VRAM usage**")
        if chart is None:
            st.info("System VRAM usage was not measured.")
        else:
            st.plotly_chart(
                chart, width="stretch", config={"displayModeBar": False}
            )
    chart = charts.hardware_avg_peak(
        summaries,
        "temperature_avg_c",
        "temperature_peak_c",
        "GPU temperature",
        "°C",
    )
    st.markdown("**GPU temperature**")
    if chart is None:
        st.info("GPU temperature was not measured.")
    else:
        st.plotly_chart(
            chart, width="stretch", config={"displayModeBar": False}
        )

    st.write("")
    ui.section(
        "Per-response timeline",
        "Each point is one stored response, ordered by question and repetition.",
    )
    metric_name = st.selectbox(
        "Timeline metric",
        options=list(SERIES_OPTIONS),
        key="hardware_series_metric",
    )
    key, label, unit, divisor = SERIES_OPTIONS[metric_name]
    series = charts.hardware_series(
        rows,
        key,
        label,
        unit,
        divisor=divisor,
    )
    if series is None:
        st.info(f"{label} was not measured.")
    else:
        st.plotly_chart(
            series, width="stretch", config={"displayModeBar": False}
        )

    availability = [
        note
        for note in dict.fromkeys(
            [row.gpu_error or "" for row in rows] + [row.sampler_error or "" for row in rows]
        )
        if note
    ]
    if availability:
        with st.expander("GPU and sampler availability notes"):
            for item in availability:
                st.markdown(f"- {item}")
    attribution = next(
        (row.attribution_note for row in rows if row.attribution_note),
        None,
    )
    if attribution:
        st.caption(attribution)
    st.caption(
        "Unavailable values are shown as blank, never as zero. Windows WDDM may "
        "report GPU utilization or temperature as unavailable."
    )


__all__ = ["render"]
