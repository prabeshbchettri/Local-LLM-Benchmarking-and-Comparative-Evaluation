"""
File: test_dashboard_data.py

Description:
Integration tests for the dashboard data layer and chart values.

Responsibilities:
- Verify empty and populated databases produce the expected read models.
- Verify dashboard numbers match the analytics engine and stored rows.
- Verify response inspection, evaluation breakdowns, and hardware aggregation.
- Verify recommendations, historical comparisons, and missing metrics.
"""

from __future__ import annotations

import pytest

from benchmark.analytics import describe
from benchmark.core.exceptions import BenchmarkError
from benchmark.dashboard import charts, data


def test_empty_database_has_no_experiments_or_analysis(dashboard_context) -> None:
    assert data.history(dashboard_context) == []
    assert data.completed_experiments(dashboard_context) == []
    assert data.latest_experiment(dashboard_context) is None
    assert data.registered_models(dashboard_context) == []
    with pytest.raises(BenchmarkError):
        data.analysis(dashboard_context, "missing-experiment")


def test_db_version_is_public_and_tracks_database_changes(
    dashboard_context, seed
) -> None:
    initial_version = data.db_version(dashboard_context)
    assert isinstance(initial_version, str)
    assert initial_version

    seed()

    assert data.db_version(dashboard_context) != initial_version


def test_dashboard_values_match_engine_and_stored_rows(
    dashboard_context, seed
) -> None:
    experiment_id, tags = seed()

    history = data.history(dashboard_context)
    assert [view.experiment_id for view in history] == [experiment_id]
    assert history[0].status == "completed"
    assert {model.tag for model in data.registered_models(dashboard_context)} == set(tags)

    analysis = data.analysis(dashboard_context, experiment_id)
    assert len(analysis.models) == 3
    assert analysis.measured_responses == 12
    assert analysis.evaluated_responses == 12
    assert analysis.experiment_status == "completed"

    rows = data.measured_rows(dashboard_context, experiment_id)
    assert len(rows) == analysis.measured_responses
    assert all(not row.cold_start for row in rows)

    for model in analysis.models:
        latencies = [
            row.latency_seconds
            for row in rows
            if row.model_tag == model.model_tag
            and row.response_status == "success"
            and row.latency_seconds is not None
        ]
        expected = describe(latencies)
        assert model.statistics.latency.mean == pytest.approx(expected.mean)
        assert model.statistics.latency.p95 == pytest.approx(expected.p95)
        assert model.statistics.latency.median == pytest.approx(expected.median)
        assert model.statistics.measured_responses == len(latencies)

    # Cold-start rows are stored but excluded by default.
    cold_rows = data.measured_rows(
        dashboard_context, experiment_id, include_cold_start=True
    )
    assert len(cold_rows) == analysis.measured_responses + 1
    assert any(row.cold_start for row in cold_rows)


def test_chart_traces_match_engine_values(dashboard_context, seed) -> None:
    experiment_id, _ = seed()
    analysis = data.analysis(dashboard_context, experiment_id)
    rows = data.measured_rows(dashboard_context, experiment_id)

    ttft = charts.ttft_comparison(analysis)
    assert ttft is not None
    mean_trace = next(trace for trace in ttft.data if trace.name == "Mean TTFT")
    assert list(mean_trace.y) == pytest.approx(
        [model.statistics.ttft.mean for model in analysis.models]
    )

    throughput = charts.throughput_comparison(analysis)
    assert throughput is not None
    mean_throughput = next(trace for trace in throughput.data if trace.name == "Mean")
    assert list(mean_throughput.y) == pytest.approx(
        [model.statistics.output_tokens_per_second.mean for model in analysis.models]
    )

    latency = charts.latency_distribution(rows)
    assert latency is not None
    for trace in latency.data:
        stored = sorted(
            row.latency_seconds
            for row in rows
            if row.model_tag == trace.name and row.latency_seconds is not None
        )
        assert sorted(trace.y) == pytest.approx(stored)

    quality = charts.quality_overall(analysis)
    assert quality is not None
    deterministic = next(trace for trace in quality.data if trace.name == "Deterministic")
    assert list(deterministic.y) == pytest.approx(
        [model.deterministic_score for model in analysis.models]
    )

    ranking = charts.composite_ranking(analysis)
    assert ranking is not None
    assert sorted(ranking.data[0].x, reverse=True) == pytest.approx(
        sorted(
            (
                model.composite_score
                for model in analysis.models
                if model.composite_score is not None
            ),
            reverse=True,
        )
    )


def test_response_inspection_exposes_prompt_answer_and_evaluation(
    dashboard_context, seed
) -> None:
    experiment_id, tags = seed()
    question_id = "code-001"
    details = data.inspections(
        dashboard_context,
        experiment_id,
        model_tag=tags[0],
        question_id=question_id,
    )
    assert len(details) == 1
    detail = details[0]
    item = data.dataset_items(dashboard_context)[question_id]
    assert detail.prompt == item.prompt
    assert detail.response_text == f"{tags[0]} seeded answer for {question_id}"
    assert detail.response_status == "success"
    assert detail.latency_seconds is not None
    assert detail.ttft_seconds is not None
    assert detail.prompt_eval_count == 40
    assert detail.eval_count == 25
    assert detail.output_tokens_per_second is not None
    assert detail.evaluation_status == "passed"
    assert detail.evaluation_type == "code_tests"
    assert detail.checks and detail.checks[0]["name"] == "code_tests"
    assert detail.judge_dimensions
    assert detail.hardware_summary is not None
    assert detail.hardware_summary["gpu_name"] == "Test GPU"


def test_evaluation_breakdown_reports_passes_by_method(
    dashboard_context, seed
) -> None:
    experiment_id, tags = seed()
    rows = data.evaluation_breakdown(dashboard_context, experiment_id)
    assert rows
    code_rows = [row for row in rows if row.evaluation_type == "code_tests"]
    json_rows = [row for row in rows if row.evaluation_type == "json_schema"]
    instruction_rows = [
        row for row in rows if row.evaluation_type == "instruction_constraints"
    ]
    assert {row.model_tag for row in code_rows} == set(tags)
    assert all(row.evaluated_responses == 1 for row in code_rows)
    assert all(0 <= row.passed_responses <= 1 for row in code_rows)
    assert json_rows and instruction_rows

    rates = charts.evaluation_type_rates(
        rows,
        None,
        ["code_tests", "json_schema", "instruction_constraints"],
    )
    assert rates is not None
    assert "Coding tests" in set(rates.data[0].x)


def test_hardware_summary_reports_average_and_peak(
    dashboard_context, seed
) -> None:
    experiment_id, tags = seed()
    rows = data.hardware_rows(dashboard_context, experiment_id)
    assert len(rows) == 12
    assert {row.model_tag for row in rows} == set(tags)
    summaries = data.hardware_summary_rows(rows)
    assert len(summaries) == 3
    first = summaries[0]
    assert first["responses"] == 4
    assert first["cpu_avg_percent"] == pytest.approx(20.0)
    assert first["cpu_peak_percent"] == pytest.approx(35.0)
    assert first["cpu_peak_percent"] >= first["cpu_avg_percent"]
    assert first["ram_avg_mib"] == pytest.approx(2048.0)
    assert first["ram_peak_mib"] == pytest.approx(3072.0)
    assert first["vram_avg_mib"] == pytest.approx(1024.0)
    assert first["temperature_peak_c"] == pytest.approx(45.0)
    assert first["notes"] == []

    chart = charts.hardware_avg_peak(
        summaries, "cpu_avg_percent", "cpu_peak_percent", "CPU", "%"
    )
    assert chart is not None
    average = next(trace for trace in chart.data if trace.name == "Average")
    peak = next(trace for trace in chart.data if trace.name == "Peak")
    assert list(average.y) == pytest.approx(
        [summary["cpu_avg_percent"] for summary in summaries]
    )
    assert max(peak.y) >= max(average.y)


def test_recommendations_use_measured_engine_scores(
    dashboard_context, seed
) -> None:
    experiment_id, _ = seed()
    sets = data.recommendations(dashboard_context, experiment_id)
    assert len(sets) == len(data.profile_names(dashboard_context))
    for result in sets:
        assert result.available is True
        assert result.recommendations
        scores = [item.composite_score for item in result.recommendations]
        assert scores == sorted(scores, reverse=True)
    top = sets[0].recommendations[0]
    analysis = data.analysis(
        dashboard_context,
        experiment_id,
        profile=top.workload,
    )
    engine_model = analysis.model(top.model_tag)
    assert engine_model is not None
    assert top.composite_score == pytest.approx(engine_model.composite_score)
    assert top.evidence_responses > 0
    assert top.rationale


def test_historical_comparison_reports_measured_deltas(
    dashboard_context, seed
) -> None:
    baseline_id, _ = seed(experiment_id="exp-baseline")
    candidate_id, _ = seed(
        experiment_id="exp-candidate",
        latency_shift=1.0,
        score_shift=5.0,
    )
    comparison = data.comparison(
        dashboard_context, baseline_id, [candidate_id]
    )
    assert comparison.baseline_experiment_id == baseline_id
    assert len(comparison.comparisons) == 1
    item = comparison.comparisons[0]
    assert item.candidate_experiment_id == candidate_id
    assert len(item.models) == 3
    for model in item.models:
        metrics = {metric.metric: metric for metric in model.metrics}
        assert metrics["p95_latency_seconds"].delta == pytest.approx(1.0)
        assert metrics["mean_ttft_seconds"].delta == pytest.approx(0.0)
        assert metrics["quality_score"].delta == pytest.approx(5.0)
        assert metrics["quality_score"].percent_change is not None

    chart = charts.comparison_changes(item)
    assert chart is not None
    assert len(chart.data) == 3


def test_missing_metrics_stay_missing_instead_of_zero(
    dashboard_context, seed
) -> None:
    experiment_id, tags = seed(
        experiment_id="exp-sparse",
        with_hardware=False,
        with_evaluations=False,
        with_cold_start=False,
    )
    analysis = data.analysis(dashboard_context, experiment_id)
    assert analysis.measured_responses == 12
    assert analysis.evaluated_responses == 0
    assert all(model.quality_score is None for model in analysis.models)
    assert all(model.composite_score is None for model in analysis.models)
    assert all(
        model.statistics.success_rate == 1.0 for model in analysis.models
    )
    assert all(not model.resources.cpu_available for model in analysis.models)
    assert data.hardware_rows(dashboard_context, experiment_id) == []
    assert data.evaluation_breakdown(dashboard_context, experiment_id) == []

    sets = data.recommendations(dashboard_context, experiment_id)
    assert all(result.available is False for result in sets)
    assert all(result.reason for result in sets)

    details = data.inspections(
        dashboard_context,
        experiment_id,
        model_tag=tags[0],
        question_id="gk-001",
    )
    assert details[0].evaluation_status is None
    assert details[0].hardware_summary is None
    assert charts.quality_overall(analysis) is None
    assert charts.latency_distribution(
        data.measured_rows(dashboard_context, experiment_id)
    ) is not None


def test_cold_start_chart_uses_stored_cold_rows(dashboard_context, seed) -> None:
    experiment_id, _ = seed()
    rows = data.measured_rows(
        dashboard_context, experiment_id, include_cold_start=True
    )
    figure = charts.cold_warm_comparison(
        rows, "latency_seconds", "Total latency (s)"
    )
    assert figure is not None
    cold = next(trace for trace in figure.data if trace.name == "Cold start")
    assert max(cold.y) == pytest.approx(9.5)
