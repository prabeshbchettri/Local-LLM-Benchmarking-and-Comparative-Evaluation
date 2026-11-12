"""
File: test_analytics_cli.py

Description:
Integration tests for the analytics and comparison CLI commands.

Responsibilities:
- Verify analyze emits JSON, persists recommendations, and reads the configured DB.
- Verify compare reports metric deltas between stored experiments.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from benchmark.__main__ import main
from benchmark.database.db import Database
from benchmark.database.repositories import Repositories
from benchmark.dataset.loader import load_dataset
from benchmark.dataset.models import EvaluationType
from benchmark.evaluation.models import EvaluationResult, EvaluationStatus
from benchmark.execution.metadata import ModelRole, build_missing_metadata
from benchmark.models.inference import (
    GenerationOptions,
    InferenceProfile,
    ResponseRecord,
    ResponseStatus,
    TimingMetrics,
    TokenMetrics,
)

pytestmark = pytest.mark.integration


def _seed(temp_settings, dataset_path, experiment_id: str, latency: float) -> None:
    database = Database.from_settings(temp_settings)
    database.initialize()
    repos = Repositories(database)
    dataset = load_dataset(dataset_path)
    dataset_id = repos.datasets.register(dataset)
    repos.experiments.create(
        experiment_id=experiment_id,
        name=experiment_id,
        dataset_id=dataset_id,
        config_hash="d" * 64,
        config={"benchmark": "test"},
        host={"platform": "test"},
        software={"python": "3.13"},
    )
    for index, spec in enumerate(temp_settings.models.candidates[:2]):
        model = repos.models.register(
            build_missing_metadata(spec, "0.40.15", ModelRole.CANDIDATE)
        )
        run = repos.runs.create(
            experiment_external_id=experiment_id,
            model_id=model.id,
            profile="quality",
        )
        now = datetime.now(timezone.utc)
        record = ResponseRecord(
            experiment_id=experiment_id,
            model_tag=model.tag,
            model_name=model.tag,
            item_id="gk-001",
            category="general_knowledge",
            profile=InferenceProfile.QUALITY,
            repetition=1,
            prompt="prompt",
            response_text="answer",
            status=ResponseStatus.SUCCESS,
            started_at=now,
            completed_at=now,
            generation_options=GenerationOptions(),
            timing=TimingMetrics(
                ttft_seconds=latency / 4,
                total_latency_seconds=latency + index,
            ),
            tokens=TokenMetrics(
                prompt_eval_count=5,
                eval_count=10,
                output_tokens_per_second=100.0 / (latency + index),
            ),
        )
        response_id = repos.responses.save(record, run.id)
        repos.evaluations.save(
            EvaluationResult(
                question_id="gk-001",
                model_id=model.tag,
                category="general_knowledge",
                evaluation_type=EvaluationType.EXACT_MATCH,
                deterministic_normalized_score=90.0 - index * 30.0,
                composite_score=90.0 - index * 30.0,
                status=EvaluationStatus.PASSED,
                judge_missing=True,
            ),
            response_id,
        )
        repos.runs.mark_completed(run.id)
    repos.experiments.mark_completed(experiment_id)
    database.dispose()


def test_analyze_cli_outputs_and_persists(
    temp_settings,
    temp_config_dir,
    dataset_path,
    capsys,
) -> None:
    _seed(temp_settings, dataset_path, "exp-cli-a", latency=2.0)

    exit_code = main(
        [
            "analyze",
            "exp-cli-a",
            "--json",
            "--persist",
            "--config-dir",
            str(temp_config_dir),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0, captured.err
    payload = json.loads(captured.out)
    assert payload["experiment_id"] == "exp-cli-a"
    assert len(payload["profiles"]) == 1
    profile = payload["profiles"][0]
    assert profile["analytics"]["profile"] == "general_purpose"
    assert profile["analytics"]["measured_responses"] == 2
    assert profile["recommendation"]["available"] is True
    assert len(profile["recommendation"]["recommendations"]) == 2

    database = Database.from_settings(temp_settings)
    try:
        stored = Repositories(database).recommendations.list_for_experiment(
            "exp-cli-a"
        )
        assert len(stored) == 2
        assert stored[0]["details"] is not None
    finally:
        database.dispose()


def test_compare_cli_reports_deltas(
    temp_settings,
    temp_config_dir,
    dataset_path,
    capsys,
) -> None:
    _seed(temp_settings, dataset_path, "exp-cli-a", latency=2.0)
    _seed(temp_settings, dataset_path, "exp-cli-b", latency=3.0)

    exit_code = main(
        [
            "compare",
            "exp-cli-a",
            "exp-cli-b",
            "--json",
            "--config-dir",
            str(temp_config_dir),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0, captured.err
    payload = json.loads(captured.out)
    assert payload["baseline_experiment_id"] == "exp-cli-a"
    comparison = payload["comparisons"][0]
    assert comparison["candidate_experiment_id"] == "exp-cli-b"
    model = comparison["models"][0]
    metrics = {item["metric"]: item for item in model["metrics"]}
    assert metrics["p95_latency_seconds"]["delta"] == pytest.approx(1.0)
    assert metrics["quality_score"]["delta"] == pytest.approx(0.0)
