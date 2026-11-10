"""
File: test_database.py

Description:
Unit tests for SQLite schema creation, repositories, integrity rules, and retrieval.

Responsibilities:
- Verify schema initialization, pragmas, and version tracking.
- Verify dataset/model/experiment/run/response/evaluation CRUD.
- Verify foreign-key enforcement, transaction rollback, and duplicate protection.
- Verify failed requests, history, and persistence across reopen.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlalchemy.exc import IntegrityError

from benchmark.core.exceptions import (
    DatabaseConflictError,
    DatabaseDuplicateError,
    ExperimentStateError,
)
from benchmark.database.db import Database
from benchmark.database.migrations import SCHEMA_VERSION
from benchmark.database.repositories import RecommendationInput, Repositories
from benchmark.database.tables import ExperimentRow
from benchmark.dataset.loader import load_dataset
from benchmark.evaluation.engine import EvaluationEngine, EvaluationRequest
from benchmark.execution.metadata import ModelRole, build_missing_metadata
from benchmark.models.inference import (
    GenerationOptions,
    InferenceProfile,
    ResponseRecord,
    ResponseStatus,
    TimingMetrics,
    TokenMetrics,
)

pytestmark = pytest.mark.unit


def _record(
    item_id: str = "gk-001",
    *,
    text: str = "Paris",
    status: ResponseStatus = ResponseStatus.SUCCESS,
    repetition: int = 1,
    is_warmup: bool = False,
) -> ResponseRecord:
    now = datetime.now(timezone.utc)
    failed = status is not ResponseStatus.SUCCESS
    return ResponseRecord(
        experiment_id="exp-test",
        model_tag="llama3.1:8b",
        item_id=item_id,
        category="general_knowledge",
        profile=InferenceProfile.QUALITY,
        repetition=repetition,
        is_warmup=is_warmup,
        prompt="prompt",
        response_text="" if failed else text,
        status=status,
        error_type="timeout" if failed else None,
        error_message="request timed out" if failed else None,
        started_at=now,
        completed_at=now,
        generation_options=GenerationOptions(),
        timing=TimingMetrics(ttft_seconds=0.2, total_latency_seconds=1.5),
        tokens=TokenMetrics(
            prompt_eval_count=5,
            eval_count=10,
            output_tokens_per_second=20.0,
        ),
    )


@pytest.fixture()
def database(temp_settings):
    db = Database.from_settings(temp_settings)
    db.initialize()
    yield db
    db.dispose()


@pytest.fixture()
def repos(database):
    return Repositories(database)


@pytest.fixture()
def dataset(dataset_path):
    return load_dataset(dataset_path)


def _experiment(repos, dataset, *, experiment_id: str = "exp-001"):
    dataset_id = repos.datasets.register(dataset)
    return repos.experiments.create(
        experiment_id=experiment_id,
        name="unit",
        dataset_id=dataset_id,
        config_hash="a" * 64,
        config={"benchmark": "test"},
        host={"platform": "test"},
        software={"python": "3"},
    )


def _model(repos, temp_settings, index: int = 0):
    spec = temp_settings.models.candidates[index]
    return repos.models.register(
        build_missing_metadata(spec, "0.40.15", ModelRole.CANDIDATE)
    )


def _run(repos, dataset, temp_settings, *, experiment_id: str = "exp-001"):
    _experiment(repos, dataset, experiment_id=experiment_id)
    model = _model(repos, temp_settings)
    return repos.runs.create(
        experiment_external_id=experiment_id,
        model_id=model.id,
        profile="quality",
    )


def test_initialization_creates_tables_and_pragmas(database, repos) -> None:
    assert database.foreign_keys_enabled() is True
    assert database.journal_mode().lower() == "wal"
    assert set(database.table_names()) == {
        "datasets",
        "errors",
        "evaluations",
        "experiments",
        "hardware_measurements",
        "models",
        "performance_metrics",
        "recommendations",
        "responses",
        "runs",
        "schema_versions",
    }
    assert database.initialize() == SCHEMA_VERSION


def test_dataset_registration_rejects_changed_content(repos, dataset) -> None:
    repos.datasets.register(dataset)
    changed = dataset.model_copy(update={"content_sha256": "b" * 64})
    with pytest.raises(DatabaseConflictError):
        repos.datasets.register(changed)


def test_response_metrics_evaluation_and_retrieval(repos, dataset, temp_settings) -> None:
    run = _run(repos, dataset, temp_settings)
    record = _record()
    response_id = repos.responses.save(record, run.id)
    assert repos.responses.save(record, run.id) == response_id

    changed = _record(text="London")
    with pytest.raises(DatabaseDuplicateError):
        repos.responses.save(changed, run.id)

    assert repos.responses.existing_keys(run.id) == {("gk-001", 1, False)}
    pending = repos.responses.pending_evaluations(run.id)
    assert [item.question_id for item in pending] == ["gk-001"]

    item = next(entry for entry in dataset.items if entry.id == "gk-001")
    result = EvaluationEngine().evaluate(
        EvaluationRequest(item=item, response_text="Paris", model_id=record.model_tag)
    )
    evaluation_id = repos.evaluations.save(result, response_id)
    assert repos.evaluations.save(result, response_id) == evaluation_id
    assert repos.responses.pending_evaluations(run.id) == []

    questions = repos.reads.question_results("exp-001", "gk-001")
    assert len(questions) == 1
    assert questions[0].status == "success"
    assert questions[0].total_latency_seconds == pytest.approx(1.5)
    assert questions[0].deterministic_normalized_score == pytest.approx(
        result.deterministic_normalized_score
    )

    results = repos.reads.model_results("exp-001")
    assert len(results) == 1
    assert results[0].successful_responses == 1
    assert results[0].failed_responses == 0
    assert results[0].mean_latency_seconds == pytest.approx(1.5)
    assert results[0].mean_output_tokens_per_second == pytest.approx(20.0)


def test_failed_request_is_stored_and_evaluated_as_skipped(
    repos, dataset, temp_settings
) -> None:
    run = _run(repos, dataset, temp_settings)
    response_id = repos.responses.save(
        _record(status=ResponseStatus.TIMEOUT),
        run.id,
    )
    item = next(entry for entry in dataset.items if entry.id == "gk-001")
    result = EvaluationEngine().evaluate(
        EvaluationRequest(
            item=item,
            response_text="",
            model_id="llama3.1:8b",
            response_status=ResponseStatus.TIMEOUT,
        )
    )
    assert result.status.value == "skipped"
    repos.evaluations.save(result, response_id)

    questions = repos.reads.question_results("exp-001")
    assert questions[0].status == "timeout"
    assert questions[0].response_text == ""
    assert questions[0].deterministic_normalized_score is None
    results = repos.reads.model_results("exp-001")
    assert (results[0].successful_responses, results[0].failed_responses) == (0, 1)


def test_transaction_rolls_back_on_error(database, repos, dataset) -> None:
    with pytest.raises(RuntimeError):
        with database.session_scope() as session:
            session.add(
                ExperimentRow(
                    experiment_id="rolled-back",
                    dataset_id=repos.datasets.register(dataset),
                    config_hash="a" * 64,
                    status="created",
                    created_at=datetime.now(timezone.utc),
                    config_json={},
                    host_json={},
                    software_json={},
                )
            )
            raise RuntimeError("boom")
    assert repos.experiments.get("rolled-back") is None


def test_foreign_key_violation_is_rejected(database, repos, dataset) -> None:
    dataset_id = repos.datasets.register(dataset)
    with pytest.raises(IntegrityError):
        with database.session_scope() as session:
            session.add(
                ExperimentRow(
                    experiment_id="orphan",
                    dataset_id=dataset_id + 999,
                    config_hash="a" * 64,
                    status="created",
                    created_at=datetime.now(timezone.utc),
                    config_json={},
                    host_json={},
                    software_json={},
                )
            )
    assert repos.experiments.get("orphan") is None


def test_cold_start_and_repeated_creation_use_run_identity(
    repos, dataset, temp_settings
) -> None:
    _experiment(repos, dataset)
    model = _model(repos, temp_settings)
    warm = repos.runs.create(
        experiment_external_id="exp-001",
        model_id=model.id,
        profile="quality",
        cold_start=False,
    )
    cold = repos.runs.create(
        experiment_external_id="exp-001",
        model_id=model.id,
        profile="quality",
        cold_start=True,
    )
    repeated = repos.runs.create(
        experiment_external_id="exp-001",
        model_id=model.id,
        profile="quality",
        cold_start=False,
    )
    assert warm.id != cold.id
    assert repeated.id == warm.id
    assert len(repos.reads.model_results("exp-001")) == 2


def test_completed_history_rejects_new_runs_and_responses(
    repos, dataset, temp_settings
) -> None:
    run = _run(repos, dataset, temp_settings)
    repos.responses.save(_record(), run.id)
    repos.runs.mark_completed(run.id)
    repos.experiments.mark_completed("exp-001")

    with pytest.raises(ExperimentStateError):
        repos.runs.create(
            experiment_external_id="exp-001",
            model_id=run.id,
            profile="consistency",
        )
    with pytest.raises(ExperimentStateError):
        repos.responses.save(_record(item_id="mr-001"), run.id)
    assert repos.experiments.get("exp-001").status == "completed"


def test_history_persists_across_reopen(temp_settings, dataset) -> None:
    first = Database.from_settings(temp_settings)
    first.initialize()
    repos = Repositories(first)
    _experiment(repos, dataset)
    path = first.path
    first.dispose()

    second = Database(path)
    second.initialize()
    try:
        history = Repositories(second).experiments.history()
        assert [view.experiment_id for view in history] == ["exp-001"]
        assert history[0].dataset_name == dataset.name
    finally:
        second.dispose()


def test_recommendations_are_ranked_and_duplicate_safe(repos, dataset, temp_settings) -> None:
    _experiment(repos, dataset)
    first = _model(repos, temp_settings, 0)
    second = _model(repos, temp_settings, 1)
    recommendations = [
        RecommendationInput(
            workload="coding",
            rank=1,
            model_tag=first.tag,
            score=88.0,
            confidence="high",
            weights={"quality": 0.6, "performance": 0.4},
            rationale="Best measured coding score",
        ),
        RecommendationInput(
            workload="coding",
            rank=2,
            model_tag=second.tag,
            score=81.0,
            confidence="medium",
            weights={"quality": 0.6, "performance": 0.4},
            rationale="Runner-up",
        ),
    ]
    assert repos.recommendations.save_many("exp-001", recommendations) == 2
    assert repos.recommendations.save_many("exp-001", recommendations) == 0
    conflicting = [
        recommendations[0].model_copy(update={"score": 99.0})
    ]
    with pytest.raises(DatabaseDuplicateError):
        repos.recommendations.save_many("exp-001", conflicting)
    stored = repos.recommendations.list_for_experiment("exp-001")
    assert [row["model_tag"] for row in stored] == [first.tag, second.tag]


def test_errors_are_appended_with_their_stage(repos, dataset) -> None:
    _experiment(repos, dataset)
    repos.errors.record(
        stage="orchestrator",
        error_type="ModelLifecycleError",
        message="model failed to load",
        experiment_external_id="exp-001",
    )
    errors = repos.errors.list_for_experiment("exp-001")
    assert len(errors) == 1
    assert errors[0]["stage"] == "orchestrator"
    assert errors[0]["error_type"] == "ModelLifecycleError"
