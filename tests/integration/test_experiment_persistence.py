"""
File: test_experiment_persistence.py

Description:
Integration tests for a full persisted experiment through the orchestrator.

Responsibilities:
- Verify every response, metric, and evaluation reaches the database.
- Verify warm-up requests are stored but excluded from measured aggregates.
- Verify interrupted experiments resume without repeating stored requests.
- Verify completed experiments and their history survive a reopen.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from benchmark.core.exceptions import BenchmarkError, ExperimentStateError
from benchmark.database.db import open_database
from benchmark.database.repositories import Repositories
from benchmark.dataset.loader import load_dataset
from benchmark.execution.metadata import ModelRole, ModelStatus
from benchmark.execution.orchestrator import ExperimentOrchestrator
from benchmark.models.inference import (
    GenerationOptions,
    ResponseRecord,
    ResponseStatus,
    TimingMetrics,
    TokenMetrics,
)

pytestmark = pytest.mark.integration

ITEM_IDS = ["gk-001", "mr-001"]


def _success_record(model_tag: str, prompt: str, kwargs: dict) -> ResponseRecord:
    now = datetime.now(timezone.utc)
    return ResponseRecord(
        experiment_id=kwargs["experiment_id"],
        model_tag=model_tag,
        item_id=kwargs["item_id"],
        category=kwargs["category"],
        profile=kwargs["profile"],
        repetition=kwargs["repetition"],
        is_warmup=kwargs["is_warmup"],
        prompt=prompt,
        response_text="A stored answer",
        status=ResponseStatus.SUCCESS,
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


class FakeEngine:
    """Deterministic engine that records calls and can interrupt once."""

    interrupted = False

    def __init__(self, *, interrupt_on_call: int | None = None) -> None:
        self.calls: list[tuple[str, int, bool]] = []
        self._interrupt_on_call = interrupt_on_call

    def run_request(self, model_tag: str, prompt: str, **kwargs) -> ResponseRecord:
        self.calls.append(
            (
                kwargs["item_id"],
                kwargs["repetition"],
                kwargs["is_warmup"],
            )
        )
        if self._interrupt_on_call == len(self.calls):
            raise KeyboardInterrupt
        return _success_record(model_tag, prompt, kwargs)


class SwallowingEngine(FakeEngine):
    """Engine that mirrors a keyboard interrupt captured inside run_request."""

    def run_request(self, model_tag: str, prompt: str, **kwargs) -> ResponseRecord:
        self.calls.append(
            (
                kwargs["item_id"],
                kwargs["repetition"],
                kwargs["is_warmup"],
            )
        )
        if self._interrupt_on_call == len(self.calls):
            self.interrupted = True
            now = datetime.now(timezone.utc)
            return ResponseRecord(
                experiment_id=kwargs["experiment_id"],
                model_tag=model_tag,
                item_id=kwargs["item_id"],
                category=kwargs["category"],
                profile=kwargs["profile"],
                repetition=kwargs["repetition"],
                is_warmup=kwargs["is_warmup"],
                prompt=prompt,
                response_text="",
                status=ResponseStatus.CANCELLED,
                error_type="InferenceCancelledError",
                error_message="Interrupted by user",
                started_at=now,
                completed_at=now,
                generation_options=GenerationOptions(),
                timing=TimingMetrics(ttft_seconds=None, total_latency_seconds=0.0),
                tokens=TokenMetrics(),
            )
        return _success_record(model_tag, prompt, kwargs)


class FakeManager:
    """Inventory-only manager used to test the pre-flight availability check."""

    def __init__(self, installed: set[str]) -> None:
        self._installed = installed

    def inventory(self, include_all: bool = False) -> list[ModelStatus]:
        tags = (
            "llama3.1:8b",
            "qwen2.5:7b-instruct",
            "mistral:7b-instruct-v0.3-q4_K_M",
        )
        return [
            ModelStatus(
                tag=tag,
                name=tag,
                role=ModelRole.CANDIDATE,
                installed=tag in self._installed,
            )
            for tag in tags
        ]


def _orchestrator(settings, database, dataset, engine) -> ExperimentOrchestrator:
    return ExperimentOrchestrator(
        settings,
        database,
        dataset,
        engine,
        manager=None,
        judge=None,
        sleep=lambda _seconds: None,
    )


def test_missing_model_is_rejected_before_any_experiment_is_created(
    temp_settings, dataset_path
) -> None:
    """A model that is not installed must not leave a failed experiment behind."""
    dataset = load_dataset(dataset_path)
    database = open_database(temp_settings)
    try:
        engine = FakeEngine()
        orchestrator = ExperimentOrchestrator(
            temp_settings,
            database,
            dataset,
            engine,
            manager=FakeManager({"llama3.1:8b"}),
            judge=None,
            sleep=lambda _seconds: None,
        )
        with pytest.raises(BenchmarkError, match="not installed.*qwen2.5:7b-instruct"):
            orchestrator.run(
                ["llama3.1:8b", "qwen2.5:7b-instruct"],
                item_ids=["gk-001"],
                warmup_requests=0,
                cooldown_seconds=0,
            )
        assert Repositories(database).experiments.history() == []
        assert engine.calls == []
    finally:
        database.dispose()


def test_run_persists_responses_metrics_and_evaluations(temp_settings, dataset_path) -> None:
    dataset = load_dataset(dataset_path)
    database = open_database(temp_settings)
    try:
        engine = FakeEngine()
        report = _orchestrator(temp_settings, database, dataset, engine).run(
            ["llama3.1:8b"],
            item_ids=ITEM_IDS,
            warmup_requests=1,
        )
        assert report.status == "completed"
        assert report.resumed is False
        assert report.responses == 3
        assert report.evaluations == 2
        assert report.successful_responses == 2
        assert report.failed_responses == 0
        assert [call[2] for call in engine.calls] == [True, False, False]

        repos = Repositories(database)
        results = repos.reads.model_results(report.experiment_id)
        assert len(results) == 1
        assert results[0].measured_responses == 2
        assert results[0].mean_latency_seconds == pytest.approx(1.5)
        assert repos.reads.question_results(report.experiment_id, "mr-001")
        stored = repos.experiments.get(report.experiment_id)
        assert stored is not None and stored.response_count == 3
    finally:
        database.dispose()


def test_interrupted_experiment_resumes_without_repeating_requests(
    temp_settings, dataset_path
) -> None:
    dataset = load_dataset(dataset_path)
    database = open_database(temp_settings)
    try:
        crashing = FakeEngine(interrupt_on_call=2)
        with pytest.raises(KeyboardInterrupt):
            _orchestrator(temp_settings, database, dataset, crashing).run(
                ["llama3.1:8b"],
                item_ids=ITEM_IDS,
                warmup_requests=0,
            )

        repos = Repositories(database)
        open_view = repos.experiments.history(1)[0]
        assert open_view.status == "running"
        assert open_view.response_count == 1
        assert len(repos.reads.question_results(open_view.experiment_id)) == 1

        resumed_engine = FakeEngine()
        report = _orchestrator(temp_settings, database, dataset, resumed_engine).run(
            ["llama3.1:8b"],
            item_ids=ITEM_IDS,
            warmup_requests=0,
            resume_experiment=open_view.experiment_id,
        )
        assert report.resumed is True
        assert report.status == "completed"
        assert report.responses == 2
        assert report.runs == 1
        assert [call[0] for call in resumed_engine.calls] == ["mr-001"]
        assert len(repos.reads.question_results(report.experiment_id)) == 2
    finally:
        database.dispose()


def test_interrupt_captured_inside_engine_leaves_experiment_resumable(
    temp_settings, dataset_path
) -> None:
    dataset = load_dataset(dataset_path)
    database = open_database(temp_settings)
    try:
        engine = SwallowingEngine(interrupt_on_call=1)
        report = _orchestrator(temp_settings, database, dataset, engine).run(
            ["llama3.1:8b"],
            item_ids=ITEM_IDS,
            warmup_requests=0,
        )

        assert report.interrupted is True
        assert report.status == "running"
        repos = Repositories(database)
        open_view = repos.experiments.get(report.experiment_id)
        assert open_view is not None and open_view.status == "running"
        assert open_view.response_count == 1
        run_id = repos.reads.model_results(report.experiment_id)[0].run_id
        run_view = repos.runs.get(run_id)
        assert run_view is not None and run_view.status == "cancelled"

        resumed_engine = FakeEngine()
        resumed = _orchestrator(temp_settings, database, dataset, resumed_engine).run(
            ["llama3.1:8b"],
            item_ids=ITEM_IDS,
            warmup_requests=0,
            resume_experiment=report.experiment_id,
        )
        assert resumed.interrupted is False
        assert resumed.resumed is True
        assert resumed.status == "completed"
        assert [call[0] for call in resumed_engine.calls] == ["mr-001"]
        final_run = repos.runs.get(run_id)
        assert final_run is not None and final_run.status == "completed"
        statuses = [
            row.response_status
            for row in repos.reads.response_inspection(report.experiment_id)
        ]
        assert sorted(statuses) == ["cancelled", "success"]
    finally:
        database.dispose()


def test_warmup_sequences_exceed_question_count_without_collisions(
    temp_settings, dataset_path
) -> None:
    dataset = load_dataset(dataset_path)
    database = open_database(temp_settings)
    try:
        engine = FakeEngine()
        report = _orchestrator(temp_settings, database, dataset, engine).run(
            ["llama3.1:8b"],
            item_ids=["gk-001", "mr-001"],
            warmup_requests=3,
            repetitions=1,
        )
        assert report.status == "completed"
        assert report.responses == 5
        assert [call for call in engine.calls if call[2]] == [
            ("gk-001", 1, True),
            ("mr-001", 2, True),
            ("gk-001", 3, True),
        ]
    finally:
        database.dispose()


def test_completed_experiment_cannot_be_resumed(temp_settings, dataset_path) -> None:
    dataset = load_dataset(dataset_path)
    database = open_database(temp_settings)
    try:
        report = _orchestrator(
            temp_settings, database, dataset, FakeEngine()
        ).run(["llama3.1:8b"], item_ids=["gk-001"], warmup_requests=0)

        with pytest.raises(ExperimentStateError):
            _orchestrator(temp_settings, database, dataset, FakeEngine()).run(
                ["llama3.1:8b"],
                item_ids=["gk-001"],
                warmup_requests=0,
                resume_experiment=report.experiment_id,
            )
    finally:
        database.dispose()


def test_resume_rejects_a_changed_configuration(
    temp_settings, dataset_path
) -> None:
    dataset = load_dataset(dataset_path)
    database = open_database(temp_settings)
    try:
        with pytest.raises(KeyboardInterrupt):
            _orchestrator(
                temp_settings,
                database,
                dataset,
                FakeEngine(interrupt_on_call=1),
            ).run(["llama3.1:8b"], item_ids=ITEM_IDS, warmup_requests=0)

        open_view = Repositories(database).experiments.history(1)[0]
        assert open_view.status == "running"
        changed_settings = temp_settings.model_copy(
            update={"config_hash": "0" * 64}
        )
        resumed_engine = FakeEngine()
        with pytest.raises(ExperimentStateError, match="different configuration"):
            _orchestrator(
                changed_settings, database, dataset, resumed_engine
            ).run(
                ["llama3.1:8b"],
                item_ids=ITEM_IDS,
                warmup_requests=0,
                resume_experiment=open_view.experiment_id,
            )
        assert resumed_engine.calls == []
    finally:
        database.dispose()


def test_history_survives_restart(temp_settings, dataset_path) -> None:
    dataset = load_dataset(dataset_path)
    database = open_database(temp_settings)
    try:
        report = _orchestrator(
            temp_settings, database, dataset, FakeEngine()
        ).run(["llama3.1:8b"], item_ids=["gk-001"], warmup_requests=0)
        path = database.path
    finally:
        database.dispose()

    reopened = open_database(temp_settings, path)
    try:
        history = Repositories(reopened).experiments.history()
        assert [view.experiment_id for view in history] == [report.experiment_id]
        assert history[0].status == "completed"
        assert history[0].evaluation_count == 1
        assert Repositories(reopened).reads.model_results(report.experiment_id)[0].run_id
    finally:
        reopened.dispose()
