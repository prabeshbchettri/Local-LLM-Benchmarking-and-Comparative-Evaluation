"""
Run new experiment page.

Responsibilities:
- Let the operator select models, dataset, repetitions, and cold/warm execution.
- Start the persisted orchestrator in a child process and monitor its progress.
- Prevent accidental duplicate submissions with a lock file and submission guard.
"""

from __future__ import annotations

import time
from pathlib import Path

import streamlit as st

from benchmark.core.exceptions import BenchmarkError
from benchmark.dashboard import data, runner, ui
from benchmark.dataset.loader import load_dataset
from benchmark.models.inference import InferenceProfile


@st.cache_data(ttl=300, show_spinner=False)
def _dataset_summary(path: str) -> dict[str, object]:
    """Load a dataset once and return the fields the runner form displays."""
    dataset = load_dataset(Path(path))
    return {
        "name": dataset.name,
        "version": dataset.version,
        "items": len(dataset.items),
        "hash": dataset.content_sha256,
    }


def _run_panel(context: data.DashboardContext) -> None:
    """Render an auto-refreshing status fragment for the recorded dashboard run."""

    @st.fragment(run_every="3s")
    def panel() -> None:
        run = runner.read_run(context.settings)
        if run is None:
            return
        live = runner.is_alive(run)
        experiment = data.experiment_by_name(context, run.experiment_name)
        with st.container(border=True):
            heading = st.columns([3, 1, 1])
            heading[0].markdown(
                f"**Dashboard run** `{run.experiment_name}`  \n"
                f"Models: {', '.join(run.models)} · {run.question_scope} · "
                f"log `{Path(run.log_path).name}`"
            )
            heading[1].metric("Process", "Running" if live else "Finished")
            heading[2].metric(
                "Experiment",
                experiment.status.title() if experiment else "Pending",
            )
            if experiment is not None:
                expected = max(run.expected_responses, 1)
                stored = experiment.response_count
                st.progress(
                    min(1.0, stored / expected),
                    text=(
                        f"{stored} of about {run.expected_responses} expected "
                        f"requests stored · {experiment.evaluation_count} evaluations"
                    ),
                )
                details = st.columns(4)
                details[0].caption(f"Experiment ID `{experiment.experiment_id}`")
                details[1].caption(f"Runs {experiment.run_count}")
                details[2].caption(f"Responses {experiment.response_count}")
                details[3].caption(f"Evaluations {experiment.evaluation_count}")
            else:
                st.info(
                    "Waiting for the child process to create the experiment record. "
                    "If this persists, inspect the runner log."
                )
            if not live:
                if experiment is not None and experiment.status == "completed":
                    st.success(
                        "The experiment finished and the database status is completed."
                    )
                elif experiment is not None and experiment.status == "failed":
                    st.error(
                        "The experiment finished with status failed. Read the runner "
                        "log for the error; stored partial results are preserved."
                    )
                elif experiment is not None and experiment.status == "cancelled":
                    st.warning("The experiment was cancelled; stored partial results remain.")
                else:
                    st.warning(
                        "The child process exited before the experiment reached a "
                        "completed or failed status. Inspect the log and database."
                    )
            with st.expander("Runner log (last 60 lines)", expanded=not live):
                log = runner.tail_log(run)
                st.code(log or "The runner has not written a log line yet.")
            with st.expander("Executed command"):
                st.code(" ".join(run.command))
            if not live:
                if st.button("Dismiss run record", key="dismiss_run_record"):
                    runner.clear_run(context.settings)
                    st.rerun(scope="app")

    panel()


def render(context: data.DashboardContext) -> None:
    """Render the run new experiment page."""
    ui.page_header(
        "Run New Experiment",
        "Launch the persisted orchestrator without blocking the dashboard.",
    )
    flash = st.session_state.pop("run_flash", None)
    if flash:
        st.success(flash)
    if context.dataset_error:
        st.error(f"The configured dataset cannot be loaded: {context.dataset_error}")

    recorded = runner.read_run(context.settings)
    live = recorded is not None and runner.is_alive(recorded)
    if recorded is not None:
        _run_panel(context)
        if live:
            st.info(
                "An experiment is running. The start form is disabled until the child "
                "process exits; the rest of the dashboard remains available."
            )

    st.write("")
    ui.section("Configure run")
    dataset_dir = context.settings.benchmark.dataset_path.parent
    configured_dataset = context.settings.benchmark.dataset_path.resolve()
    options = sorted(
        {path.resolve() for path in dataset_dir.glob("*.json")}
        | {configured_dataset},
        key=lambda path: path.name.lower(),
    )
    labels = {path: path.name for path in options}
    selected_dataset = st.selectbox(
        "Dataset",
        options=options,
        format_func=lambda path: labels[path],
        index=options.index(configured_dataset),
        key="run_dataset",
        help=(
            "The canonical dataset is the configured default. Other validated JSON "
            "datasets in the dataset directory can also be selected."
        ),
    )
    dataset_error: str | None = None
    summary: dict[str, object] | None = None
    try:
        summary = _dataset_summary(str(selected_dataset))
    except Exception as exc:
        dataset_error = f"{type(exc).__name__}: {exc}"
        st.error(f"The selected dataset failed validation: {dataset_error}")
    if summary is not None:
        st.caption(
            f"{summary['name']} {summary['version']} · "
            f"{summary['items']} questions · "
            f"sha256 `{str(summary['hash'])[:12]}…`"
        )

    question_count = int(summary["items"]) if summary else 1

    with st.form("run_experiment_form"):
        columns = st.columns(2)
        with columns[0]:
            models = st.multiselect(
                "Models",
                options=data.configured_candidates(context),
                default=data.configured_candidates(context),
                help=(
                    "Only configured candidate models are allowed by the "
                    "orchestrator. The blind judge model is configured separately."
                ),
            )
            profile_values = [profile.value for profile in InferenceProfile]
            default_profile = (
                InferenceProfile.QUALITY.value
                if InferenceProfile.QUALITY.value in profile_values
                else profile_values[0]
            )
            profile = st.selectbox(
                "Inference profile",
                options=profile_values,
                index=profile_values.index(default_profile),
                help=(
                    "Quality runs every selected question. Performance, consistency, "
                    "and cold-start profiles change repetition defaults."
                ),
            )
            scope = st.radio(
                "Question set",
                options=["First N questions", "All questions"],
                horizontal=True,
                help=(
                    "A subset is useful for smoke checks; all questions produces the "
                    "complete quality evidence."
                ),
            )
        with columns[1]:
            limit = st.number_input(
                "Number of questions",
                min_value=1,
                max_value=max(question_count, 1),
                value=min(10, max(question_count, 1)),
                step=1,
                help="Ignored when 'All questions' is selected.",
            )
            repetitions = st.number_input(
                "Repetitions per question",
                min_value=1,
                max_value=20,
                value=1,
                step=1,
                help="Each repetition is stored as a separate response.",
            )
            warmup = st.number_input(
                "Warm-up requests per model",
                min_value=0,
                max_value=10,
                value=int(context.settings.benchmark.warmup_requests),
                step=1,
                help="Ignored for cold-start runs, which unload before every request.",
            )
            cold_start = st.checkbox(
                "Cold-start execution",
                value=False,
                help=(
                    "Unload the model before every measured request and run with "
                    "keep_alive=0. Cold-start runs are excluded from default analytics."
                ),
            )
        st.caption(
            f"Configured cooldown between models is "
            f"{context.settings.benchmark.cooldown_seconds_between_models}s. "
            "The runner requires Ollama at "
            f"{context.settings.models.ollama_base_url}."
        )
        submitted = st.form_submit_button(
            "Start experiment",
            type="primary",
            disabled=live or dataset_error is not None or not models,
            help=(
                "Disabled while another dashboard experiment is running."
                if live
                else None
            ),
        )

    if submitted:
        now = time.monotonic()
        last = st.session_state.get("last_start_monotonic")
        if last is not None and now - last < 10.0:
            st.warning(
                "The same form was submitted moments ago; this duplicate request "
                "was ignored."
            )
        else:
            try:
                run = runner.start_run(
                    context.settings,
                    project_root=context.settings.project_root,
                    models=models,
                    profile=profile,
                    dataset_path=selected_dataset,
                    dataset_question_count=question_count,
                    limit=None if scope == "All questions" else int(limit),
                    repetitions=int(repetitions),
                    warmup=int(warmup),
                    cold_start=cold_start,
                )
            except (BenchmarkError, OSError) as exc:
                st.error(f"Could not start the experiment: {exc}")
            else:
                st.session_state["last_start_monotonic"] = now
                st.session_state["run_flash"] = (
                    f"Started {run.experiment_name} (PID {run.pid}). "
                    "Progress refreshes every three seconds."
                )
                st.rerun()

    st.write("")
    st.caption(
        "The dashboard never runs inference in the Streamlit process. The child "
        "process writes every response to SQLite as it completes, so the other pages "
        "stay usable while the experiment runs."
    )


__all__ = ["render"]
