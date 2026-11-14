"""
Individual response inspection page.

Responsibilities:
- Select an experiment, model, category, and question.
- Show the original prompt, expected answer, generated answer, evaluation results,
  latency, token counts, error information, and stored hardware summary.
"""

from __future__ import annotations

import streamlit as st

from benchmark.core.exceptions import BenchmarkError
from benchmark.dashboard import data, ui


def render(context: data.DashboardContext) -> None:
    """Render the response inspection page."""
    ui.page_header(
        "Response Inspection",
        "Read the exact stored prompt, answer, evaluation, and timing for one question.",
    )
    view = ui.select_experiment(context, "inspection_experiment")
    if view is None:
        return
    try:
        with st.spinner("Loading stored responses…"):
            rows = data.measured_rows(
                context,
                view.experiment_id,
                include_cold_start=True,
                include_smoke=True,
            )
    except BenchmarkError as exc:
        st.warning(f"Stored responses are unavailable: {exc}")
        return
    if not rows:
        ui.empty_state(
            "No stored responses",
            "This experiment has no measured responses; warm-up requests are not "
            "shown here.",
        )
        return

    models = sorted({row.model_tag for row in rows})
    categories = sorted({row.category or "unknown" for row in rows})
    selectors = st.columns([2, 2, 2])
    with selectors[0]:
        model = st.selectbox(
            "Model",
            options=models,
            key="inspection_model",
            help="Only measured responses for this model are inspected.",
        )
    model_rows = [row for row in rows if row.model_tag == model]
    with selectors[1]:
        category = st.selectbox(
            "Category",
            options=["All categories"] + categories,
            key="inspection_category",
        )
    if category != "All categories":
        model_rows = [
            row for row in model_rows if (row.category or "unknown") == category
        ]
    question_ids = sorted({row.question_id for row in model_rows})
    if not question_ids:
        ui.empty_state(
            "No questions for this selection",
            "Choose a different model or category.",
        )
        return
    with selectors[2]:
        question_id = st.selectbox(
            "Question ID",
            options=question_ids,
            key="inspection_question",
        )

    try:
        with st.spinner("Loading response detail…"):
            details = data.inspections(
                context,
                view.experiment_id,
                model_tag=model,
                question_id=question_id,
            )
    except BenchmarkError as exc:
        st.warning(f"Response detail is unavailable: {exc}")
        return
    if not details:
        ui.empty_state(
            "No matching response",
            "No stored response matches this model and question.",
        )
        return

    selections = sorted(
        {
            (detail.repetition, bool(detail.cold_start), detail.run_profile)
            for detail in details
        }
    )
    selection = st.selectbox(
        "Repetition",
        options=selections,
        key="inspection_repetition",
        format_func=lambda option: (
            f"repetition {option[0]} - {'cold start' if option[1] else 'warm start'}"
            f" - {option[2]}"
        ),
        help="Repetition, start mode, and inference profile come from the stored rows.",
    )
    repetition, cold_start, run_profile = selection
    detail = next(
        item
        for item in details
        if (
            item.repetition,
            bool(item.cold_start),
            item.run_profile,
        )
        == selection
    )
    selection_key = f"{repetition}_{int(cold_start)}_{run_profile}"
    item = data.dataset_items(context).get(question_id)

    body = st.columns(2)
    with body[0]:
        st.markdown("**Original prompt**")
        st.text_area(
            "Prompt",
            value=detail.prompt,
            height=180,
            disabled=True,
            key=f"inspection_prompt_{question_id}_{selection_key}",
            label_visibility="collapsed",
        )
        if detail.system_prompt:
            with st.expander("System instruction"):
                st.text(detail.system_prompt)
        st.markdown("**Expected answer**")
        expected = None
        expected_kind = "No expected answer is defined for this item."
        if item is not None:
            if item.expected_answer:
                expected = item.expected_answer
                expected_kind = "Stored expected_answer."
            elif item.reference_answer:
                expected = item.reference_answer
                expected_kind = "Stored reference_answer (this item is not exact-match)."
        if expected:
            st.text_area(
                "Expected answer",
                value=expected,
                height=120,
                disabled=True,
                key=f"inspection_expected_{question_id}_{selection_key}",
                label_visibility="collapsed",
            )
            st.caption(expected_kind)
        else:
            st.info(expected_kind)
        if item is not None:
            st.caption(
                f"Item `{item.id}` · {item.category.value.replace('_', ' ')} · "
                f"{item.difficulty.value} · evaluation `{item.evaluation_type.value}`"
            )
    with body[1]:
        st.markdown("**Generated answer**")
        st.text_area(
            "Generated answer",
            value=detail.response_text or "(empty response)",
            height=180,
            disabled=True,
            key=f"inspection_answer_{question_id}_{selection_key}",
            label_visibility="collapsed",
        )
        if detail.error_message:
            st.error(f"{detail.error_type or 'error'}: {detail.error_message}")
        elif detail.response_status != "success":
            st.warning(
                f"Response status is `{detail.response_status}`; no successful "
                "answer was stored."
            )
        st.caption(
            f"Status `{detail.response_status}` · profile `{detail.run_profile}` · "
            f"cold start {'yes' if detail.cold_start else 'no'} · "
            f"finish reason `{detail.finish_reason or 'unknown'}` · "
            f"{detail.stream_chunk_count} stream chunks"
        )

    st.write("")
    ui.metric_row(
        [
            (
                "Total latency",
                ui.format_seconds(detail.latency_seconds),
                "Monotonic client clock from request start to completion.",
            ),
            (
                "Time to first token",
                ui.format_seconds(detail.ttft_seconds),
                "Client-measured TTFT; unavailable for failed streams.",
            ),
            (
                "Output tokens per second",
                ui.format_rate(detail.output_tokens_per_second),
                "Ollama-reported eval count divided by eval duration.",
            ),
            (
                "Combined score",
                ui.format_score(detail.composite_score),
                "Stored composite evaluation score.",
            ),
        ]
    )
    ui.metric_row(
        [
            (
                "Prompt tokens",
                ui.format_count(detail.prompt_eval_count),
                "Ollama prompt evaluation count.",
            ),
            (
                "Output tokens",
                ui.format_count(detail.eval_count),
                "Ollama generated token count.",
            ),
            (
                "Prompt rate",
                ui.format_rate(detail.prompt_tokens_per_second),
                "Ollama-reported prompt processing speed.",
            ),
            (
                "Model load time",
                ui.format_seconds(detail.load_duration_seconds),
                "Ollama-reported model load duration.",
            ),
        ]
    )

    st.write("")
    ui.section("Evaluation results")
    if detail.evaluation_status is None:
        st.info("This response has no stored evaluation.")
    else:
        evaluation_metrics = st.columns(4)
        evaluation_metrics[0].metric(
            "Deterministic",
            ui.format_score(detail.deterministic_normalized_score),
            help=f"{detail.evaluation_type or 'unknown'} evaluator",
            border=True,
        )
        evaluation_metrics[1].metric(
            "LLM judge",
            ui.format_score(detail.judge_normalized_score),
            help=f"Judge status: {detail.judge_status or 'not configured'}",
            border=True,
        )
        evaluation_metrics[2].metric(
            "Composite",
            ui.format_score(detail.composite_score),
            help="Stored combined score for this response.",
            border=True,
        )
        evaluation_metrics[3].metric(
            "Passed",
            "Yes"
            if detail.passed is True
            else "No"
            if detail.passed is False
            else "—",
            help=f"Evaluation status: {detail.evaluation_status}",
            border=True,
        )
        if detail.checks:
            st.dataframe(
                ui.clean_frame(
                    [
                        {
                            "Check": check.get("name"),
                            "Passed": check.get("passed"),
                            "Score": check.get("score"),
                            "Max score": check.get("max_score"),
                            "Detail": check.get("detail"),
                        }
                        for check in detail.checks
                    ]
                ),
                width="stretch",
                hide_index=True,
            )
        if detail.judge_dimensions:
            st.dataframe(
                ui.clean_frame(
                    [
                        {
                            "Dimension": name.replace("_", " ").title(),
                            "Score (1-5)": value.get("score"),
                            "Rationale": value.get("rationale"),
                        }
                        for name, value in detail.judge_dimensions.items()
                    ]
                ),
                width="stretch",
                hide_index=True,
            )
        if detail.evaluation_errors:
            for error in detail.evaluation_errors:
                st.warning(
                    f"{error.get('stage')}: {error.get('error_type')} — "
                    f"{error.get('message')}"
                )

    with st.expander("Stored hardware summary"):
        if detail.hardware_summary:
            st.json(detail.hardware_summary, expanded=False)
        else:
            st.info("No hardware summary was stored with this response.")

    if len(details) > 1:
        st.write("")
        ui.section("All repetitions for this question")
        st.dataframe(
            ui.clean_frame(
                [
                    {
                        "Repetition": item.repetition,
                        "Run profile": item.run_profile,
                        "Start mode": "cold" if item.cold_start else "warm",
                        "Status": item.response_status,
                        "Latency (s)": item.latency_seconds,
                        "TTFT (s)": item.ttft_seconds,
                        "Deterministic": item.deterministic_normalized_score,
                        "LLM judge": item.judge_normalized_score,
                        "Composite": item.composite_score,
                        "Error": item.error_message,
                    }
                    for item in details
                ]
            ),
            width="stretch",
            hide_index=True,
            column_config={
                "Latency (s)": st.column_config.NumberColumn(format="%.3f"),
                "TTFT (s)": st.column_config.NumberColumn(format="%.3f"),
                "Deterministic": st.column_config.NumberColumn(format="%.1f"),
                "LLM judge": st.column_config.NumberColumn(format="%.1f"),
                "Composite": st.column_config.NumberColumn(format="%.1f"),
            },
        )


__all__ = ["render"]
