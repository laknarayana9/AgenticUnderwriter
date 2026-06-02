"""Streamlit demo for the governed underwriting workflow.

Run with:
    streamlit run demo_app.py
"""

from __future__ import annotations

import json
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st

from workflows import UnderwritingWorkflow


ROOT = Path(__file__).resolve().parent
SAMPLES_PATH = ROOT / "examples" / "demo_submissions.json"


@st.cache_resource
def get_workflow() -> UnderwritingWorkflow:
    with redirect_stdout(StringIO()):
        return UnderwritingWorkflow()


@st.cache_data
def load_samples() -> Dict[str, Dict[str, Any]]:
    with SAMPLES_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    st.set_page_config(
        page_title="Agentic Underwriter Demo",
        page_icon="",
        layout="wide",
    )

    st.title("Agentic Underwriter")
    st.caption(
        "Governed homeowner-insurance workflow with cited decisions, "
        "audit events, and human review routing."
    )

    samples = load_samples()
    sample_names = list(samples.keys())
    selected_sample = st.sidebar.selectbox(
        "Sample submission",
        sample_names,
        format_func=lambda value: value.replace("_", " ").title(),
    )

    if "submission_json" not in st.session_state or st.sidebar.button("Load sample"):
        st.session_state.submission_json = json.dumps(samples[selected_sample], indent=2)

    st.sidebar.markdown("### Quick Start")
    st.sidebar.code("streamlit run demo_app.py", language="bash")

    left, right = st.columns([0.95, 1.05], gap="large")

    with left:
        st.subheader("Submission")
        submission_text = st.text_area(
            "Edit JSON",
            key="submission_json",
            height=520,
            label_visibility="collapsed",
        )
        run_clicked = st.button("Run underwriting workflow", type="primary", use_container_width=True)

    if run_clicked:
        try:
            submission = json.loads(submission_text)
            workflow = get_workflow()
            with st.spinner("Running intake, retrieval, assessment, rating, and packaging..."):
                st.session_state.workflow_state = workflow.run(submission)
            st.session_state.resume_error = None
        except Exception as exc:
            st.session_state.workflow_state = None
            st.session_state.resume_error = str(exc)

    with right:
        st.subheader("Decision")
        if st.session_state.get("resume_error"):
            st.error(st.session_state.resume_error)
        state = st.session_state.get("workflow_state")
        if state is None:
            st.info("Load a sample or paste a submission, then run the workflow.")
        else:
            render_state(state)


def render_state(state: Any) -> None:
    packet = state.decision_packet
    status = state.status.replace("_", " ").title()
    decision = packet.decision.value if packet else "WAITING"
    confidence = packet.decision_confidence if packet else 0.0
    review_needed = bool(packet.needs_human_review) if packet else state.status == "waiting_for_info"

    metric_cols = st.columns(4)
    metric_cols[0].metric("Status", status)
    metric_cols[1].metric("Decision", decision)
    metric_cols[2].metric("Confidence", f"{confidence:.2f}")
    metric_cols[3].metric("Human Review", "Yes" if review_needed else "No")

    if state.status == "waiting_for_info":
        render_followup_form(state)

    if packet:
        st.markdown("#### Rationale")
        st.write(packet.reason_summary)

        if packet.review_reason_codes:
            st.markdown("#### Reason Codes")
            st.write(", ".join(packet.review_reason_codes))

        if packet.next_steps:
            st.markdown("#### Next Steps")
            for step in packet.next_steps:
                st.write(f"- {step}")

        if packet.premium_indication:
            st.markdown("#### Premium Indication")
            st.json(packet.premium_indication)

        render_citations(packet.citations)

    render_events(state.events)

    with st.expander("Raw workflow state"):
        st.json(state.model_dump(mode="json"))


def render_followup_form(state: Any) -> None:
    st.warning("Additional information is required before the workflow can continue.")
    questions = state.required_questions or []
    with st.form("followup_answers"):
        answers: Dict[str, Any] = {}
        for question in questions:
            key = question.get("answer_key") or question.get("question_id") or question.get("field_path")
            label = question.get("question_text") or question.get("question") or key
            qtype = question.get("question_type", "text")
            if qtype == "numeric":
                answers[key] = st.number_input(label, min_value=0, max_value=100, value=10)
            elif qtype == "boolean":
                answers[key] = st.toggle(label, value=False)
            elif qtype == "choice":
                options = question.get("options") or []
                answers[key] = st.selectbox(label, options)
            else:
                answers[key] = st.text_input(label)

        submitted = st.form_submit_button("Resume with answers", use_container_width=True)

    if submitted:
        try:
            workflow = get_workflow()
            with st.spinner("Resuming the same run..."):
                st.session_state.workflow_state = workflow.resume(state, answers)
            st.session_state.resume_error = None
            st.rerun()
        except Exception as exc:
            st.session_state.resume_error = str(exc)
            st.rerun()


def render_citations(citations: List[Dict[str, Any]]) -> None:
    st.markdown("#### Citations")
    if not citations:
        st.write("No citations attached.")
        return

    for citation in citations[:8]:
        chunk_id = citation.get("chunk_id", "unknown chunk")
        source = f"{citation.get('doc_id', 'unknown')} / {citation.get('section', 'unknown section')}"
        with st.expander(f"{chunk_id}"):
            st.caption(source)
            excerpt = citation.get("excerpt") or citation.get("text") or ""
            st.write(excerpt)


def render_events(events: List[Dict[str, Any]]) -> None:
    st.markdown("#### Audit Events")
    if not events:
        st.write("No audit events recorded yet.")
        return

    for event in events:
        label = event.get("event", "event").replace("_", " ").title()
        timestamp = event.get("timestamp", "")
        st.write(f"- **{label}** {timestamp}")


if __name__ == "__main__":
    main()
