"""LangGraph orchestration of the governed HO3 workflow (alternative engine).

A real `StateGraph` reimplementation of the same workflow the native engine runs,
with the **same governance boundary**: every node delegates to the existing
deterministic agents/services, the ACCEPT/REFER/DECLINE decision is produced by
`app/underwriting_rules.py` inside the assess node, and the rationale still flows
through `StructuredLLMService` + the generator–critic loop (no raw-LLM bypass).

What this adds over the native engine: idiomatic LangGraph orchestration and
**durable human-in-the-loop pause/resume via `interrupt()` + a SQLite
checkpointer** — a paused run survives a process restart and resumes from its
checkpoint. The native engine remains the governed default; this is selected with
`WORKFLOW_ENGINE=langgraph`.

Design notes (LangGraph 1.x best practices):
- Lean `TypedDict` state with `add` reducers for accumulators (events, verdicts);
  everything in state is checkpointed on each transition, so it stays minimal and
  JSON-able (the decision packet is stored as a dict).
- Nodes are thin and return partial updates; no input mutation.
- Nodes around `interrupt()` are idempotent — on resume a node re-runs from the
  top, and `interrupt()` returns the resume value instead of pausing again.
"""

from __future__ import annotations

import os
import sqlite3
import uuid
from datetime import datetime
from operator import add
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional, TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from app.pii_masker import PIIMasker
from app.vision_service import fold_vision_into_submission
from models.schemas import HO3Submission
from workflows.agent_workflow import UnderwritingWorkflow

DEFAULT_CHECKPOINT_DB = "storage/langgraph_checkpoints.sqlite"


class UWGraphState(TypedDict, total=False):
    submission_raw: Dict[str, Any]
    additional_answers: Dict[str, Any]
    enrichment: Optional[Dict[str, Any]]
    retrieval: Optional[Dict[str, Any]]
    assessment: Optional[Dict[str, Any]]
    verification: Optional[Dict[str, Any]]
    rating: Optional[Dict[str, Any]]
    decision_packet: Optional[Dict[str, Any]]   # stored as JSON dict (checkpoint-safe)
    status: str
    # Accumulators — survive checkpoint restores and grow across nodes/resumes.
    events: Annotated[List[Dict[str, Any]], add]
    critic_verdicts: Annotated[List[Dict[str, Any]], add]


class LangGraphUnderwritingWorkflow:
    """LangGraph engine that reuses the native workflow's governed components."""

    def __init__(self, native: Optional[UnderwritingWorkflow] = None, checkpoint_db: Optional[str] = None):
        # Reuse the native workflow for every governed component (agents, rules,
        # rating, vision, and the shared critic loop) — zero logic duplication.
        self.native = native or UnderwritingWorkflow()

        db_path = checkpoint_db or os.getenv("LANGGRAPH_CHECKPOINT_DB", DEFAULT_CHECKPOINT_DB)
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self.checkpointer = SqliteSaver(self._conn)
        try:
            self.checkpointer.setup()
        except Exception:
            pass  # setup is idempotent / lazily handled
        self.graph = self._build_graph()

    # --- graph construction -------------------------------------------------
    def _build_graph(self):
        g = StateGraph(UWGraphState)
        g.add_node("intake", self._intake)
        g.add_node("enrich", self._enrich)
        g.add_node("retrieve", self._retrieve)
        g.add_node("assess", self._assess)
        g.add_node("rate", self._rate)
        g.add_node("decide", self._decide)
        g.add_edge(START, "intake")
        g.add_edge("intake", "enrich")
        g.add_edge("enrich", "retrieve")
        g.add_edge("retrieve", "assess")
        g.add_edge("assess", "rate")
        g.add_edge("rate", "decide")
        g.add_edge("decide", END)
        return g.compile(checkpointer=self.checkpointer)

    # --- nodes (thin wrappers over native components) -----------------------
    def _intake(self, state: UWGraphState) -> Dict[str, Any]:
        norm = self.native.intake_normalizer.normalize(state["submission_raw"])
        if norm["missing_info"]:
            # Pauses durably; on resume `interrupt` returns the supplied answers.
            answers = interrupt({"stage": "intake_normalization", "questions": norm["questions"]})
            updated, applied = self.native._apply_followup_answers(
                state["submission_raw"], norm["questions"], answers,
            )
            return {
                "submission_raw": updated,
                "additional_answers": {**state.get("additional_answers", {}), **applied},
                "events": [self._event("missing_info_answers_received", answers=applied)],
            }
        return {}

    def _enrich(self, state: UWGraphState) -> Dict[str, Any]:
        submission = HO3Submission(**state["submission_raw"])
        enrichment = self.native.enrichment_agent.enrich(submission)
        followups = self.native._detect_contextual_missing_info(submission, enrichment)
        if followups:
            answers = interrupt({"stage": "contextual_missing_info", "questions": followups})
            updated, applied = self.native._apply_followup_answers(
                state["submission_raw"], followups, answers,
            )
            return {
                "submission_raw": updated,
                "enrichment": self.native.enrichment_agent.enrich(HO3Submission(**updated)),
                "additional_answers": {**state.get("additional_answers", {}), **applied},
                "events": [self._event("missing_info_answers_received", answers=applied)],
            }
        return {"enrichment": enrichment}

    def _retrieve(self, state: UWGraphState) -> Dict[str, Any]:
        return {"retrieval": self.native.retrieval_agent.retrieve(state["enrichment"])}

    def _assess(self, state: UWGraphState) -> Dict[str, Any]:
        assessment = self.native.assessor_agent.assess(state["enrichment"], state["retrieval"])
        verification = self.native.verifier_agent.verify(assessment)
        if not verification.get("decision_allowed", True) and verification.get("forced_decision"):
            assessment["preliminary_decision"] = verification["forced_decision"]
        return {"assessment": assessment, "verification": verification}

    def _rate(self, state: UWGraphState) -> Dict[str, Any]:
        submission = HO3Submission(**state["submission_raw"])
        rating_data = self.native._prepare_rating_data(submission, state["enrichment"])
        rating = self.native.rating_tool.calculate_premium(rating_data["coverage_amount"], rating_data)
        return {"rating": rating}

    def _decide(self, state: UWGraphState) -> Dict[str, Any]:
        masker = PIIMasker()
        _, mask_map = masker.mask_submission_context(state["submission_raw"])
        events: List[Dict[str, Any]] = []
        masked_fields = masker.fields_masked(mask_map)
        if masked_fields:
            events.append(self._event("pii_masked", fields_masked=masked_fields))

        retrieved = (state.get("retrieval") or {}).get("retrieved_chunks", [])
        facts_used = (state.get("assessment") or {}).get("facts_used", {})
        packet, verdicts, critic_events = self.native.package_decision_with_critic(
            state["assessment"], state.get("rating"), [], retrieved, facts_used, mask_map,
        )
        events.extend(critic_events)
        status = "pending_review" if packet.needs_human_review else "completed"
        events.append(self._event("workflow_completed", decision=packet.decision.value))
        return {
            "decision_packet": packet.model_dump(mode="json"),
            "critic_verdicts": verdicts,
            "events": events,
            "status": status,
        }

    @staticmethod
    def _event(name: str, **fields: Any) -> Dict[str, Any]:
        return {"event": name, "timestamp": datetime.now().isoformat(), **fields}

    # --- public API ---------------------------------------------------------
    def run(self, submission_raw: Dict[str, Any], image_bytes: Optional[bytes] = None,
            thread_id: Optional[str] = None) -> Dict[str, Any]:
        submission_raw = self.native._ensure_ho3_raw(submission_raw)
        if image_bytes:
            evidence = self.native.vision_service.extract_evidence(image_bytes)
            submission_raw, _ = fold_vision_into_submission(
                submission_raw, evidence, self.native.vision_service.config.min_confidence,
            )
        thread_id = thread_id or str(uuid.uuid4())
        init: UWGraphState = {
            "submission_raw": submission_raw,
            "additional_answers": {},
            "status": "processing",
            "events": [],
            "critic_verdicts": [],
        }
        result = self.graph.invoke(init, self._config(thread_id))
        return self._summarize(result, thread_id)

    def resume(self, thread_id: str, answers: Dict[str, Any]) -> Dict[str, Any]:
        """Resume a durably-paused run by thread_id with the supplied answers."""
        result = self.graph.invoke(Command(resume=answers), self._config(thread_id))
        return self._summarize(result, thread_id)

    @staticmethod
    def _config(thread_id: str) -> Dict[str, Any]:
        return {"configurable": {"thread_id": thread_id}}

    @staticmethod
    def _summarize(result: Dict[str, Any], thread_id: str) -> Dict[str, Any]:
        if "__interrupt__" in result:
            payload = getattr(result["__interrupt__"][0], "value", {}) or {}
            return {
                "thread_id": thread_id,
                "status": "waiting_for_info",
                "interrupted": True,
                "questions": payload.get("questions", []),
                "decision": None,
                "reason_codes": [],
            }
        packet = result.get("decision_packet") or {}
        return {
            "thread_id": thread_id,
            "status": result.get("status"),
            "interrupted": False,
            "decision": packet.get("decision"),
            "reason_codes": packet.get("review_reason_codes", []),
            "decision_packet": packet,
            "events": result.get("events", []),
        }
