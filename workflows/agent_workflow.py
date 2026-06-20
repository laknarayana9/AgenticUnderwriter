"""
Governed underwriting workflow using specialized agent role boundaries.

Implements the canonical HO3 quote-to-underwrite flow with deterministic
eligibility decisioning, cited evidence, pause/resume, rating, and HITL review.
"""
import os
import uuid
from copy import deepcopy
from typing import Dict, Any, List, Optional
from datetime import datetime
import time

from models.schemas import HO3Submission, WorkflowState, QuoteSubmission
from observability import get_tracer, record_workflow_latency
from app.llm_service import StructuredLLMService
from workflows.agents import (
    IntakeNormalizerAgent,
    PlannerRouterAgent,
    EnrichmentAgent,
    RetrievalAgent,
    UnderwritingAssessorAgent,
    VerifierGuardrailAgent,
    DecisionPackagerAgent
)
from workflows.hitl import get_hitl_workflow
from app.rag_engine import RAGEngine
from app.rating import RatingTool
from app.pii_masker import PIIMasker
from workflows.critic import CriticAgent


class UnderwritingWorkflow:
    """
    Governed HO3 underwriting workflow using specialized agent role boundaries.
    """

    def __init__(self, db=None):
        self.llm_service = StructuredLLMService()

        # Initialize agents
        self.intake_normalizer = IntakeNormalizerAgent(
            llm_service=self.llm_service,
            prompt_version="v1.0",
        )
        self.planner_router = PlannerRouterAgent()
        self.enrichment_agent = EnrichmentAgent()
        self.retrieval_agent = None  # Will be initialized with RAG engine
        self.assessor_agent = UnderwritingAssessorAgent()
        self.verifier_agent = VerifierGuardrailAgent()
        self.decision_packager = DecisionPackagerAgent(llm_service=self.llm_service)

        # Initialize supporting services
        self.rag_engine = RAGEngine()
        self.rag_engine.ingest_documents()
        self.retrieval_agent = RetrievalAgent(rag_engine=self.rag_engine)
        self.rating_tool = RatingTool()
        
        # Initialize HITL workflow
        self.hitl_workflow = get_hitl_workflow(db)

        # Initialize critic agent (distinct LLM from generator, configured via CRITIC_LLM_* env vars)
        self.critic_agent = CriticAgent()
        self._critic_max_retries = int(os.getenv("CRITIC_MAX_RETRIES", "2"))

    def run(self, submission_raw: Dict[str, Any]) -> WorkflowState:
        """
        Run the underwriting workflow with a raw HO3 submission.
        """
        return self._run(submission_raw)

    def resume(self, previous_state: WorkflowState, additional_answers: Dict[str, Any]) -> WorkflowState:
        """
        Resume a paused workflow using answers supplied by an underwriter or agent.
        """
        if previous_state.status != "waiting_for_info":
            raise ValueError("Only runs waiting for information can be resumed")

        updated_submission, applied_answers = self._apply_followup_answers(
            previous_state.submission_raw or {},
            previous_state.required_questions,
            additional_answers,
        )
        merged_answers = dict(previous_state.additional_answers)
        merged_answers.update(applied_answers)
        prior_events = list(previous_state.events)
        prior_events.append({
            "event": "missing_info_answers_received",
            "timestamp": datetime.now().isoformat(),
            "answers": applied_answers,
            "answered_fields": sorted(applied_answers.keys()),
        })

        return self._run(
            updated_submission,
            run_id=previous_state.run_id,
            quote_id=previous_state.quote_id,
            prior_events=prior_events,
            additional_answers=merged_answers,
        )

    def _run(
        self,
        submission_raw: Dict[str, Any],
        run_id: Optional[str] = None,
        quote_id: Optional[str] = None,
        prior_events: Optional[List[Dict[str, Any]]] = None,
        additional_answers: Optional[Dict[str, Any]] = None,
    ) -> WorkflowState:
        """
        Shared implementation for run() and resume(). Accepts optional prior_events
        and additional_answers to support mid-workflow resumption.
        """
        submission_raw = self._ensure_ho3_raw(submission_raw)
        tracer = get_tracer("underwriting_workflow")
        workflow_start = time.time()
        
        with tracer.start_as_current_span("underwriting_workflow_execution") as span:
            run_id = run_id or str(uuid.uuid4())
            quote_id = quote_id or f"Q-2026-{uuid.uuid4().hex[:6].upper()}"

            span.set_attribute("run_id", run_id)
            span.set_attribute("quote_id", quote_id)

            # QuoteSubmission is retained for backward compat with storage/response serialization
            quote_submission = self._convert_to_quote_submission(submission_raw)
            
            workflow_state = WorkflowState(
                run_id=run_id,
                quote_id=quote_id,
                status="processing",
                quote_submission=quote_submission,
                submission_raw=submission_raw,
                current_node="start",
                additional_answers=additional_answers or {},
                events=prior_events or []
            )

            try:
                # Step 1: Intake Normalization
                with tracer.start_as_current_span("intake_normalization"):
                    workflow_state.current_node = "intake_normalization"
                    normalization_result = self.intake_normalizer.normalize(submission_raw)
                    workflow_state.missing_info = normalization_result["missing_info"]
                    workflow_state.required_questions = normalization_result["questions"]

                # Step 2: Planning/Routing
                with tracer.start_as_current_span("planner_routing"):
                    workflow_state.current_node = "planner"
                    routing_decision = self.planner_router.route(
                        submission_raw,
                        workflow_state.missing_info
                    )

                if routing_decision["route"] == "waiting_for_info":
                    return self._pause_for_missing_info(workflow_state, normalization_result["questions"])

                workflow_state.submission_canonical = HO3Submission(**submission_raw)

                if routing_decision["route"] in ["hard_decline_candidate", "hard_refer"]:
                    workflow_state.status = "pending_review"
                    decision_type = "DECLINE" if routing_decision["route"] == "hard_decline_candidate" else "REFER"
                    decision_packet = self.decision_packager.package(
                        {
                            "preliminary_decision": decision_type,
                            "eligibility_score": 0.3,
                            "risk_factors": [{"code": code, "severity": "high", "because": reason, "citations": []} 
                                           for code, reason in zip(routing_decision["reason_codes"], routing_decision["reason_codes"])],
                            "required_questions": [],
                            "citations_used": [],
                            "confidence": 0.8
                        },
                        None,
                        []
                    )
                    workflow_state.decision_packet = decision_packet
                    
                    # Create HITL task for hard decline/refer
                    self.hitl_workflow.create_hitl_task(
                        run_id=run_id,
                        task_type="review",
                        description=f"Hard {decision_type} required: {', '.join(routing_decision['reason_codes'])}",
                        priority="urgent",
                        metadata={"decision": decision_type, "reason_codes": routing_decision["reason_codes"]}
                    )
                    
                    return workflow_state

                # Step 3: Enrichment
                with tracer.start_as_current_span("enrichment"):
                    workflow_state.current_node = "enrichment"
                    enrichment_data = self.enrichment_agent.enrich(
                        workflow_state.submission_canonical
                    )
                    workflow_state.enrichment = enrichment_data

                followup_questions = self._detect_contextual_missing_info(
                    workflow_state.submission_canonical,
                    enrichment_data,
                )
                if followup_questions:
                    workflow_state.missing_info = [q["question_id"] for q in followup_questions]
                    workflow_state.required_questions = followup_questions
                    return self._pause_for_missing_info(workflow_state, followup_questions)

                # Step 4: Retrieval
                with tracer.start_as_current_span("retrieval"):
                    workflow_state.current_node = "retrieval"
                    retrieval_result = self.retrieval_agent.retrieve(enrichment_data)
                    workflow_state.retrieval = retrieval_result

                # Step 5: Underwriting Assessment
                with tracer.start_as_current_span("underwriting_assessment"):
                    workflow_state.current_node = "assessment"
                    assessment_result = self.assessor_agent.assess(enrichment_data, retrieval_result)
                    workflow_state.assessment = assessment_result

                # Step 6: Verification
                with tracer.start_as_current_span("verification"):
                    workflow_state.current_node = "verification"
                    verification_result = self.verifier_agent.verify(assessment_result)
                    workflow_state.verification = verification_result

                # Check if verification failed - but still continue to decision packaging
                needs_review = not verification_result.get("decision_allowed", True)
                if needs_review:
                    workflow_state.status = "pending_review"
                    # Override assessment decision if verification failed
                    if verification_result.get("forced_decision"):
                        assessment_result["preliminary_decision"] = verification_result["forced_decision"]

                # Step 7: Rating
                with tracer.start_as_current_span("rating"):
                    workflow_state.current_node = "rating"
                    rating_data = self._prepare_rating_data(
                        workflow_state.submission_canonical,
                        enrichment_data
                    )
                    rating_result = self.rating_tool.calculate_premium(
                        rating_data["coverage_amount"],
                        rating_data
                    )
                    workflow_state.rating = rating_result

                # Step 8: Decision Packaging with generator-critic loop
                # PII is masked inside StructuredLLMService before any LLM call.
                with tracer.start_as_current_span("decision_packaging"):
                    workflow_state.current_node = "decision_packaging"

                    # Log which PII fields will be masked (no values logged)
                    _masker = PIIMasker()
                    _, _mask_map = _masker.mask_submission_context(submission_raw)
                    _masked_fields = _masker.fields_masked(_mask_map)
                    if _masked_fields:
                        workflow_state.events.append({
                            "event": "pii_masked",
                            "timestamp": datetime.now().isoformat(),
                            "fields_masked": _masked_fields,
                        })

                    workflow_steps_summary = self._summarize_completed_workflow_steps(workflow_state)
                    retrieved_chunks = (workflow_state.retrieval or {}).get("retrieved_chunks", [])
                    facts_used = assessment_result.get("facts_used", {})

                    decision_packet = None
                    for attempt in range(self._critic_max_retries + 1):
                        decision_packet = self.decision_packager.package(
                            assessment_result,
                            rating_result,
                            workflow_steps_summary,
                            mask_map=_mask_map,
                        )

                        # Critic verifies the LLM-generated rationale.
                        # Pass _mask_map so the critic can scrub PII from its own prompt.
                        verdict = self.critic_agent.verify_rationale(
                            rationale=decision_packet.producer_rationale,
                            retrieved_chunks=retrieved_chunks,
                            facts_used=facts_used,
                            attempt=attempt,
                            mask_map=_mask_map,
                        )
                        workflow_state.critic_verdicts.append(verdict.model_dump())
                        workflow_state.events.append({
                            "event": "critic_verdict",
                            "timestamp": datetime.now().isoformat(),
                            "attempt": attempt,
                            "passed": verdict.passed,
                            "invalid_citations": verdict.invalid_citation_ids,
                            "unsupported_facts": verdict.unsupported_facts,
                        })

                        if verdict.passed:
                            break

                        if attempt < self._critic_max_retries:
                            # Feed structured feedback to the generator on the next pass
                            assessment_result["critic_feedback"] = verdict.feedback_for_generator
                        else:
                            # Exhausted retries — emit a deterministic fallback rationale
                            workflow_state.events.append({
                                "event": "critic_fallback",
                                "timestamp": datetime.now().isoformat(),
                                "reason": "critic rejected rationale after max retries; using deterministic fallback",
                            })
                            from app.prompt_templates import PRODUCER_RATIONALE_RETRY_SUFFIX  # noqa: F401 (import unused but documents intent)
                            # Re-package with fallback forced by disabling LLM for this call
                            decision_packet = self.decision_packager.package(
                                {**assessment_result, "_force_fallback": True},
                                rating_result,
                                workflow_steps_summary,
                            )

                    workflow_state.rationale_retry_count = max(0, len(workflow_state.critic_verdicts) - 1)
                    workflow_state.decision_packet = decision_packet

                if decision_packet.needs_human_review:
                    workflow_state.status = "pending_review"
                    
                    # Create HITL task for human review
                    self.hitl_workflow.create_hitl_task(
                        run_id=run_id,
                        task_type="review",
                        description=f"Review required for decision: {decision_packet.decision.value}",
                        priority="medium",
                        metadata={
                            "decision": decision_packet.decision.value,
                            "risk_factors": assessment_result.get("risk_factors", []),
                            "eligibility_score": assessment_result.get("eligibility_score", 0)
                        }
                    )
                else:
                    workflow_state.status = "completed"

                workflow_state.events.append({
                    "event": "workflow_completed",
                    "timestamp": datetime.now().isoformat(),
                    "decision": decision_packet.decision.value
                })

                workflow_duration = time.time() - workflow_start
                record_workflow_latency("underwriting", workflow_duration * 1000)
                span.set_attribute("workflow.duration_ms", workflow_duration * 1000)
                span.set_attribute("workflow.status", workflow_state.status)

                return workflow_state

            except Exception as e:
                workflow_state.status = "failed"
                workflow_state.events.append({
                    "event": "workflow_error",
                    "timestamp": datetime.now().isoformat(),
                    "error": str(e)
                })
                span.set_attribute("error", str(e))
                raise

    def _pause_for_missing_info(self, workflow_state: WorkflowState, questions: List[Dict[str, Any]]) -> WorkflowState:
        workflow_state.status = "waiting_for_info"
        workflow_state.current_node = "waiting_for_info"
        workflow_state.required_questions = questions
        workflow_state.missing_info = [q.get("question_id", q.get("field_path", "missing_info")) for q in questions]

        decision_packet = self.decision_packager.package(
            {
                "preliminary_decision": "REFER",
                "eligibility_score": 0.5,
                "risk_factors": [
                    {
                        "code": q.get("question_id", "MISSING_INFO").upper(),
                        "severity": "medium",
                        "because": q.get("question", q.get("question_text", "Additional information is required.")),
                        "citations": [],
                    }
                    for q in questions
                ],
                "required_questions": questions,
                "citations_used": [],
                "confidence": 0.7,
                "reasoning": "Additional information is required before underwriting can continue.",
            },
            None,
            []
        )
        workflow_state.decision_packet = decision_packet

        question_texts = [q.get("question", str(q)) for q in questions]
        self.hitl_workflow.create_hitl_task(
            run_id=workflow_state.run_id,
            task_type="request_info",
            description=f"Additional information required: {', '.join(question_texts)}",
            priority="high",
            metadata={"decision": "REFER", "questions": questions}
        )
        workflow_state.events.append({
            "event": "workflow_paused_for_missing_info",
            "timestamp": datetime.now().isoformat(),
            "missing_info": workflow_state.missing_info,
            "questions": questions,
        })

        return workflow_state

    def _convert_to_quote_submission(self, ho3_data: Dict[str, Any]) -> QuoteSubmission:
        """
        Convert HO3 submission to QuoteSubmission for backward compatibility.
        """
        applicant = ho3_data.get("applicant", {})
        risk = ho3_data.get("risk", {})
        coverage = ho3_data.get("coverage_request", {})

        return QuoteSubmission(
            applicant_name=applicant.get("full_name", ""),
            address=risk.get("property_address", ""),
            property_type=risk.get("dwelling_type", "single_family"),
            coverage_amount=coverage.get("coverage_a", 500000),
            construction_year=risk.get("year_built"),
            square_footage=None,
            roof_type=None,
            foundation_type=None,
            additional_info=None
        )

    def _ensure_ho3_raw(self, submission_raw: Dict[str, Any]) -> Dict[str, Any]:
        """Accept legacy quote submissions while the public API migrates to HO3."""
        if {"applicant", "risk", "coverage_request"}.issubset(submission_raw.keys()):
            return submission_raw

        dwelling_type = submission_raw.get("property_type", "single_family")
        if dwelling_type not in {"single_family", "condo", "townhouse", "row_house", "manufactured", "commercial"}:
            dwelling_type = "single_family"

        return {
            "applicant": {
                "full_name": submission_raw.get("applicant_name", "Unknown Applicant")
            },
            "risk": {
                "property_address": submission_raw.get("address", ""),
                "occupancy": "owner_occupied_primary",
                "dwelling_type": dwelling_type,
                "year_built": submission_raw.get("construction_year") or 2000,
                "roof_age_years": 10 if submission_raw.get("roof_type") else None,
                "construction_type": "frame",
                "stories": 1
            },
            "coverage_request": {
                "coverage_a": submission_raw.get("coverage_amount", 250000),
                "deductible": 1000
            }
        }

    def _detect_contextual_missing_info(
        self,
        submission: HO3Submission,
        enrichment: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Ask for information that only becomes required after enrichment."""
        hazard_profile = enrichment.get("hazard_profile", {})
        questions = []

        if (
            hazard_profile.get("wildfire_band") in {"High", "Severe"}
            and submission.risk.wildfire_mitigation_evidence is None
        ):
            questions.append({
                "question_id": "wildfire_mitigation_evidence",
                "field_path": "risk.wildfire_mitigation_evidence",
                "answer_key": "risk.wildfire_mitigation_evidence",
                "question": "Is defensible-space or wildfire mitigation evidence documented for this property?",
                "question_text": "Is defensible-space or wildfire mitigation evidence documented for this property?",
                "question_type": "boolean",
                "required": True,
                "context": {
                    "wildfire_band": hazard_profile.get("wildfire_band"),
                    "property_address": submission.risk.property_address,
                },
            })

        return self.llm_service.word_missing_info_questions(
            questions,
            submission_context={
                "stage": "contextual_missing_info",
                "applicant": submission.applicant.model_dump(),
                "risk": submission.risk.model_dump(),
                "hazard_profile": hazard_profile,
            },
        )

    def _apply_followup_answers(
        self,
        submission_raw: Dict[str, Any],
        questions: List[Dict[str, Any]],
        answers: Dict[str, Any],
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        updated_submission = deepcopy(submission_raw)
        applied_answers: Dict[str, Any] = {}
        question_paths = {
            question.get("question_id"): question.get("field_path")
            for question in questions
            if question.get("question_id") and question.get("field_path")
        }
        question_paths.update({
            question.get("answer_key"): question.get("field_path")
            for question in questions
            if question.get("answer_key") and question.get("field_path")
        })
        fallback_paths = {
            "applicant_name": "applicant.full_name",
            "full_name": "applicant.full_name",
            "property_address": "risk.property_address",
            "occupancy": "risk.occupancy",
            "roof_age_years": "risk.roof_age_years",
            "wildfire_mitigation_evidence": "risk.wildfire_mitigation_evidence",
            "mitigation_notes": "risk.mitigation_notes",
        }

        for key, value in answers.items():
            field_path = question_paths.get(key) or fallback_paths.get(key) or (key if "." in key else None)
            if not field_path:
                continue
            coerced_value = self._coerce_followup_answer(field_path, value)
            self._set_nested_value(updated_submission, field_path, coerced_value)
            applied_answers[field_path] = coerced_value

        unanswered = [
            question.get("field_path")
            for question in questions
            if question.get("required") and question.get("field_path") not in applied_answers
        ]
        if unanswered:
            raise ValueError(f"Missing answers for required fields: {', '.join(unanswered)}")

        return updated_submission, applied_answers

    def _coerce_followup_answer(self, field_path: str, value: Any) -> Any:
        if field_path.endswith("roof_age_years") and value is not None:
            return int(value)
        if field_path.endswith("wildfire_mitigation_evidence"):
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                normalized = value.strip().lower()
                if normalized in {"yes", "y", "true", "1", "documented", "provided"}:
                    return True
                if normalized in {"no", "n", "false", "0", "missing", "not provided"}:
                    return False
            return bool(value)
        return value

    def _set_nested_value(self, payload: Dict[str, Any], field_path: str, value: Any) -> None:
        current = payload
        parts = field_path.split(".")
        for part in parts[:-1]:
            current = current.setdefault(part, {})
        current[parts[-1]] = value

    def _create_need_info_decision(self, questions: list) -> Dict[str, Any]:
        """Create a decision for missing information."""
        from models.schemas import DecisionType

        return {
            "decision": DecisionType.REFER,
            "rationale": "Additional information required to complete underwriting",
            "citations": [],
            "required_questions": questions,
            "next_steps": ["Provide required information and resubmit"]
        }

    def _create_refer_decision(self, reasons: list) -> Dict[str, Any]:
        """Create a referral decision."""
        from models.schemas import DecisionType

        return {
            "decision": DecisionType.REFER,
            "rationale": "; ".join(reasons),
            "citations": [],
            "required_questions": [],
            "next_steps": ["Manual underwriter review required"]
        }

    def _prepare_rating_data(self, submission: HO3Submission, enrichment: Dict) -> Dict[str, Any]:
        """Prepare data for rating engine."""
        property_profile = enrichment.get("property_profile", {})
        hazard_profile = enrichment.get("hazard_profile", {})

        return {
            "coverage_amount": submission.coverage_request.coverage_a,
            "property_type": submission.risk.dwelling_type,
            "hazard_scores": {
                "wildfire_risk": hazard_profile.get("wildfire_risk_score", 0),
                "flood_risk": hazard_profile.get("flood_risk_score", 0),
                "wind_risk": hazard_profile.get("wind_risk_score", 0),
                "earthquake_risk": hazard_profile.get("earthquake_risk_score", 0)
            },
            "territory": property_profile.get("territory", "MediumRiskCounty"),
            "construction_year": submission.risk.year_built
        }

    def _summarize_completed_workflow_steps(self, workflow_state: WorkflowState) -> list:
        """Summarize completed workflow steps for audit-facing decision packets."""
        completed_steps = []
        
        if workflow_state.enrichment:
            completed_steps.append({
                "tool": "enrichment_agent",
                "timestamp": datetime.now().isoformat(),
                "status": "completed"
            })
        
        if workflow_state.retrieval:
            completed_steps.append({
                "tool": "retrieval_agent",
                "timestamp": datetime.now().isoformat(),
                "status": "completed",
                "metrics": workflow_state.retrieval.get("retrieval_metrics", {})
            })
        
        if workflow_state.assessment:
            completed_steps.append({
                "tool": "underwriting_assessor",
                "timestamp": datetime.now().isoformat(),
                "status": "completed"
            })
        
        if workflow_state.verification:
            completed_steps.append({
                "tool": "verifier_guardrail",
                "timestamp": datetime.now().isoformat(),
                "status": "completed"
            })
        
        return completed_steps


def run_agent_workflow(submission_raw: Dict[str, Any]) -> WorkflowState:
    """
    Convenience function to run the agent workflow.
    """
    workflow = UnderwritingWorkflow()
    return workflow.run(submission_raw)
