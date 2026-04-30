from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Dict, Any, Optional
import uuid
from datetime import datetime

from models.schemas import QuoteSubmission, RunRecord, WorkflowState, HO3Submission
from workflows.agent_workflow import PhaseAWorkflow
from storage.database import db

# Initialize FastAPI app
app = FastAPI(
    title="Agentic Quote-to-Underwrite API",
    description="An agentic workflow for insurance quote processing and underwriting",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8000"],  # Restrict to localhost for development
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],  # Specific methods instead of wildcard
    allow_headers=["Content-Type", "Authorization"],  # Specific headers instead of wildcard
)


# Request/Response models
class QuoteRunRequest(BaseModel):
    submission: QuoteSubmission
    use_agentic: bool = False  # Enable agentic behavior
    additional_answers: Optional[Dict[str, Any]] = None  # Answers to missing info questions


class HO3RunRequest(BaseModel):
    submission: HO3Submission  # Canonical HO3 submission format


class MissingInfoAnswerRequest(BaseModel):
    answers: Dict[str, Any]
    answered_by: Optional[str] = None


class QuoteRunResponse(BaseModel):
    run_id: str
    status: str
    decision: Optional[Dict[str, Any]] = None
    premium: Optional[Dict[str, Any]] = None
    citations: Optional[list] = None
    required_questions: Optional[list] = None
    requires_human_review: bool = False
    referral_triggers: Optional[list] = None
    message: str


class RunStatusResponse(BaseModel):
    run_id: str
    status: str
    created_at: datetime
    updated_at: datetime
    workflow_state: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


class RunListResponse(BaseModel):
    runs: list
    total_count: int


def store_run_record(workflow_state: WorkflowState, status: Optional[str] = None, error_message: Optional[str] = None):
    """
    Store the workflow result in the database.
    """
    run_id = workflow_state.run_id or str(uuid.uuid4())
    workflow_state.run_id = run_id
    record_status = status or workflow_state.status
    existing_record = db.get_run_record(run_id)
    created_at = existing_record.created_at if existing_record else datetime.now()
    
    # Create node outputs for audit trail
    node_outputs = {
        "validation": {
            "missing_info": workflow_state.missing_info,
            "tool_calls": [call.model_dump() for call in workflow_state.tool_calls if call.tool_name == "validate_submission"]
        },
        "enrichment": {
            "normalized_address": workflow_state.enrichment_result.normalized_address.model_dump() if workflow_state.enrichment_result else None,
            "hazard_scores": workflow_state.enrichment_result.hazard_scores.model_dump() if workflow_state.enrichment_result else None,
            "tool_calls": [call.model_dump() for call in workflow_state.tool_calls if call.tool_name in ["address_normalize", "hazard_score"]]
        },
        "retrieval": {
            "guidelines_count": len(workflow_state.retrieved_guidelines),
            "citations": [chunk.doc_id for chunk in workflow_state.retrieved_guidelines],
            "tool_calls": [call.model_dump() for call in workflow_state.tool_calls if call.tool_name == "rag_retrieval"]
        },
        "assessment": {
            "eligibility_score": workflow_state.uw_assessment.eligibility_score if workflow_state.uw_assessment else None,
            "triggers": [t.model_dump() for t in workflow_state.uw_assessment.triggers] if workflow_state.uw_assessment else [],
            "tool_calls": [call.model_dump() for call in workflow_state.tool_calls if call.tool_name == "underwriting_assessment"]
        },
        "rating": {
            "premium": workflow_state.premium_breakdown.model_dump() if workflow_state.premium_breakdown else None,
            "tool_calls": [call.model_dump() for call in workflow_state.tool_calls if call.tool_name == "rating_calculation"]
        },
        "decision": {
            "decision": workflow_state.decision.decision if workflow_state.decision else None,
            "rationale": workflow_state.decision.rationale if workflow_state.decision else None,
            "tool_calls": [call.model_dump() for call in workflow_state.tool_calls if call.tool_name == "decision_making"]
        }
    }
    
    run_record = RunRecord(
        run_id=run_id,
        created_at=created_at,
        updated_at=datetime.now(),
        status=record_status,
        workflow_state=workflow_state,
        node_outputs=node_outputs,
        error_message=error_message
    )
    
    db.save_run_record(run_record)
    return run_id


def _build_quote_response(workflow_state: WorkflowState, run_id: str) -> QuoteRunResponse:
    if workflow_state.decision_packet:
        decision_dict = {
            "decision": workflow_state.decision_packet.decision.value,
            "rationale": workflow_state.decision_packet.reason_summary,
            "reasoning": workflow_state.decision_packet.reason_summary,
            "confidence": workflow_state.decision_packet.decision_confidence,
            "citations": workflow_state.decision_packet.citations,
            "review_reason_codes": workflow_state.decision_packet.review_reason_codes,
            "next_steps": workflow_state.decision_packet.next_steps,
        }
        premium_dict = workflow_state.decision_packet.premium_indication
        citations = workflow_state.decision_packet.citations
    else:
        decision_dict = workflow_state.decision.model_dump() if workflow_state.decision else None
        premium_dict = workflow_state.premium_breakdown.model_dump() if workflow_state.premium_breakdown else None
        citations = workflow_state.uw_assessment.citations if workflow_state.uw_assessment else []

    if workflow_state.status == "waiting_for_info":
        message = "Additional information required to continue"
    elif workflow_state.decision_packet:
        decision = workflow_state.decision_packet.decision.value
        if decision == "ACCEPT":
            message = "Quote accepted for policy issuance"
        elif decision == "REFER":
            message = "Quote referred for manual review"
        elif decision == "DECLINE":
            message = "Quote declined"
        else:
            message = "Processing complete"
    else:
        message = "Processing complete"

    return QuoteRunResponse(
        run_id=run_id,
        status=workflow_state.status,
        decision=decision_dict,
        premium=premium_dict,
        citations=citations,
        required_questions=workflow_state.required_questions,
        requires_human_review=workflow_state.decision_packet.needs_human_review if workflow_state.decision_packet else False,
        referral_triggers=workflow_state.decision_packet.review_reason_codes if workflow_state.decision_packet else [],
        message=message,
    )


def _extract_submission_payload(request: Dict[str, Any]) -> Dict[str, Any]:
    """Accept both wrapped and direct JSON payloads for demo/backward compatibility."""
    if "submission" in request and isinstance(request["submission"], dict):
        return request["submission"]
    return request


def _normalize_ho3_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize older demo payloads into the canonical HO3 schema."""
    risk = dict(payload.get("risk", {}))
    coverage = dict(payload.get("coverage_request", {}))

    if "dwelling_type" not in risk and "property_type" in risk:
        risk["dwelling_type"] = risk["property_type"]
    risk.setdefault("dwelling_type", "single_family")
    risk.setdefault("year_built", payload.get("construction_year", 2000))
    risk.setdefault("construction_type", "frame")
    if risk["construction_type"] not in {"frame", "masonry", "superior_masonry", "fire_resistive", "manufactured"}:
        risk["construction_type"] = "fire_resistive" if risk["construction_type"] == "steel" else "frame"
    risk.setdefault("stories", 1)
    if "roof_age_years" not in risk:
        risk["roof_age_years"] = 10 if risk.get("roof_type") else None

    if "coverage_a" not in coverage and "coverage_amount" in coverage:
        coverage["coverage_a"] = coverage["coverage_amount"]

    return {
        "applicant": payload.get("applicant", {"full_name": payload.get("applicant_name", "Unknown Applicant")}),
        "risk": risk,
        "coverage_request": coverage,
        "quote_id": payload.get("quote_id"),
    }


@app.post("/quote/run", response_model=QuoteRunResponse)
async def run_quote_processing(request: Dict[str, Any]):
    """
    Process a quote submission through the underwriting workflow.
    """
    submission_payload = _extract_submission_payload(request)
    quote_submission = QuoteSubmission(**submission_payload)

    try:
        # Use 7-agent system for all processing
        workflow_state = PhaseAWorkflow(db=db).run(quote_submission.model_dump())
        
        # Store the run record
        run_id = store_run_record(workflow_state)
        return _build_quote_response(workflow_state, run_id)
        
    except Exception as e:
        # Create a failed run record
        error_state = WorkflowState(quote_submission=quote_submission)
        run_id = store_run_record(error_state, status="failed", error_message=str(e))
        
        raise HTTPException(
            status_code=500,
            detail=f"Processing failed: {str(e)}"
        )


@app.post("/quote/ho3", response_model=QuoteRunResponse)
async def run_ho3_quote_processing(request: Dict[str, Any]):
    """
    Process an HO3 submission through the Phase A workflow with 7 specialized agents.
    """
    submission_payload = _normalize_ho3_payload(_extract_submission_payload(request))

    try:
        # Run Phase A workflow
        workflow_state = PhaseAWorkflow(db=db).run(submission_payload)
        
        # Store the run record
        run_id = store_run_record(workflow_state)
        return _build_quote_response(workflow_state, run_id)
        
    except Exception as e:
        # Create a failed run record
        applicant = submission_payload.get("applicant", {})
        risk = submission_payload.get("risk", {})
        coverage = submission_payload.get("coverage_request", {})
        error_state = WorkflowState(quote_submission=QuoteSubmission(
            applicant_name=applicant.get("full_name", ""),
            address=risk.get("property_address", ""),
            property_type=risk.get("dwelling_type", "single_family"),
            coverage_amount=coverage.get("coverage_a", 250000)
        ))
        run_id = store_run_record(error_state, status="failed", error_message=str(e))
        
        raise HTTPException(
            status_code=500,
            detail=f"Processing failed: {str(e)}"
        )


@app.get("/runs/{run_id}", response_model=RunStatusResponse)
async def get_run_status(run_id: str):
    """
    Get the status and details of a specific run.
    """
    run_record = db.get_run_record(run_id)
    
    if run_record is None:
        raise HTTPException(status_code=404, detail="Run not found")
    
    return RunStatusResponse(
        run_id=run_record.run_id,
        status=run_record.status,
        created_at=run_record.created_at,
        updated_at=run_record.updated_at,
        workflow_state=run_record.workflow_state.model_dump(),
        error_message=run_record.error_message
    )


@app.post("/runs/{run_id}/answers", response_model=QuoteRunResponse)
async def answer_missing_info(run_id: str, request: MissingInfoAnswerRequest):
    """
    Submit missing-information answers and resume the same quote run.
    """
    run_record = db.get_run_record(run_id)
    if run_record is None:
        raise HTTPException(status_code=404, detail="Run not found")
    if run_record.status != "waiting_for_info":
        raise HTTPException(status_code=409, detail="Run is not waiting for information")

    open_task = db.get_open_hitl_task_for_run(run_id)

    try:
        workflow = PhaseAWorkflow(db=db)
        workflow_state = workflow.resume(run_record.workflow_state, request.answers)
        resumed_run_id = store_run_record(workflow_state, status=workflow_state.status)

        if open_task:
            db.process_hitl_action(
                open_task["task_id"],
                {"action": "request_info", "answers": request.answers},
                request.answered_by or "agent",
            )

        return _build_quote_response(workflow_state, resumed_run_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/runs", response_model=RunListResponse)
async def list_runs(limit: int = 50, status: Optional[str] = None):
    """
    List recent runs with optional status filter.
    """
    runs = db.list_runs(limit=limit, status=status)
    
    return RunListResponse(
        runs=runs,
        total_count=len(runs)
    )


@app.get("/runs/{run_id}/audit")
async def get_run_audit(run_id: str):
    """
    Get the full audit trail for a run including all node outputs.
    """
    run_record = db.get_run_record(run_id)
    
    if run_record is None:
        raise HTTPException(status_code=404, detail="Run not found")
    
    return {
        "run_id": run_record.run_id,
        "status": run_record.status,
        "created_at": run_record.created_at,
        "updated_at": run_record.updated_at,
        "workflow_state": run_record.workflow_state.model_dump(),
        "node_outputs": run_record.node_outputs,
        "tool_calls": [call.model_dump() for call in run_record.workflow_state.tool_calls],
        "error_message": run_record.error_message
    }


@app.get("/stats")
async def get_statistics():
    """
    Get basic statistics about the system.
    """
    return db.get_statistics()


@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "version": "1.0.0"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
