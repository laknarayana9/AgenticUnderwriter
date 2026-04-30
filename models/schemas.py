from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any, Literal
from datetime import datetime
from enum import Enum


class DecisionType(str, Enum):
    ACCEPT = "ACCEPT"
    REFER = "REFER"
    DECLINE = "DECLINE"


# Canonical HO3 Schema - Phase A Enhancement
class Applicant(BaseModel):
    """Applicant information for HO3 policy"""
    full_name: str = Field(..., description="Full legal name of the applicant")
    email: Optional[str] = Field(None, description="Email address")
    phone: Optional[str] = Field(None, description="Phone number")


class RiskProfile(BaseModel):
    """Property risk information for HO3 underwriting"""
    property_address: str = Field(..., description="Full property address")
    occupancy: Literal["owner_occupied_primary", "owner_occupied_secondary", "tenant_occupied", "vacant"] = Field(..., description="Property occupancy type")
    dwelling_type: Literal["single_family", "condo", "townhouse", "row_house", "manufactured", "commercial"] = Field(..., description="Type of dwelling")
    year_built: int = Field(..., ge=1800, le=2026, description="Year property was built")
    roof_age_years: Optional[int] = Field(None, ge=0, description="Age of roof in years")
    construction_type: Literal["frame", "masonry", "superior_masonry", "fire_resistive", "manufactured"] = Field(..., description="Construction type")
    stories: int = Field(default=1, ge=1, le=5, description="Number of stories")
    wildfire_mitigation_evidence: Optional[bool] = Field(None, description="Whether defensible-space or wildfire mitigation evidence is documented")
    mitigation_notes: Optional[str] = Field(None, description="Producer or underwriter notes describing mitigation evidence")


class CoverageRequest(BaseModel):
    """Coverage request for HO3 policy (Coverages A-F)"""
    coverage_a: float = Field(..., gt=0, description="Coverage A - Dwelling limit")
    coverage_b_pct: float = Field(default=10, ge=0, le=100, description="Coverage B - Other structures as % of A")
    coverage_c_pct: float = Field(default=50, ge=0, le=100, description="Coverage C - Personal property as % of A")
    coverage_d_pct: float = Field(default=20, ge=0, le=100, description="Coverage D - Loss of use as % of A")
    coverage_e: float = Field(default=300000, ge=0, description="Coverage E - Liability limit")
    coverage_f: float = Field(default=5000, ge=0, description="Coverage F - Medical payments limit")
    deductible: float = Field(default=1000, ge=0, description="Policy deductible")


class QuoteSubmission(BaseModel):
    applicant_name: str
    address: str
    property_type: str = Field(..., description="e.g., single_family, condo, commercial")
    coverage_amount: float = Field(..., gt=0)
    construction_year: Optional[int] = None
    square_footage: Optional[float] = None
    roof_type: Optional[str] = None
    foundation_type: Optional[str] = None
    additional_info: Optional[str] = None


# Phase A Enhancement: Canonical HO3 Submission
class HO3Submission(BaseModel):
    """Canonical HO3 submission using structured schema"""
    applicant: Applicant
    risk: RiskProfile
    coverage_request: CoverageRequest
    quote_id: Optional[str] = Field(None, description="Quote ID if resuming")


class NormalizedAddress(BaseModel):
    street_address: str
    city: str
    state: str
    zip_code: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    county: Optional[str] = None


class HazardScores(BaseModel):
    wildfire_risk: float = Field(..., ge=0, le=1, description="0-1 scale")
    flood_risk: float = Field(..., ge=0, le=1, description="0-1 scale")
    wind_risk: float = Field(..., ge=0, le=1, description="0-1 scale")
    earthquake_risk: float = Field(..., ge=0, le=1, description="0-1 scale")


class PremiumBreakdown(BaseModel):
    base_premium: float
    hazard_surcharge: float
    total_premium: float
    rating_factors: Dict[str, float] = Field(default_factory=dict)


class EnrichmentResult(BaseModel):
    normalized_address: NormalizedAddress
    hazard_scores: HazardScores
    property_details: Dict[str, Any] = Field(default_factory=dict)


class RetrievalChunk(BaseModel):
    doc_id: str
    doc_version: str
    section: str
    chunk_id: str
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    relevance_score: Optional[float] = None
    score: Optional[float] = Field(None, description="Similarity score from retrieval")
    effective_date: Optional[str] = Field(None, description="Effective date of the guideline")


class DecisionPacket(BaseModel):
    """Final decision packet suitable for producer-facing explanation, underwriter review, and audit"""
    decision: DecisionType
    decision_confidence: float = Field(..., ge=0, le=1, description="Confidence in the decision")
    reason_summary: str = Field(..., description="Concise summary of the decision rationale")
    citations: List[Dict[str, Any]] = Field(default_factory=list, description="List of citation objects with chunk_id, section, doc_version")
    premium_indication: Optional[Dict[str, Any]] = Field(None, description="Premium indication with annual_premium, currency")
    needs_human_review: bool = Field(default=False, description="Whether human review is required")
    review_reason_codes: List[str] = Field(default_factory=list, description="Reason codes for review")
    next_steps: List[str] = Field(default_factory=list, description="Recommended next steps")
    facts_used: Dict[str, Any] = Field(default_factory=dict, description="Key facts used in decision")
    evidence_cited: List[str] = Field(default_factory=list, description="Evidence chunk IDs cited")
    tool_calls_summary: List[Dict[str, Any]] = Field(default_factory=list, description="Summary of tool calls made")
    trace_link: Optional[str] = Field(None, description="Link to trace in observability system")


class UWTrigger(BaseModel):
    trigger_type: str  # e.g., "high_hazard", "missing_info", "guideline_violation"
    description: str
    severity: str  # e.g., "low", "medium", "high"
    requires_action: bool = False


class UWQuestion(BaseModel):
    question_id: str
    question_text: str
    question_type: str  # e.g., "text", "choice", "numeric"
    required: bool = True
    options: Optional[List[str]] = None


class UWAssessment(BaseModel):
    eligibility_score: float = Field(..., ge=0, le=1)
    triggers: List[UWTrigger] = Field(default_factory=list)
    required_questions: List[UWQuestion] = Field(default_factory=list)
    reasoning: str
    citations: List[str] = Field(default_factory=list)
    confidence: float = Field(..., ge=0, le=1)


class Decision(BaseModel):
    decision: DecisionType
    rationale: str
    citations: List[str] = Field(default_factory=list)
    premium: Optional[PremiumBreakdown] = None
    required_questions: List[UWQuestion] = Field(default_factory=list)
    next_steps: List[str] = Field(default_factory=list)


class ToolCall(BaseModel):
    model_config = ConfigDict(json_encoders={
        datetime: lambda v: v.isoformat()
    })
    
    tool_name: str
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    timestamp: datetime
    execution_time_ms: Optional[int] = None


class WorkflowState(BaseModel):
    model_config = ConfigDict(json_encoders={
        datetime: lambda v: v.isoformat()
    })

    # Original fields for backward compatibility
    quote_submission: QuoteSubmission
    enrichment_result: Optional[EnrichmentResult] = None
    retrieved_guidelines: List[RetrievalChunk] = Field(default_factory=list)
    uw_assessment: Optional[UWAssessment] = None
    decision: Optional[Decision] = None
    premium_breakdown: Optional[PremiumBreakdown] = None
    tool_calls: List[ToolCall] = Field(default_factory=list)
    current_node: Optional[str] = None
    missing_info: List[str] = Field(default_factory=list)
    required_questions: List[Dict[str, Any]] = Field(default_factory=list)
    additional_answers: Dict[str, Any] = Field(default_factory=dict)
    citation_guardrail_triggered: bool = False

    run_id: Optional[str] = Field(None, description="Unique run identifier")
    quote_id: Optional[str] = Field(None, description="Quote identifier")
    status: Literal["processing", "waiting_for_info", "pending_review", "completed", "failed"] = Field("processing", description="Current status")

    # Canonical HO3 submission (new format)
    submission_raw: Optional[Dict[str, Any]] = Field(None, description="Raw submission data")
    submission_canonical: Optional[HO3Submission] = Field(None, description="Canonical HO3 submission")

    # Enhanced enrichment with confidence map
    enrichment: Optional[Dict[str, Any]] = Field(None, description="Detailed enrichment including property_profile, hazard_profile, location_profile, confidence_map")

    # Enhanced retrieval with metrics
    retrieval: Optional[Dict[str, Any]] = Field(None, description="Retrieval details including queries, filters, evidence_chunks, retrieval_metrics")

    # Governed web search
    search: Optional[Dict[str, Any]] = Field(None, description="Web search details including enabled flag, queries, results, policy_decisions")

    # Assessment and verification
    assessment: Optional[Dict[str, Any]] = Field(None, description="Structured assessment output")
    verification: Optional[Dict[str, Any]] = Field(None, description="Verification/guardrail output")

    # Rating
    rating: Optional[Dict[str, Any]] = Field(None, description="Rating details")

    # Final decision packet
    decision_packet: Optional[DecisionPacket] = Field(None, description="Final decision packet")

    # Events log
    events: List[Dict[str, Any]] = Field(default_factory=list, description="Event log for audit trail")

    # Trace information
    trace: Optional[Dict[str, Any]] = Field(None, description="Trace information including phoenix_trace_id, phoenix_url")


class RunRecord(BaseModel):
    model_config = ConfigDict(json_encoders={
        datetime: lambda v: v.isoformat()
    })
    
    run_id: str
    created_at: datetime
    updated_at: datetime
    status: str  # e.g., "running", "completed", "failed", "waiting_for_info"
    workflow_state: WorkflowState
    node_outputs: Dict[str, Any] = Field(default_factory=dict)
    error_message: Optional[str] = None


# API Response Models
class QuoteRunRequest(BaseModel):
    submission: QuoteSubmission
    use_agentic: bool = False  # Enable agentic behavior
    additional_answers: Optional[Dict[str, Any]] = None  # Answers to missing info questions


class QuoteRunResponse(BaseModel):
    run_id: str
    status: str
    decision: Optional[Dict[str, Any]] = None
    premium: Optional[Dict[str, Any]] = None
    citations: Optional[list] = None
    required_questions: Optional[list] = None
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

class HumanReviewRecord(BaseModel):
    model_config = ConfigDict(json_encoders={
        datetime: lambda v: v.isoformat()
    })
    
    run_id: str
    status: str  # e.g., "pending_review", "human_approved"
    requires_human_review: bool = True
    final_decision: Optional[str] = None  # e.g., "ACCEPT", "REJECT", "REFER"
    reviewer: Optional[str] = None
    review_timestamp: Optional[datetime] = None
    approved_premium: Optional[float] = None
    reviewer_notes: Optional[str] = None
    review_priority: Optional[str] = None
    assigned_reviewer: Optional[str] = None
    estimated_review_time: Optional[str] = None
    submission_timestamp: Optional[datetime] = None
    review_deadline: Optional[datetime] = None


class QuoteRecord(BaseModel):
    model_config = ConfigDict(json_encoders={
        datetime: lambda v: v.isoformat()
    })
    
    run_id: str
    status: str  # e.g., "completed", "processing"
    timestamp: datetime
    message: str
    processing_time_ms: int
    submission: Dict[str, Any]  # Original submission data
    decision: Optional[Dict[str, Any]] = None  # Decision details
    premium: Optional[Dict[str, Any]] = None  # Premium calculation
    rce_adjustment: Optional[Dict[str, Any]] = None  # RCE adjustment info
    requires_human_review: bool = False
    human_review_details: Optional[Dict[str, Any]] = None
    required_questions: Optional[List[Dict[str, Any]]] = None
    citations: Optional[List[Dict[str, Any]]] = None
