"""
Minimal agent implementations for underwriting workflow
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from app.underwriting_rules import (
    build_rule_query,
    evaluate_underwriting_rules,
    findings_as_risk_factors,
)
from models.schemas import DecisionPacket, DecisionType, HO3Submission

logger = logging.getLogger(__name__)

class BaseAgent:
    """Base agent class"""
    
    def __init__(self, prompt_version: str = "v1.0"):
        self.prompt_version = prompt_version
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process data"""
        return data

class IntakeNormalizerAgent(BaseAgent):
    """Intake normalization agent"""
    
    def normalize(self, submission_data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize submission data"""
        logger.info("Normalizing submission data")
        missing_info = []
        questions = []
        risk = submission_data.get('risk', {})
        applicant = submission_data.get('applicant', {})
        
        # Check for missing required fields
        if _is_missing_or_uncertain(applicant.get('full_name')):
            missing_info.append('applicant_name')
            questions.append(_question(
                "applicant_name",
                "applicant.full_name",
                "What is the applicant's full legal name?",
                "text",
            ))
        
        if _is_missing_or_uncertain(risk.get('property_address')):
            missing_info.append('property_address')
            questions.append(_question(
                "property_address",
                "risk.property_address",
                "What is the full property address?",
                "text",
            ))

        occupancy = risk.get('occupancy')
        if occupancy not in {"owner_occupied_primary", "owner_occupied_secondary", "tenant_occupied", "vacant"}:
            missing_info.append('occupancy')
            questions.append(_question(
                "occupancy",
                "risk.occupancy",
                "What is the property's occupancy?",
                "choice",
                options=["owner_occupied_primary", "owner_occupied_secondary", "tenant_occupied", "vacant"],
            ))

        if _is_missing_or_uncertain(risk.get('roof_age_years')):
            missing_info.append('roof_age_years')
            questions.append(_question(
                "roof_age_years",
                "risk.roof_age_years",
                "What is the roof age in years?",
                "numeric",
            ))
        
        return {
            "normalized_data": submission_data, 
            "status": "normalized",
            "missing_info": missing_info,
            "questions": questions
        }
    
    def process(self, submission_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process submission data"""
        return self.normalize(submission_data)

class PlannerRouterAgent(BaseAgent):
    """Planner router agent"""
    
    def route(self, submission_data: Any, missing_info: List) -> Dict[str, Any]:
        """Route to appropriate processing path"""
        logger.info("Routing submission")
        if missing_info:
            return {"route": "waiting_for_info", "reason_codes": missing_info, "data": submission_data}
        return {"route": "standard", "data": submission_data}
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process routing"""
        return self.route(data, [])

class EnrichmentAgent(BaseAgent):
    """Enrichment agent"""
    
    def enrich(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich data with external information"""
        logger.info("Enriching data")
        submission = _coerce_submission(data)
        address = submission.risk.property_address.lower()

        wildfire_band = "Low"
        flood_sfha = False
        territory = "MediumRiskCounty"

        if any(token in address for token in ["severe fire zone", "severe wildfire"]):
            wildfire_band = "Severe"
            territory = "HighRiskCounty"
        elif any(token in address for token in ["sacramento", "santa rosa", "paradise", "fire zone"]):
            wildfire_band = "High"
            territory = "HighRiskCounty"
        if any(token in address for token in ["river", "fresno"]):
            flood_sfha = True
            territory = "HighRiskCounty"
        if any(token in address for token in ["palo alto", "san diego", "oakland"]):
            territory = "LowRiskCounty"

        hazard_profile = {
            "wildfire_band": wildfire_band,
            "wildfire_risk_score": {"Low": 0.15, "Moderate": 0.45, "High": 0.78, "Severe": 0.95}[wildfire_band],
            "flood_sfha": flood_sfha,
            "flood_risk_score": 0.82 if flood_sfha else 0.2,
            "wind_risk_score": 0.25,
            "earthquake_risk_score": 0.35,
        }

        return {
            "submission": submission.model_dump(),
            "property_profile": {
                "address": submission.risk.property_address,
                "occupancy": submission.risk.occupancy,
                "dwelling_type": submission.risk.dwelling_type,
                "year_built": submission.risk.year_built,
                "roof_age_years": submission.risk.roof_age_years,
                "construction_type": submission.risk.construction_type,
                "wildfire_mitigation_evidence": submission.risk.wildfire_mitigation_evidence,
                "territory": territory,
            },
            "hazard_profile": hazard_profile,
            "confidence_map": {
                "property_profile": 0.9,
                "hazard_profile": 0.65,
                "note": "Hazards are provided by the deterministic enrichment profile."
            },
            "retrieval_plan": {
                "query": _build_retrieval_query(submission, hazard_profile),
                "filters": ["HO3", "homeowners"],
                "limit": 10
            }
        }
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process enrichment"""
        return self.enrich(data)

class RetrievalAgent(BaseAgent):
    """Retrieval agent"""

    def __init__(self, rag_engine=None, prompt_version: str = "v1.0"):
        super().__init__(prompt_version=prompt_version)
        self.rag_engine = rag_engine
    
    def retrieve(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve relevant guidelines and rules"""
        logger.info("Retrieving guidelines")
        plan = data.get("retrieval_plan", {})
        query = plan.get("query", "homeowners underwriting eligibility referral decline roof wildfire flood")
        limit = plan.get("limit", 8)
        chunks = self.rag_engine.retrieve(query, n_results=limit) if self.rag_engine else []
        return {
            "query": query,
            "retrieved_chunks": [chunk.model_dump() for chunk in chunks],
            "retrieval_metrics": {
                "chunk_count": len(chunks),
                "retriever": "rag_engine" if self.rag_engine else "none"
            },
            "data": data
        }
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process retrieval"""
        return self.retrieve(data)

class UnderwritingAssessorAgent(BaseAgent):
    """Underwriting assessment agent"""
    
    def assess(self, data: Dict[str, Any], retrieval_result: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make underwriting decisions"""
        logger.info("Assessing underwriting risk")

        submission = _coerce_submission(data.get("submission", data))
        hazard_profile = data.get("hazard_profile", {})
        property_profile = data.get("property_profile", {})
        citations = _extract_citations(retrieval_result or {})

        evaluation = evaluate_underwriting_rules(
            submission,
            hazard_profile=hazard_profile,
            property_profile=property_profile,
        )
        risk_factors = findings_as_risk_factors(evaluation.findings)

        return {
            "decision": evaluation.decision.value,
            "confidence": evaluation.confidence,
            "eligibility_score": evaluation.eligibility_score,
            "risk_factors": risk_factors,
            "reasoning": _reason_summary(evaluation.decision.value, risk_factors),
            "citations": citations,
            "facts_used": evaluation.facts_used,
            "ruleset_version": evaluation.ruleset_version,
            "rules_fired": [finding.rule_id for finding in evaluation.findings],
        }
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process assessment"""
        return self.assess(data)

class VerifierGuardrailAgent(BaseAgent):
    """Verifier guardrail agent"""
    
    def verify(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Verify decisions and citations"""
        logger.info("Verifying decision")
        citations = data.get("citations", [])
        decision = data.get("decision", "REFER")
        if decision in {"REFER", "DECLINE"} and not citations:
            return {
                "verified": False,
                "decision_allowed": False,
                "forced_decision": "REFER",
                "reason": "Referral or decline decisions require guideline citations."
            }
        return {
            "verified": True,
            "decision_allowed": True,
            "citation_count": len(citations),
            "data": data
        }
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process verification"""
        return self.verify(data)

class DecisionPackagerAgent(BaseAgent):
    """Decision packager agent"""
    
    def package(self, decision_data: Dict[str, Any], rating_data: Any, citations: List) -> DecisionPacket:
        """Package final decisions"""
        logger.info("Packaging decision")

        decision_value = decision_data.get("decision") or decision_data.get("preliminary_decision", "ACCEPT")
        decision = DecisionType(decision_value)
        evidence = decision_data.get("citations") or citations
        review_codes = [
            factor.get("code", "REVIEW")
            for factor in decision_data.get("risk_factors", [])
        ]

        next_steps = decision_data.get("next_steps")
        if not next_steps and decision_data.get("required_questions"):
            next_steps = ["Provide required information and resume this quote run."]

        return DecisionPacket(
            decision=decision,
            decision_confidence=decision_data.get("confidence", 0.8),
            reason_summary=decision_data.get("reasoning", "Decision completed"),
            citations=evidence,
            premium_indication=rating_data if isinstance(rating_data, dict) else {"annual_premium": rating_data, "currency": "USD"},
            needs_human_review=decision != DecisionType.ACCEPT,
            review_reason_codes=review_codes,
            next_steps=next_steps or _next_steps(decision, review_codes),
            facts_used=decision_data.get("facts_used", {}),
            evidence_cited=[citation.get("chunk_id", "") for citation in evidence if isinstance(citation, dict)],
            tool_calls_summary=citations if evidence is not citations else [],
            trace_link=f"trace://decision/{datetime.now().timestamp()}"
        )
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process packaging"""
        return self.package(data, None, [])


def _coerce_submission(data: Any) -> HO3Submission:
    if isinstance(data, HO3Submission):
        return data
    if isinstance(data, dict) and "applicant" in data and "risk" in data and "coverage_request" in data:
        return HO3Submission(**data)
    if isinstance(data, dict) and "submission" in data:
        return _coerce_submission(data["submission"])
    raise ValueError("Expected HO3 submission data")


def _is_missing_or_uncertain(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "unknown", "unk", "tbd", "n/a", "na", "unsure", "uncertain"}
    return False


def _question(
    question_id: str,
    field_path: str,
    question: str,
    question_type: str,
    options: Optional[List[str]] = None,
) -> Dict[str, Any]:
    payload = {
        "question_id": question_id,
        "field_path": field_path,
        "answer_key": field_path,
        "question": question,
        "question_text": question,
        "question_type": question_type,
        "required": True,
    }
    if options:
        payload["options"] = options
    return payload


def _build_retrieval_query(submission: HO3Submission, hazard_profile: Dict[str, Any]) -> str:
    evaluation = evaluate_underwriting_rules(submission, hazard_profile=hazard_profile)
    return build_rule_query(evaluation)


def _extract_citations(retrieval_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    citations = []
    for chunk in retrieval_result.get("retrieved_chunks", [])[:5]:
        citations.append({
            "chunk_id": chunk.get("chunk_id"),
            "doc_id": chunk.get("doc_id"),
            "doc_version": chunk.get("doc_version"),
            "section": chunk.get("section"),
            "relevance_score": chunk.get("relevance_score"),
            "excerpt": chunk.get("text", "")[:240],
        })
    return citations


def _reason_summary(decision: str, risk_factors: List[Dict[str, Any]]) -> str:
    if not risk_factors:
        return "Risk is eligible under the governed HO3 rules; no referral or decline triggers were found."
    reasons = "; ".join(factor["because"] for factor in risk_factors)
    return f"{decision} based on underwriting triggers: {reasons}"


def _next_steps(decision: DecisionType, review_codes: List[str]) -> List[str]:
    if decision == DecisionType.ACCEPT:
        return ["Proceed to bind subject to normal producer confirmation."]
    if decision == DecisionType.DECLINE:
        return ["Document decline rationale and provide required adverse-action review."]
    if "WILDFIRE_HIGH" in review_codes:
        return ["Request defensible-space evidence and route to underwriting review."]
    if "FLOOD_SFHA" in review_codes:
        return ["Confirm flood insurance status and route to underwriting review."]
    return ["Route to underwriting review with cited risk factors."]
