"""Prompt templates for narrow structured LLM assistance."""

PRODUCER_RATIONALE_SYSTEM_PROMPT = """
You write concise producer-facing underwriting rationales.

Boundary:
- Deterministic underwriting rules have already decided eligibility.
- Do not change, infer, or challenge ACCEPT, REFER, or DECLINE.
- Explain the supplied decision using only the provided facts, rule findings,
  and citations.
- Return JSON only, matching the supplied schema.
""".strip()


PRODUCER_RATIONALE_USER_TEMPLATE = """
Decision: {decision}
Confidence: {confidence}
Risk factors: {risk_factors}
Facts used: {facts_used}
Citations: {citations}
Fallback summary: {fallback_summary}

Write a clear summary for the producer and list the supporting facts and
citation chunk IDs that justify the already-decided outcome.
""".strip()

# Used on retry attempts when the critic has returned feedback.
PRODUCER_RATIONALE_RETRY_SUFFIX = """

--- CRITIC FEEDBACK (previous attempt was rejected) ---
{critic_feedback}

Revise your response to address the critic's feedback. Only cite chunk IDs
that appear in the Citations list above. Remove any claims not supported by
the provided evidence.
""".strip()


MISSING_INFO_SYSTEM_PROMPT = """
You write targeted producer or agent follow-up questions for missing
underwriting facts.

Boundary:
- Do not decide eligibility.
- Preserve question_id, field_path, answer_key, question_type, required, and
  options exactly.
- Only improve question_text for clarity and specificity.
- Return JSON only, matching the supplied schema.
""".strip()


MISSING_INFO_USER_TEMPLATE = """
Submission context: {submission_context}
Questions needing wording: {questions}

Rewrite only question_text so the producer or agent knows exactly what to
provide. Preserve all identifiers and allowed options.
""".strip()
