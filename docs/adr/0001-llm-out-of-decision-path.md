# ADR 0001: The LLM is not in the eligibility decision path

- **Status:** Accepted
- **Date:** 2026-05-03
- **Decision owner:** Engineering
- **Context:** Agentic Underwriter — Governed AI Underwriting Workflow Platform

## Context

Agentic Underwriter is a workflow platform that converts homeowner-insurance (HO3) submissions into ACCEPT / REFER / DECLINE recommendations with cited evidence and a full audit trail. The system operates in a regulated insurance context where a recommendation can lead to a binding action (a quote issued, a submission referred to an underwriter, an applicant declined) and where every decision must be:

- **Reproducible** — the same inputs must produce the same recommendation, deterministically, across runs and across deployments.
- **Auditable** — for any decision, an underwriter, compliance reviewer, or regulator must be able to trace the recommendation back to specific rules and source guideline passages.
- **Defensible** — the reasoning behind a decision must withstand challenge from a state insurance regulator, a producer disputing a referral, or an applicant disputing a decline.
- **Stable under model change** — swapping or upgrading the language model must not change which submissions are accepted, referred, or declined.

These are not aspirational properties. They are operating constraints in regulated insurance underwriting, and they are the constraints that determine whether a system like this can be productionized at all.

The central architectural question of the system is therefore: **what role should the language model play in producing a decision?**

## Decision

**The eligibility decision is owned entirely by a deterministic rules engine. The language model is excluded from the decision path.**

Concretely:

- The `UnderwritingAssessorAgent` evaluates `app/underwriting_rules.py` against a normalized submission and produces the decision (ACCEPT / REFER / DECLINE) and the structured `reason_codes`.
- The LLM is invoked in exactly two narrow places, both downstream of the decision:
  1. **Producer rationale** — a natural-language summary of the already-made decision, intended for the producer/agent submitting the application.
  2. **Missing-info question wording** — given a structured list of missing fields produced by `IntakeNormalizerAgent`, the LLM rewrites the questions into producer-friendly language.
- Both LLM calls go through `StructuredLLMService` with Pydantic-validated output schemas and deterministic fallback paths. If the model is unavailable or returns invalid output, the workflow falls back to templated copy and the decision is unaffected.
- The `VerifierGuardrailAgent` performs a final consistency check: it verifies that the rendered rationale references the same rule set and citations the rules engine actually fired. A mismatch fails the run.

The result: every ACCEPT, REFER, and DECLINE in the system is reproducible from `(submission, ruleset_version, guideline_corpus_version)` alone. The model can be swapped, upgraded, or removed without changing a single decision.

## Alternatives considered

### Alternative 1: LLM-as-judge with rules as guardrails

A common pattern is to let the LLM produce a recommendation, then apply rules as post-hoc filters that override the model when it disagrees with hard constraints.

**Rejected because:**
- Reproducibility breaks under model upgrades. Sonnet → Opus, or any temperature drift, can shift decisions in ways that are difficult to detect or attribute.
- The audit trail becomes a probabilistic artifact rather than a deterministic one. "The model said X, then the rule overrode it" is not a clean story for a regulator.
- The system inherits the model's failure modes (hallucinated citations, sycophancy, prompt-injection sensitivity) on the hot path, even when guardrailed.

### Alternative 2: LLM as orchestrator (agentic ReAct loop)

Let the LLM plan the workflow — choosing which tools to call, when to retrieve, when to ask for missing info, when to declare a decision.

**Rejected because:**
- The orchestration in homeowner underwriting is fundamentally not open-ended. The submission shape is well-defined, the rule set is small (~tens of rules), and the workflow stages are stable. The benefits of dynamic planning are minimal; the costs (non-determinism, latency variance, cost variance, debuggability) are substantial.
- Audit logs of LLM-driven orchestration ("the agent decided to retrieve again") are harder to reason about than deterministic state-machine transitions.
- This is the most common reason "agent-style" systems struggle in regulated environments. The right pattern for a well-bounded workflow is an explicit state machine with the LLM as a tool, not a planner.

### Alternative 3: LLM-only with prompt-based "rules"

Encode the rules in the prompt and rely on the model to apply them.

**Rejected because:**
- This is the regulated-AI equivalent of putting business logic in spreadsheets: it appears to work until the moment it stops, with no mechanism for catching the failure.
- Rule changes become prompt changes, which become model behavior changes, which require full eval re-validation. The blast radius of a small policy update is the entire system.

### Alternative 4 (rejected): Pure rules with no LLM at all

Do everything deterministically — including the producer-facing copy.

**Considered, rejected because:**
- The two LLM tasks the system *does* use the model for are exactly the tasks where natural-language flexibility adds genuine value: a producer reading "Roof age exceeds 25 years; please confirm last roof replacement date and provide an inspection report if available" is meaningfully better-served than by a fixed string. These calls don't affect the decision, only the explanation, and they have a deterministic fallback.
- The narrow, bounded use of the LLM is the right point on the curve: capture the language-quality benefit, take none of the decision-stability risk.

## Consequences

### Positive

- **Reproducibility.** The 196-case CI eval gate enforces 100% decision accuracy, 100% reason-code match, and 100% retrieval recall@5 deterministically on every commit. A model regression cannot break this gate, because the model is not in the path.
- **Auditability.** Every decision has a deterministic chain: submission fields → rules fired → reason codes → retrieved citations → producer rationale. The audit log captures each transition.
- **Stability under model change.** The system can adopt a new model tier or provider without re-running the decision-quality eval. We re-run only the rationale-quality and missing-info-wording evals.
- **Cost predictability.** LLM cost is bounded — at most two calls per submission (rationale, missing-info), and zero calls when the deterministic fallback is used.
- **Failure isolation.** When the LLM is unavailable or producing bad output, the system degrades to templated copy. Decisions continue to be produced and remain valid.

### Negative

- **The "agentic AI" framing is weakened.** The system is correctly described as a constrained-LLM RAG workflow, not as an agent. This is a positioning cost, but it is honest, and senior reviewers consistently rate honesty about scope above marketing language.
- **Less benefit from model improvements on the decision quality axis.** A smarter model does not produce better decisions in this architecture. This is a deliberate trade — we accept it because decision *correctness* is a rules problem, not a reasoning problem, in homeowner underwriting.
- **Rule maintenance is a real engineering burden.** New peril types, new states with different regulations, and new product lines all require rule additions. The mitigation is the eval harness: any rule change is caught by the regression gate.

### Neutral

- This pattern does not generalize to underwriting domains where the rule set is genuinely fuzzy (e.g., specialty commercial lines, surplus lines with bespoke policy language). For those, a different architecture — likely with stronger LLM involvement and a much larger eval set — would be appropriate. ADR 0002 will capture that scope distinction when we extend.

## How the decision is enforced technically

Five mechanisms hold the line:

1. **Code structure.** `UnderwritingAssessorAgent` does not import or hold a reference to `StructuredLLMService`. The LLM service is constructed only by `IntakeNormalizerAgent` (for missing-info wording) and `DecisionPackagerAgent` (for rationale). Static inspection alone is sufficient to verify the LLM is not in the decision path.
2. **Schema boundaries.** `MissingInfoQuestionOutput` and `ProducerRationaleOutput` are the only schemas the LLM is permitted to populate. Neither contains a `decision` field. The Pydantic boundary makes a "decision leak" through the LLM mechanically impossible.
3. **Verifier guardrail.** `VerifierGuardrailAgent` cross-checks rendered rationale against the rules engine's output. A rationale that asserts a decision the rules did not produce fails the run.
4. **CI eval gate.** The 196-case regression suite is enforced at thresholds of 1.0 for decision accuracy, reason-code match, and retrieval recall@5. The gate runs with `LLM_STRUCTURED_OUTPUT_ENABLED=false` to prove the rules engine and retrieval are sufficient on their own.
5. **Audit events.** Every state transition emits a structured audit event recording the rule(s) fired, citations retrieved, and the decision produced. The trail is independent of LLM output.

## Failure modes catalog

This ADR is partnered with a failure-modes catalog enumerating ten ways the system can be wrong and the guardrail that catches each. The decision-path-exclusion ADR closes off an entire class of failures (model-induced decision drift) but does not close off all of them.

The remaining failure classes — stale guideline corpus, retrieval miss, edge-case rule gap, intake normalization error, etc. — are addressed by the eval harness, the verifier, the audit log, and the human-in-the-loop review queue. Each is documented in `docs/failure-modes.md`.

## Extending the decision

If we add new insurance lines (HO5, dwelling fire, condo, renters), the same pattern applies: a per-line rules module, a per-line guideline corpus, a per-line eval set with stratified cases. The LLM remains out of the decision path. The shared infrastructure (retrieval, verifier, audit log, review queue) does not change.

If we add a line where the rules genuinely cannot be enumerated (e.g., specialty commercial), we will write **ADR 0002** to scope that domain separately rather than relax this ADR. Mixing decision-stability assumptions across domains is a category error we will not introduce.

## References

- `app/underwriting_rules.py` — rule engine and decision logic
- `app/llm_service.py` — bounded LLM service with Pydantic schemas and fallback
- `workflows/agent_workflow.py` — 7-stage pipeline orchestration
- `workflows/agents.py` — stage handlers (deterministic; "Agent" is a role label, not an LLM driver)
- `evals/run.py` and `evals/datasets/ho3_labeled.jsonl` — 196-case CI regression suite
- `.github/workflows/ci.yml` — eval gate enforced at 1.0 thresholds
