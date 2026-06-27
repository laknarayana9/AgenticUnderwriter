# Failure-modes catalog

Companion to [ADR-0001](adr/0001-llm-out-of-decision-path.md). Keeping the LLM
out of the eligibility decision path closes off one entire failure class
(model-induced decision drift) but not all of them. This catalog enumerates
eleven ways the system can be wrong, the guardrail(s) that catch each, and the
**residual risk** that remains. Residual risk is stated honestly — a guardrail
that "fails open" is called out as such.

Severity is the impact if the failure reaches a bound decision; Likelihood is
rough and qualitative.

| # | Failure mode | Primary guardrail | Residual risk |
|---|--------------|-------------------|---------------|
| 1 | Model-induced decision drift | LLM excluded from decision path (ADR-0001) | None for the decision |
| 2 | Hallucinated citation in rationale | Deterministic citation pre-check | Very low |
| 3 | Unsupported fact in rationale | Generator–critic faithfulness loop | Critic fails open if unavailable |
| 4 | Stale guideline corpus | Versioned citations + startup re-ingest | No automated freshness alert |
| 5 | Retrieval miss | Hybrid retrieval + recall gate + citation guardrail | Recall < 1.0 at low k |
| 6 | Hallucinated intake field (extractor) | Abstention training + missing-info gate | A wrong non-null value bypasses the gate |
| 7 | Intake normalization error | Pydantic enums + missing-info gate | Valid-but-wrong values pass |
| 8 | Edge-case rule gap | Stratified eval suite + severity precedence | Unknown-unknowns |
| 9 | PII leakage to provider | PII masking before every prompt | Free-text PII outside known fields |
| 10 | Silent LLM failure / fail-open | Deterministic fallbacks everywhere | Unverified rationale text may ship |
| 11 | Vision misread / hallucinated attribute | Abstention + confidence gate + missing-info gate | A confident-but-wrong attribute folds in; PII in raw photos |

---

## 1. Model-induced decision drift

**Failure:** a model upgrade, provider swap, or temperature change shifts which
submissions are accepted, referred, or declined.

**Guardrail:** the eligibility decision is produced by `app/underwriting_rules.py`,
not by any model (ADR-0001). The CI eval gate runs with
`LLM_STRUCTURED_OUTPUT_ENABLED=false` to prove the rules engine and retrieval
decide on their own.

**Residual risk:** none for the decision. Rationale *wording* can vary across
models; that is intended and is bounded by failure mode 3.

## 2. Hallucinated citation in the rationale

**Failure:** the LLM rationale cites a `chunk_id` that was never retrieved.

**Guardrail:** a deterministic pre-check in `workflows/critic.py` rejects any
cited id not present in the retrieved set — **before** any critic LLM call, so it
holds even when the critic model is unavailable. The `VerifierGuardrailAgent`
separately requires every REFER/DECLINE to carry citations.

**Residual risk:** very low. The check is exact-match on retrieved ids.

## 3. Unsupported fact in the rationale

**Failure:** the rationale asserts a fact not supported by the evidence (e.g.
"prior claims") even though every cited id is real.

**Guardrail:** the generator–critic loop runs a separate critic model to verify
faithfulness against the retrieved excerpts; on failure it retries with feedback,
then **fails closed to a deterministic rationale**.

**Residual risk — stated plainly:** when the critic model is *unavailable*, the
LLM faithfulness check currently **fails open** (the deterministic citation
pre-check still runs, but an unsupported-but-cited claim can pass). A
fail-closed-to-deterministic-rationale fix is designed; until it lands, this is a
known gap. The decision is unaffected regardless, and adverse decisions route to
human review.

## 4. Stale guideline corpus

**Failure:** the system decides against a superseded version of a guideline.

**Guardrail:** every citation carries `doc_id` + `doc_version`; the ruleset is
versioned (`RULESET_VERSION`); the corpus is re-ingested on startup; and the
audit trail records both versions, so any decision is reproducible against the
exact corpus that produced it.

**Residual risk:** there is **no automated freshness check or alert** on corpus
age — a guideline that should have been updated but wasn't will be applied
silently. Mitigation today is the versioned audit trail (detectable after the
fact) and document-owner process, not an automated gate.

## 5. Retrieval miss

**Failure:** the relevant guideline is not retrieved, so an adverse decision
lacks its supporting citation.

**Guardrail:** hybrid BM25 + dense retrieval with a cross-encoder reranker
(`app/rag_engine.py`, see `docs/retrieval_eval.md`); a `retrieval_recall` CI gate;
and — critically — the citation guardrail **fails safe**: a REFER/DECLINE with no
citation is forced to a referral rather than shipped uncited.

**Residual risk:** recall is not 1.0 at low k on the synthetic corpus (measured
in `docs/retrieval_eval.md`). A miss degrades to a referral (safe), not to a
wrong accept.

## 6. Hallucinated intake field (extraction front-end)

**Failure:** the optional fine-tuned extractor invents a value for a field the
producer note never stated (e.g. a roof age), and the workflow proceeds on the
fabricated fact.

**Guardrail:** the fine-tune is trained for **abstention** (null on unstated
fields), scored by `refusal_correctness`; a null then triggers the deterministic
missing-info gate, which pauses to ask. `scripts/extraction_workflow_demo.py`
demonstrates this contrast end to end.

**Residual risk — the important one:** the missing-info gate catches *missing*
(null) fields, **not a confidently wrong non-null value**. A hallucinated but
plausible roof age bypasses the pause. Mitigations: the extractor is optional and
off the default path; extraction quality is measured (exact-match + refusal
correctness); and adverse decisions still reach human review. Do not enable the
extractor on the hot path without monitoring its field accuracy.

## 7. Intake normalization error

**Failure:** a supplied field is mis-normalized (e.g. occupancy mapped to the
wrong enum).

**Guardrail:** the API boundary validates against Pydantic enums (`IntakeRiskProfile`
etc.), rejecting invalid values outright; uncertain/blank values route to the
missing-info gate.

**Residual risk:** a value that is *valid but wrong* (a real enum, incorrect for
this risk) passes validation. Caught only by downstream human review, not
automatically.

## 8. Edge-case rule gap

**Failure:** a submission combination no rule covers yields ACCEPT when it should
have referred.

**Guardrail:** the 196-case **stratified** eval suite exercises accept/refer/
decline/missing combinations; severity precedence (`DECLINE > REFER > ACCEPT`)
means any fired trigger dominates; ACCEPT is only produced when *no* trigger
fires.

**Residual risk:** unknown-unknowns — a risk pattern outside the rule set. This is
the inherent rule-maintenance burden (ADR-0001 Consequences). Mitigation is
additive: every new pattern becomes a rule plus eval cases, guarded by the
regression gate.

## 9. PII leakage to a provider

**Failure:** applicant PII (name, email, phone, address) is sent to an external
LLM.

**Guardrail:** `app/pii_masker.py` computes a mask map from the original
submission and scrubs **every** prompt — generator and critic — before it leaves
the process. The eval harness includes a `pii_leak_rate` check, and external LLM
calls are disabled by default.

**Residual risk:** masking covers known structured PII fields. PII embedded in
*free text* (e.g. a producer note that names a third party) outside those fields
is not guaranteed to be masked.

## 10. Silent LLM failure / fail-open

**Failure:** a provider is down or returns junk, and the system proceeds without
the assistance the LLM was providing.

**Guardrail:** deterministic fallbacks throughout — rationale falls back to
`_reason_summary`, missing-info wording to templated copy — and the decision is
rules-based, so it is unaffected. Streaming-monitor timeouts (`streaming/explain.py`)
prevent the ops path from hanging on a slow model.

**Residual risk:** overlaps with failure mode 3 — a *down critic* currently lets
unverified rationale text ship. The decision and citations remain valid; the
exposure is misleading explanatory prose, surfaced to a human reviewer who can
compare it against the structured decision and `facts_used`.

---

## 11. Vision misread / hallucinated attribute

**Failure:** the optional vision intake reads a property photo wrong — e.g. claims
defensible space that isn't there — and folds a false value into
`wildfire_mitigation_evidence`.

**Guardrail:** the vision model is trained/prompted to **abstain** (`visible=false`)
when it cannot assess an attribute; `fold_vision_into_submission` applies a value
only when it is visible **and** above `VISION_MIN_CONFIDENCE`, and **never
overwrites** a producer-supplied value. Below threshold or abstained → the
deterministic missing-info gate asks a human. The decision remains rules-owned.

**Residual risk — the important one:** a **confident-but-wrong** non-null
attribute clears the gate (the abstention/confidence guard catches *uncertainty*,
not *confident error*) — the same shape as failure mode #6 for text extraction.
Mitigations: vision is off the default path; only `defensible_space_present` maps
to a rule, and that path still routes high-wildfire risks to human review;
extraction quality is measured (`evals/vision_eval.py`). **PII:** property photos
may contain faces/plates/house numbers; hosted providers send the image out, so
the **local Ollama vision provider (`VISION_PROVIDER=ollama`) keeps photos
on-device** — closing the privacy gap — and only the image SHA is stored.

## How these are caught in aggregate

No single mechanism catches everything. The defense is layered:

- **Deterministic decision spine** — closes failure mode 1 entirely.
- **Citation pre-check + critic loop** — failure modes 2–3.
- **Versioned corpus + audit trail** — failure mode 4 (detection, not prevention).
- **Hybrid retrieval + recall gate + fail-safe referral** — failure mode 5.
- **Abstention training + missing-info gate** — failure mode 6 (partial).
- **Schema validation** — failure mode 7.
- **Stratified eval + severity precedence** — failure mode 8.
- **PII masking** — failure mode 9.
- **Deterministic fallbacks + timeouts** — failure mode 10.
- **Human-in-the-loop review** — the backstop for the residual risk in 3, 6, 7,
  and 10: every non-ACCEPT decision reaches a human, who sees the structured
  decision, fired rules, citations, and critic verdicts.

The honest summary: the architecture eliminates the highest-severity failure
(decision drift) by construction and degrades safely (toward referral and human
review) on most of the rest. The two residual risks worth active attention are
**critic fail-open (3/10)** and **extraction hallucination of a non-null value
(6)** — both are scoped, monitored, and backstopped by human review, and both
have a clear path to closure.
