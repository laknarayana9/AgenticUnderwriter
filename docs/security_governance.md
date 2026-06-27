# Security & data governance

This is a **portfolio demo**, not a production deployment. This document states
plainly what is implemented today versus what a regulated insurance deployment
would require — because for a system that decides on people's coverage, the
honest gap list matters more than a green checkmark.

## Implemented today

| Control | Where | Notes |
|---|---|---|
| **PII masking before model egress** | `app/pii_masker.py` | A mask map built from the submission scrubs **every** prompt (generator and critic) before it leaves the process. Applicant name/email/phone/address never reach a hosted LLM. |
| **Local-only inference option** | Ollama providers (text + vision) | `LLM_PROVIDER=ollama` / `VISION_PROVIDER=ollama` keep all data — including property photos — on-device. Zero external egress. |
| **Minimal data retention for images** | `app/vision_service.py` | Only the image **SHA-256** is stored as provenance; raw photos are not persisted. |
| **Deterministic, cited, auditable decisions** | `underwriting_rules.py`, audit events | Every decision traces to fired rules + retrieved citations + a versioned ruleset (`RULESET_VERSION`). `GET /runs/{id}/audit` exposes the full chain. |
| **Input validation at the boundary** | Pydantic intake models | Typed enums and bounded ranges reject malformed input before any logic runs. |
| **Rate limiting** | `modal_app.py` | In-memory limiter (single-container demo). |
| **Secrets via env / Modal Secrets** | — | No keys committed; providers degrade gracefully when keys are absent. |

## Not implemented — required before production (designed, not built)

These are **deliberate scope cuts for a demo**, called out so they aren't mistaken
for oversights:

1. **Authentication & authorization.** The API is currently **unauthenticated** —
   the single biggest gap for a regulated claim. Production needs: OIDC/JWT (or
   mTLS for service-to-service) on every endpoint; **RBAC** distinguishing
   producer / underwriter / admin (e.g. only an underwriter may override a
   decision via `/reviews/{id}/actions`); and per-tenant/carrier isolation.
2. **PII at rest.** Submissions (incl. applicant name/email/phone) are stored in
   SQLite in the clear. Production needs: encryption at rest, field-level
   encryption for PII, and a **retention + right-to-erasure** workflow
   (CCPA/GDPR) keyed by `run_id`, with a configurable TTL/expiry job.
3. **Tamper-evident audit + access logging.** The audit trail is informative but
   not tamper-evident; production needs append-only/signed audit storage and
   who-viewed-what access logs (regulators ask).
4. **Distributed rate limiting / WAF** once `max_containers > 1` (the in-memory
   limiter is single-container only), plus abuse/DoS protection.
5. **Durable, access-controlled state.** Move off SQLite (app DB *and* the
   LangGraph checkpoint DB) to a managed, encrypted store; today both are
   single-writer local files.

## Threat model (what this architecture already resists)

- **Prompt injection cannot change a decision.** Because the LLM is out of the
  eligibility decision path (ADR-0001), a malicious instruction in a submission or
  guideline cannot flip ACCEPT/REFER/DECLINE — the deterministic rules decide.
  Worst case is misleading *rationale prose*, which the generator–critic loop and
  the deterministic citation pre-check constrain (and `docs/failure-modes.md`
  documents the residual fail-open).
- **Hallucinated citations** are caught deterministically (the critic pre-check
  rejects any cited chunk not actually retrieved).
- **Guideline/data poisoning** is bounded by a versioned, controlled corpus
  (`doc_version` on every citation) — though there is no automated freshness gate
  (failure mode #4).
- **Vision PII exposure** (faces/plates in photos) is mitigated by the local
  vision provider and SHA-only storage (failure mode #11).

## Model & decision governance

- The eligibility logic is a **versioned ruleset** (`RULESET_VERSION`), so any
  decision is reproducible from `(submission, ruleset_version, corpus_version)` —
  the property a state insurance regulator or an adverse-action review needs.
- Model swaps do not change decisions (the LLM is out of the path), so model
  governance is decoupled from decision governance — you re-validate wording
  quality, not decision quality, on a model change.

## The honest one-liner

The architecture is strong on **decision auditability and data-egress control**
(masking + local inference), and deliberately **unbuilt on perimeter security**
(authn/authz, encryption at rest, erasure). For a real deployment, items 1–2 above
are non-negotiable and would be the first work.
