# AgenticUnderwriter — Capabilities Reference

> **Purpose:** 3-minute interview cheat sheet. For each capability: what is real, what is mocked, and why.

---

## The One-Sentence Pitch

A governed agentic workflow that turns a homeowners insurance application into a cited ACCEPT / REFER / DECLINE decision — with deterministic rules as the source of truth, LLM strictly in an assistance role, and every design choice documented and testable without API keys.

---

## Capability Map

### 1. Deterministic Underwriting Rules Engine

**What it does:** Evaluates HO3 submissions against versioned eligibility rules (roof age, occupancy, construction year, wildfire band, coverage limits) and emits a structured `RuleEvaluation` with decision, confidence, reason codes, and citation queries.

**Real:** 100%. `app/underwriting_rules.py` is pure Python, no LLM, no external call. Every rule has a `rule_id`, a `reason_code`, a severity, and a `citation_query` that drives the retrieval step. The ruleset is versioned (`UW-RULESET-2026.04`).

**Mock:** None. This is the governed core.

**Why this matters in an interview:** "The LLM cannot override or influence the decision. `evaluate_underwriting_rules()` runs deterministically before any LLM call is made. This is ADR-0001 — LLM out of the decision path."

---

### 2. Seven-Step Agent Orchestration

**What it does:** Runs intake normalization → planning/routing → enrichment → retrieval → assessment → verifier guardrail → decision packaging as a sequential pipeline with named agent boundaries.

**Real:** The orchestration structure, agent interfaces, state machine, and data flow are all real. Every agent has a distinct role and communicates through `WorkflowState`.

**Mock:** The agents are thin wrappers — `PlannerRouterAgent` does literal routing logic (no ML), `EnrichmentAgent` uses hardcoded address-string matching instead of a real geocoder + FEMA API. `VerifierGuardrailAgent` does rule-based sanity checks.

**Why the mocks:** The interfaces are production-realistic. The comment in `EnrichmentAgent.enrich()` documents the exact production replacement: geocode → FEMA NFHL (flood) → Cal Fire FHSZ (wildfire) → CoreLogic (catastrophe). Swapping the implementation doesn't change the interface the rest of the pipeline sees.

---

### 3. Missing-Info Loop + Same-Run Resume

**What it does:** Detects missing or uncertain fields (roof age, occupancy, applicant name, property address) at two gates — intake normalization and post-enrichment contextual check. Pauses the workflow, generates LLM-worded follow-up questions, and resumes on the same `run_id` with the supplied answers, preserving full audit trail.

**Real:** 100% real. The pause/resume logic, audit event preservation, two-gate detection, and answer application are all real code exercised by the test suite.

**Mock:** The question wording goes through `StructuredLLMService.word_missing_info_questions()` — if `LLM_STRUCTURED_OUTPUT_ENABLED=false` (the default), a deterministic fallback returns the raw question text. The UX degrades gracefully; the logic doesn't change.

---

### 4. RAG Retrieval — Lexical, BM25, Hybrid, Reranking

**What it does:** Retrieves guideline chunks from a synthetic HO3 document corpus to ground decisions. Supports four modes: lexical (TF-IDF cosine), BM25, RRF hybrid fusion, and cross-encoder reranking.

**Real:** The retrieval algorithms (BM25, RRF with paper-correct k=60, cross-encoder reranking via `cross-encoder/ms-marco-MiniLM-L-6-v2`) are genuinely implemented in `app/rag_engine.py`. Chunking strategies (header-based semantic, fixed-size) are configurable and compared in `scripts/compare_chunking.py`.

**Mock / Synthetic:**
- The **document corpus** is a synthetic HO3 guidelines document written to match the underwriting rules — it is not a real insurance filing. This is unavoidable (real policy forms are proprietary).
- The **default embedding model** (`hashing-underwriting-v1`) is a deterministic hash function, not a real semantic embedding. This keeps CI hermetic with no model downloads. Real semantic embeddings (Nebius `BAAI/bge-en-icl`, or local `sentence-transformers/all-MiniLM-L6-v2`) are supported via env var.

**Why the mocks:** Synthetic corpus + hash embeddings = zero external dependencies, reproducible CI, identical results across machines. The retrieval *mechanics* are real; the *content* is illustrative.

---

### 5. PII Masking at LLM Boundary

**What it does:** Scrubs applicant name, email, phone, and property address from the submission dict before any LLM prompt is constructed. Applies an ephemeral token map (`[APPLICANT_NAME]`, etc.) and restores values in the response. The `pii_leak_rate` eval metric asserts zero PII appears in producer rationale output.

**Real:** 100% real. `app/pii_masker.py` is production-grade — field paths are explicit, the mask map is never persisted or logged, and the eval harness runs a regex + literal-value scan on every rationale output.

**Mock:** None.

---

### 6. Generator-Critic Loop for Rationale Grounding

**What it does:** After the decision is made, a generator LLM (OpenAI / Nebius / Claude / Gemini / Ollama) produces a producer-facing rationale. A separate critic LLM (defaults to Claude to avoid self-grading bias) reviews whether every claim is grounded in the retrieved evidence. The generator gets up to 2 retry attempts with structured feedback before the rationale is released.

**Real:** The multi-step retry loop, critic independence (separate provider by default), structured feedback schema, and deterministic pre-check (citation IDs must exist in retrieved chunks before the LLM call) are all real.

**Mock:** If `LLM_STRUCTURED_OUTPUT_ENABLED=false`, the generator returns a deterministic fallback rationale and the critic is skipped. Tests run entirely in this mode — no API key required.

**Why this matters:** "The critic uses Claude by default even when the generator uses GPT-4o. Self-grading bias is a well-documented failure mode in LLM evaluation — having an independent judge is a deliberate architectural decision, not an afterthought."

---

### 7. Vision Intake — Property Photo → Submission Enrichment

**What it does:** An optional fenced `vision_intake` stage at the top of the workflow accepts a property photo, extracts structured evidence (roof condition, stories, construction type, defensible space), and folds confident attributes into the submission before the rules run. Low-confidence or failed extractions leave the field null, triggering the missing-info gate.

**Real:** The vision service interface, confidence gating, fold-into-submission logic, and `vision_evidence_applied` audit event are all real. Two providers are implemented:
- `OpenAIVisionProvider` — GPT-4o via the OpenAI SDK
- `OllamaVisionProvider` — `llama3.2-vision` via local Ollama REST API (no egress, for PII-sensitive deployments)

**Mock / Stub:** The test suite uses a `FakeVisionProvider` that returns deterministic evidence dicts. No real images or API calls in CI.

**Why the providers:** "OpenAI for quality, Ollama for privacy. Property photos contain addresses and faces — for on-prem deployments, `VISION_PROVIDER=ollama` keeps all image data local. Same interface, zero code change."

---

### 8. HITL Review Queue

**What it does:** REFER and DECLINE decisions with `needs_human_review=True` are inserted into a review queue. Underwriters can view pending reviews, accept/reject with notes, and actions are recorded with timestamps.

**Real:** The queue logic, action recording, and API endpoints (`/reviews/pending`, `/reviews/{run_id}/actions`) are real. Persistence is SQLite via `UnderwritingDB`.

**Mock:** There is no real underwriter UI beyond the API. The Streamlit demo provides a basic interface. Access control (auth, role-based permissions) is not implemented — documented explicitly in `docs/security_governance.md` as an open gap with the production closure path.

---

### 9. Observability — Spans, Metrics, Streaming Monitor

**What it does:** Every workflow run emits spans (trace_id, span_id, stage name, duration_ms, status). Request-level metrics (latency p50/p95, citation coverage, failure rate, cost estimate, adverse decision rate) accumulate in a bounded ring buffer. A streaming monitor detects anomalies (cost spikes, latency outliers) and exposes them via `/monitor/anomalies` and a WebSocket feed.

**Real:** `observability.py` is fully real — custom span/tracer abstractions, `MetricsCollector` with percentile math, `StreamMonitor` with ring buffer. The `timed_stage()` context manager records per-stage latency into the workflow state for the `/latency-budget` endpoint.

**Mock:** The optional Langfuse sink (`_LangfuseSink`) and the OpenTelemetry exporter (`TRACE_BACKEND=otel`) are real adapter code but untested against live systems. Default backend is in-memory.

---

### 10. LangSmith Tracing + Dataset + Evaluators

**What it does:** `@traceable` wrapper on `run_workflow` sends traces to LangSmith when `LANGSMITH_TRACING=true`. `upload-dataset` pushes the 206-case golden set as a versioned LangSmith dataset. `run-eval` runs `client.evaluate` with three registered evaluators (`decision_accuracy`, `retrieval_recall@5`, `faithfulness`), producing a shareable experiment link with per-case scores and comparison view.

**Real:** All of it — real LangSmith SDK integration, real dataset upload, real `client.evaluate`. Tests use mocks; the live path requires `LANGSMITH_API_KEY`.

**Why:** "This closes the Week-4 assignment gap and puts a named tool on the résumé. The CI harness is the better artifact for correctness; LangSmith is the better artifact for showing stakeholders."

---

### 11. Dual-Engine Orchestration — Native + LangGraph

**What it does:** The same governed workflow runs on two interchangeable engines. The native engine is a hand-rolled explicit state machine (default). The LangGraph engine (`POST /quote/ho3/langgraph`) is a real `StateGraph` with durable pause/resume via a SQLite checkpointer — a paused run survives a process restart and resumes by `thread_id`.

**Real:** Both engines are real. Decision parity is CI-asserted across all 206 golden cases (`test_dual_engine_dataset_parity`). The LangGraph engine uses `interrupt()` for HITL, `add` reducers for accumulator fields, idempotent nodes, and a threading lock around the SQLite connection.

**Mock / Limitation:** The SQLite checkpointer serializes concurrent requests behind a mutex. This is correct for the single-instance demo; production path is `PostgresSaver` (one-line swap, same interface).

**Why two engines:** "LangGraph is the framework name hiring managers search for. The native engine demonstrates I can build the same thing from scratch. Running both with CI-asserted parity shows I understand what a framework is actually doing."

---

### 12. LLM-as-Judge Calibration

**What it does:** Measures judge-human agreement for the critic's faithfulness verdict on a 24-case hand-labeled dataset with injected defects (fabricated citations, unsupported claims, flipped conclusions). Reports agreement rate, Cohen's kappa, and fail-open rate (judge misses a bad rationale).

**Real:** The calibration harness, metric math (kappa calculation), and the 24 human-labeled cases are real. The `llm` backend uses the actual production critic prompt.

**Mock / Stub:** The default backend (`--backend simulated`) is a deterministic stand-in that scores based on keyword heuristics, not a real LLM. The committed snapshot and report were generated by the simulated backend, not a real judge run.

**Why the simulated backend:** "Calibration with a real LLM costs money and is non-reproducible in CI. The simulated backend lets you check metric math and fixture integrity deterministically. The `--backend llm` path runs the real judge when you have a key."

---

### 13. Fine-Tune Track (Nebius Token Factory)

**What it does:** A LoRA extraction fine-tune pipeline: generates structured extraction training pairs from the HO3 submission schema, formats them as JSONL, and submits a fine-tune job to Nebius Token Factory.

**Real:** The pipeline logic, JSONL generation, and submission script are real. The Nebius API integration uses their OpenAI-compatible endpoint.

**Mock:** The demo runs as a dry-run (`scripts/extraction_workflow_demo.py`) with no actual training data or live job without a `NEBIUS_API_KEY`. The fine-tuned model is not served anywhere in the live demo.

---

### 14. Multi-Provider LLM Support

**What it does:** The `StructuredLLMService` accepts pluggable providers: OpenAI, Nebius (OpenAI-compatible), Claude (Anthropic SDK), Gemini, Ollama, LangChain. All providers implement the same `StructuredJSONProvider` protocol. PII masking, critic loop, and deterministic fallback apply regardless of provider.

**Real:** All six providers are implemented. The protocol pattern means governance (PII masking, fallback, critic) is never bypassed by switching providers.

**Mock:** Gemini, Ollama, and LangChain providers are implemented but not integration-tested with live keys in CI. OpenAI and Claude are the two exercised in real runs.

---

## What to Say in 3 Minutes

**"The architecture has one hard rule — the underwriting decision comes from deterministic Python code, not an LLM. The LLM has three allowed roles: wording follow-up questions, generating a producer rationale, and grading that rationale as a critic. Those boundaries are tested, documented in ADR-0001, and enforced at the code level."**

**"The things that look real but are stubs: hazard enrichment uses address-string matching instead of FEMA/CoreLogic APIs, the RAG corpus is synthetic (real policy forms are proprietary), and the default embedding model is a hash function so CI runs with no downloads. Every stub has a comment documenting the production replacement and why the interface is the same."**

**"The things that are genuinely real: the retrieval algorithms (BM25, RRF, cross-encoder), the PII masking with eval-asserted leak rate, the generator-critic loop with independent judge, the LangGraph durable checkpointing, the 206-case eval harness, and the LangSmith experiment tracking."**

---

## Quick Mock vs Real Summary Table

| Capability | Real | Mocked / Stub | Reason for stub |
|---|---|---|---|
| Underwriting rules engine | ✅ Full | — | — |
| Agent orchestration structure | ✅ Full | — | — |
| Hazard enrichment | Interface ✅ | Logic (address strings) | FEMA/CoreLogic APIs are paid/complex |
| RAG retrieval algorithms | ✅ Full | — | — |
| RAG document corpus | Interface ✅ | Content (synthetic) | Real policy forms are proprietary |
| Default embeddings | Interface ✅ | Hash function | Zero CI dependencies |
| Missing-info loop / resume | ✅ Full | — | — |
| PII masking | ✅ Full | — | — |
| Generator-critic loop | ✅ Full | Fallback when no key | Graceful degradation |
| Vision intake (OpenAI) | ✅ Full | Fake provider in tests | No images in CI |
| Vision intake (Ollama) | ✅ Full | — | — |
| HITL review queue | ✅ Full | No auth / no real UI | Auth is infra, not AI scope |
| Observability / metrics | ✅ Full | — | — |
| LangSmith integration | ✅ Full | Mocked in unit tests | Live path needs key |
| LangGraph dual-engine | ✅ Full | SQLite instead of Postgres | Single-instance demo |
| LLM-as-judge calibration | ✅ Full | Simulated backend default | Reproducible CI, cost control |
| Fine-tune pipeline | ✅ Full | Dry-run without key | No trained model served |
| Multi-provider LLM | ✅ Full | Some untested without keys | — |
| Premium rating | Interface ✅ | Magic constants | Actuarial rates are proprietary |
