# Design: LLM-as-Judge Calibrated Against Human Labels

- **Date:** 2026-06-25
- **Status:** Approved
- **Owner:** Engineering

## Problem

The shipped rationale gatekeeper — `CriticAgent` in [workflows/critic.py](../../../workflows/critic.py) —
is an LLM judge that decides whether a producer rationale is faithful to the
retrieved underwriting evidence. It has two unmeasured weaknesses:

1. **It fails open.** When no critic LLM is configured, or any error occurs, it
   returns `passed=True` (`critic.py` lines 142–144 and 188–191). We never
   quantified how often a real judge *misses* an unfaithful rationale.
2. **The eval suite never judges a stochastic surface.** `evals/run.py`'s
   `faithfulness` metric is a deterministic citation/fact-grounding check, and
   `decision_accuracy` is governed by deterministic rules — both are
   reproducible-by-construction and therefore not evidence that the *AI* layer
   is trustworthy.

This feature adds a **calibration harness**: it pits the judge against a
hand-labeled set of rationales and reports judge↔human agreement plus the
judge's failure modes — most importantly the **false-negative rate** (unfaithful
rationales the judge passed), which is the fail-open risk quantified.

## Goals

- Measure judge↔human agreement on a balanced, hand-labeled set.
- Surface the judge's **failure modes**, especially false negatives.
- Keep the production critic prompt the single source of truth so the
  calibration characterizes the gatekeeper that actually ships.
- Stay reproducible and CI-safe (offline, no API key required).

## Non-Goals

- Putting the judge into the eligibility decision path (see ADR 0001 — it stays
  out).
- Replacing the deterministic faithfulness metric; this complements it.
- A hard CI gate on the agreement number (it is non-deterministic with a live
  model). A soft threshold flag is provided instead.

## Components

### 1. Hand-labeled calibration set — `evals/datasets/judge_calibration.jsonl`

~24 self-contained records (rationale + evidence + human gold label), so no live
workflow run is needed to evaluate. Class-balanced (~half faithful, ~half
unfaithful). Unfaithful records carry **injected defects** by category:

- `unsupported_claim` — a claim with no backing excerpt
- `fabricated_citation` — a `citation_chunk_id` not present in the evidence
- `overstated_severity` — evidence supports a milder finding than the rationale states
- `wrong_conclusion` — cites a real excerpt but draws the opposite conclusion

Record schema:

```json
{
  "id": "JCAL-001",
  "category": "faithful | unsupported_claim | fabricated_citation | overstated_severity | wrong_conclusion",
  "rationale": {"summary": "...", "supporting_facts": ["..."], "citation_chunk_ids": ["..."]},
  "evidence": [{"chunk_id": "uw_guidelines_...", "text": "..."}],
  "human_label": "faithful | unfaithful",
  "human_note": "one line: why a human judged it this way"
}
```

### 2. Judge wrapper — `evals/judge_calibration.py`

A thin `JudgeAgent` that scores each record's rationale against its evidence.
Two backends:

- **`llm`** — reuses the *exact* `_CRITIC_SYSTEM_PROMPT`, `_CRITIC_USER_TEMPLATE`,
  and `_CriticResponseModel` imported from `workflows/critic.py`, plus the same
  deterministic citation pre-check. This is the shipped gatekeeper. Requires an
  API key.
- **`simulated`** — a deterministic stand-in used when no API key is available
  (this environment, and CI). It runs the real citation pre-check, then a
  lexical-overlap faithfulness heuristic over `supporting_facts` vs evidence
  text. It is a genuine (weak) judge: it reliably catches fabricated citations
  and clearly unsupported claims, but **fails open on semantic defects**
  (overstated severity, flipped conclusions) where the words still overlap.
  All numbers it produces are real measurements of this stand-in.

The two backends share the citation pre-check and the verdict shape, so the
harness, metrics, and report are identical regardless of backend.

### 3. Calibration metrics

Positive class = "unfaithful / should be caught". Computed:

- N and class balance
- Agreement (accuracy)
- Confusion matrix (TP / FP / TN / FN)
- Precision, Recall (catch rate), F1 on the unfaithful class
- **False-negative rate** = FN / (TP + FN) — the fail-open headline
- **Cohen's kappa** — agreement beyond chance (defends against circularity)
- Enumerated disagreement table: each miss with the human note

### 4. Snapshot + report

- `--record` (live `llm` backend, needs key): writes per-record verdicts to
  `evals/datasets/judge_calibration_snapshot.json` (with backend, provider,
  model, timestamp) and the report to `evals/reports/judge_calibration.md`.
- Default (offline): recompute the report from the committed snapshot —
  deterministic, no key.
- `--min-agreement` / `--min-kappa` soft-threshold flags return non-zero on
  breach for opt-in use; **not** wired into the blocking eval CI gate.

### 5. Test — `tests/product/test_judge_calibration.py`

- Fixtures load, are well-formed, and class-balanced.
- Metric math (kappa, FNR, confusion matrix, P/R/F1) verified against a
  deterministic fake judge with known outputs — no key, no snapshot dependency.

## Data Flow

```
judge_calibration.jsonl
  -> JudgeAgent (shared critic prompt / pre-check; llm or simulated backend)
  -> per-record verdict {passed, unsupported_facts}
  -> compare to human_label
  -> metrics (agreement, P/R/F1, FNR, kappa, disagreements)
  -> judge_calibration_snapshot.json + evals/reports/judge_calibration.md
```

## Provenance / Honesty

This environment has no live API key, so the committed snapshot and report are
produced by the **simulated** backend, not a live LLM. This is stated plainly in
[docs/judge_calibration_provenance.md](../../judge_calibration_provenance.md),
which documents the exact simulation model and the one command
(`python -m evals.judge_calibration --record`) that recalibrates against the
real LLM critic. No agreement number is presented as if it came from a live
model when it did not.

## Documentation

- `docs/judge_calibration.md` — methodology and how to reproduce.
- `docs/judge_calibration_provenance.md` — how the committed numbers were
  generated (simulated backend) and how to regenerate with a real model.
- README — one bullet under the eval/trust section.
