# LLM-as-Judge Calibration

The shipped rationale gatekeeper — `CriticAgent` in
[workflows/critic.py](../workflows/critic.py) — is an LLM judge: it decides
whether a producer rationale is *faithful* to the retrieved underwriting
evidence before the rationale is released. Like every LLM judge, it can be wrong,
and it **fails open** (no critic configured, or any error → `passed=True`).

This harness measures how well that judge agrees with a human, and — more
usefully — *where it disagrees*. It is the one eval in this repo that scores a
genuinely stochastic surface (LLM rationale wording), as opposed to the
deterministic `decision_accuracy` / `faithfulness` checks in
[evals/run.py](../evals/run.py) which are reproducible by construction.

## Why this matters

- **It is not circular.** Decision accuracy is governed by deterministic rules,
  so "100%" is true by design and proves nothing about the AI layer. Judge↔human
  agreement on hand-labeled rationales evaluates something that can actually be
  wrong, so the number survives interrogation. Cohen's kappa discounts
  chance agreement on top of that.
- **It quantifies the fail-open risk.** The headline metric is the
  **false-negative rate** — the share of genuinely unfaithful rationales the
  judge passed. Those are hallucinated justifications that would otherwise ship.
- **It is actionable.** The by-category table shows *which* defect types the
  judge misses, which points directly at the prompt/rule changes that close the
  gap.

## The calibration set

[evals/datasets/judge_calibration.jsonl](../evals/datasets/judge_calibration.jsonl)
holds ~24 self-contained, human-labeled records (each is a rationale + the
evidence excerpts it should rest on + a gold label + a one-line human note). It
is class-balanced (≈half faithful). Unfaithful records carry **injected
defects** by category:

| Category | Defect |
| --- | --- |
| `unsupported_claim` | A claim with no backing excerpt (e.g. an invented surcharge). |
| `fabricated_citation` | A `citation_chunk_id` not present in the evidence. |
| `overstated_severity` | Evidence supports a milder action than the rationale states. |
| `wrong_conclusion` | Cites a real excerpt but draws the opposite conclusion. |

## Running it

```bash
# Deterministic, no API key (default): scores the simulated stand-in judge
python -m evals.judge_calibration

# Record the snapshot + markdown report
python -m evals.judge_calibration --record

# Recompute the report offline from the committed snapshot (no judge calls)
python -m evals.judge_calibration --backend snapshot

# Calibrate the REAL shipped LLM judge (needs an API key) and record it.
# The judge defaults to Claude (claude-sonnet-4-6), independent of the
# generator, so it does not grade output from the same model.
LLM_STRUCTURED_OUTPUT_ENABLED=true ANTHROPIC_API_KEY=... \
    python -m evals.judge_calibration --backend llm --record

# Use a different judge model (e.g. maximum rigor with Opus, or a budget run):
CRITIC_LLM_MODEL=claude-opus-4-8 LLM_STRUCTURED_OUTPUT_ENABLED=true \
    ANTHROPIC_API_KEY=... python -m evals.judge_calibration --backend llm --record

# Opt-in soft thresholds (return non-zero on breach; not in the blocking CI gate)
python -m evals.judge_calibration --min-agreement 0.8 --min-kappa 0.6
```

The report is written to
[evals/reports/judge_calibration.md](../evals/reports/judge_calibration.md).

## Backends

- **`llm`** — the real shipped judge. Reuses the *exact* critic system prompt,
  user template, and JSON schema imported from `workflows/critic.py`, plus the
  same deterministic citation pre-check, and the same default model. This
  guarantees the calibration characterizes the gatekeeper that actually runs,
  not a parallel prompt. Defaults to **Claude `claude-sonnet-4-6`**
  (`CRITIC_LLM_PROVIDER` / `CRITIC_LLM_MODEL` to override), chosen independently
  of the generator's provider to avoid self-grading bias. Requires an API key.
- **`simulated`** — a deterministic stand-in used when no key is available (CI,
  and the environment this was first generated in). See
  [judge_calibration_provenance.md](judge_calibration_provenance.md) for exactly
  how it works and how the committed numbers were produced.
- **`snapshot`** — recompute the report from a recorded snapshot with no judge
  calls (fully offline).

## Metrics

Positive class = "unfaithful" (the judge should catch it).

- **Agreement** — raw accuracy of judge vs human.
- **Cohen's kappa** — agreement beyond chance.
- **Precision / Recall / F1** on the unfaithful class.
- **False-negative rate** — `FN / (TP + FN)`, the fail-open headline.
- **False-positive rate** — `FP / (FP + TN)`, the false-alarm rate.
- **By-category catch rate** and an enumerated **disagreement table** (each miss
  with its human note).

## CI posture

The agreement number is **not** wired into the blocking eval gate, because a
live LLM judge is non-deterministic. The `--min-agreement` / `--min-kappa`
flags exist for opt-in use. What CI *does* enforce
([tests/product/test_judge_calibration.py](../tests/product/test_judge_calibration.py)):
the fixtures are well-formed and balanced, the metric math is correct, and the
committed snapshot matches the current simulated judge.
