# Fine-tuning: HO3 intake extraction (Tier 3 / Project 4)

A supervised LoRA fine-tune for the structured-extraction task the intake
normalizer performs: **free-text producer note → canonical HO3 JSON**. Trained and
served through **Nebius Token Factory** (OpenAI-compatible), so the resulting model
is just another `LLM_MODEL` pointed at the Nebius base URL — it drops into the
existing provider boundary with no new runtime code.

## Why this task

Fine-tuning earns its keep when a well-defined task needs *consistent* structure
and even careful prompting falls short. Structured extraction with **abstention**
is exactly that: the model must normalize messy phrasing ("re-roofed 12 years
ago", "Coverage A of 450k") into canonical fields **and leave unstated fields
null instead of guessing**. That abstention is the `refusal_correctness` metric —
a model that invents a roof age the note never mentioned is worse than one that
returns null.

## Pipeline

```
finetune/
  schema.py            # extraction fields + system prompt + canonical values
  generate_dataset.py  # synth (note -> JSON) pairs, OpenAI chat JSONL, seeded
  metrics.py           # json validity, exact match, field accuracy, refusal correctness
  submit.py            # upload + create LoRA job + poll (Nebius Token Factory)
  evaluate.py          # base vs fine-tuned on the holdout
```

Generated data lands in the gitignored `finetune/data/` (regenerate any time).

## Run it

```bash
# 1. Generate the dataset (no spend). 3k train / 300 holdout, deterministic.
python -m finetune.generate_dataset --train 3000 --holdout 300

# 2. Validate the dataset + preview the job request (no spend).
python -m finetune.submit --dry-run

# 3. Launch the LoRA fine-tune (spends credit). Needs NEBIUS_API_KEY.
NEBIUS_API_KEY=... python -m finetune.submit \
    --base-model meta-llama/Llama-3.1-8B-Instruct --epochs 3
#    → prints the fine-tuned model id: ft:meta-llama/Llama-3.1-8B-Instruct-...

# 4. Before/after eval on the holdout (small inference spend).
NEBIUS_API_KEY=... python -m finetune.evaluate \
    --base-model meta-llama/Llama-3.1-8B-Instruct \
    --tuned-model 'ft:meta-llama/Llama-3.1-8B-Instruct-...'
#    → evals/reports/finetune_eval.md
```

## Metrics (the before/after table)

`evaluate.py` produces:

| metric | meaning |
|---|---|
| JSON valid | fraction of outputs that parse as a JSON object |
| exact match | fraction where **every** field equals gold |
| field accuracy | mean per-field correctness |
| refusal correctness | of fields absent in the note, fraction the model left null (abstention) |

**Hypothesis to confirm:** the base 8B model with a careful prompt already gets
high JSON validity but loses points on field accuracy (phrasing normalization)
and especially refusal correctness (it tends to hallucinate plausible values).
The fine-tune should close both gaps — that's the story the table tells.

## Cost

LoRA on an 8B model over ~3k short examples × 3 epochs is a small job; the
holdout eval is a few hundred short completions. This fits comfortably in the
$100 Token Factory credit, but **confirm current fine-tuning + inference pricing**
on the Nebius console before launching — pricing changes over time.

## Scope notes

- **SFT/LoRA only.** Token Factory's managed API is supervised-fine-tune focused.
  A DPO preference phase (the video's phase 2) would need raw GPU (Nebius AI
  Cloud) + a framework like TRL, and is intentionally out of scope here.
- **Base model.** `meta-llama/Llama-3.1-8B-Instruct` is the documented, supported
  base; swap via `--base-model` if Token Factory's model list offers a better fit.
- **Serving.** Point the app at the tuned model with
  `LLM_PROVIDER=nebius LLM_MODEL='ft:...'` — no code change, since the structured
  LLM service already targets Nebius's OpenAI-compatible endpoint.
