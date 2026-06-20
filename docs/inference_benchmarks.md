# Local inference benchmarks (Tier 2.3–2.5)

Local small-language-model inference measured on real hardware via Ollama. All
numbers below are from `llama3.2:3b` on this machine; reproduce with the scripts
noted under each section. The reports are regenerated to `evals/reports/`.

## 2.3 — Throughput, latency, and memory

`python scripts/benchmark_inference.py --models llama3.2:3b --runs 2`

| Model | Temp | Runs | TTFT p50 (ms) | TTFT p95 (ms) | Total p50 (ms) | tok/s | JSON valid % | Mem (MB) |
|-------|------|------|---------------|---------------|----------------|-------|--------------|----------|
| llama3.2:3b | 0.0 | 16 | 254.9 | 376.9 | 736.8 | 40.3 | 87.5% | 2430.4 |

**How these are measured (not estimated):**
- **TTFT** — wall time from request to the first streamed token, over Ollama's
  `/api/generate` streaming response.
- **tok/s** — server-side `eval_count / eval_duration` reported by Ollama, i.e.
  true decode throughput, not characters ÷ 4.
- **Memory** — the loaded-model footprint from Ollama `/api/ps` (2430 MB here),
  which matches the `llama-server` runner RSS; falls back to a psutil scan of the
  runner process when `/api/ps` is unavailable.
- **JSON valid %** — minimal quality proxy: does the output parse as a JSON object.

**Reading it:** a 3B model gives a ~255 ms TTFT and ~40 tok/s on this CPU/GPU —
fast enough for interactive use, and the ~2.4 GB footprint is the real cost of
keeping it resident. These are exactly the practical trade-offs that justify (or
rule out) local inference for a given latency/cost/privacy requirement.

## 2.4 — Temperature variance

`python scripts/temperature_study.py --runs 5`

| Temperature | Consistency | Distinct rate | JSON valid % |
|-------------|-------------|---------------|--------------|
| 0.0 | 1.00 | 0.20 | 87.5% |
| 0.7 | 0.55 | 0.625 | 85.0% |

- **consistency** = fraction of runs matching the most common output for a prompt
  (1.0 = perfectly repeatable).
- **distinct rate** = unique outputs / runs (0.2 = 1/5 → identical every time).

**Reading it:** at temperature 0 the model is effectively deterministic
(consistency 1.0, every one of 5 runs identical). At 0.7 nearly half the runs
diverge (consistency 0.55, distinct rate 0.625). This is why the governed
workflow pins temperature 0 — the same submission must yield the same rationale
for the decision to be reproducible and auditable. Higher temperature buys
diversity you do not want on a regulated decision path.

## 2.5 — Quantization study (how to run)

The benchmark harness takes multiple model tags, so a quantization comparison is
the same script over several quant levels. This machine only has the default
`llama3.2:3b`, so pull the variants first (each ~2 GB):

```bash
ollama pull llama3.2:3b-instruct-q4_K_M
ollama pull llama3.2:3b-instruct-q5_K_M
ollama pull llama3.2:3b-instruct-q8_0

python scripts/benchmark_inference.py \
  --models llama3.2:3b-instruct-q4_K_M \
           llama3.2:3b-instruct-q5_K_M \
           llama3.2:3b-instruct-q8_0 \
  --runs 3
```

The output table puts tok/s, TTFT, memory, and JSON-valid % side by side per
quant level — the quality-vs-speed-vs-memory trade-off. Expectation to verify on
your hardware: lower-bit quants (q4) use less memory and decode faster, higher-bit
quants (q8) cost more memory and time for marginally better output quality.

## Notes

- These scripts target Ollama specifically because its streaming API exposes
  authoritative server-side token counts and durations. They no-op with a clear
  message when Ollama is unreachable, so they never block CI.
- Cross-provider *quality/cost* comparison (OpenAI/Claude/Gemini/Nebius/Ollama on
  the rationale task) lives in `scripts/compare_models.py`; this file is about
  local-inference latency/throughput/memory.
