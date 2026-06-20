"""Unit tests for the local inference benchmark and temperature study.

These exercise the metric math and report rendering without requiring a running
Ollama server, so they stay hermetic in CI. The actual measurement paths
(streaming, /api/ps) are covered by manual runs documented in
docs/inference_benchmarks.md.
"""

from scripts.benchmark_inference import (
    ModelBenchmark,
    RunSample,
    _is_json,
    render_markdown,
)


def test_is_json_detects_objects_and_rejects_prose():
    assert _is_json('{"roof_age_years": 12}')
    assert _is_json('Here is the answer: {"eligible": false} done')  # embedded object
    assert not _is_json("The roof age is twelve years.")
    assert not _is_json('{"unterminated": ')


def test_summary_computes_percentiles_and_rates():
    bench = ModelBenchmark(model="llama3.2:3b", temperature=0.0, memory_mb=2430.4)
    bench.samples = [
        RunSample(ttft_ms=100, total_ms=500, tokens_per_sec=40, prompt_tokens=10, output_tokens=20, json_valid=True),
        RunSample(ttft_ms=200, total_ms=700, tokens_per_sec=30, prompt_tokens=10, output_tokens=25, json_valid=True),
        RunSample(ttft_ms=300, total_ms=900, tokens_per_sec=50, prompt_tokens=10, output_tokens=30, json_valid=False),
    ]
    s = bench.summary()
    assert s["runs"] == 3
    assert s["errors"] == 0
    assert s["ttft_p50_ms"] == 200  # median of 100/200/300
    assert s["tokens_per_sec"] == 40.0  # mean of 40/30/50
    assert s["json_valid_pct"] == round(100 * 2 / 3, 1)
    assert s["memory_mb"] == 2430.4


def test_summary_handles_all_errored_runs():
    bench = ModelBenchmark(model="missing-model", temperature=0.0)
    bench.samples = [RunSample(0, 0, 0, 0, 0, False, error="not found")]
    s = bench.summary()
    assert s["runs"] == 0
    assert s["errors"] == 1


def test_render_markdown_includes_model_row_and_error_row():
    ok = ModelBenchmark(model="llama3.2:3b", temperature=0.0, memory_mb=2430.4)
    ok.samples = [RunSample(ttft_ms=120, total_ms=600, tokens_per_sec=42, prompt_tokens=8, output_tokens=20, json_valid=True)]
    errored = ModelBenchmark(model="not-pulled", temperature=0.0)
    errored.samples = [RunSample(0, 0, 0, 0, 0, False, error="model not found")]

    md = render_markdown([ok, errored])
    assert "llama3.2:3b" in md
    assert "not-pulled" in md
    assert "tok/s" in md
