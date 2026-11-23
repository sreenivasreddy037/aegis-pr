"""Benchmark evaluation: finding recall/precision on 12 curated PRs + DeepEval faithfulness.

Offline (`MOCK_LLM=true`) this measures the deterministic pipeline and acts as a regression
gate. With `OPENAI_API_KEY` set, the DeepEval Faithfulness metric additionally checks that
finding explanations are grounded in the evidence the agents retrieved.
"""

import json
import os

import pytest

from app.config import PROJECT_ROOT, Settings
from app.evaluation import BenchmarkResult, run_benchmark

RESULTS_DIR = PROJECT_ROOT / "evals" / "results"
MIN_RECALL = 0.90
MIN_PRECISION = 0.85
MIN_FAITHFULNESS = 0.85


@pytest.fixture(scope="module")
def benchmark() -> BenchmarkResult:
    result = run_benchmark()
    RESULTS_DIR.mkdir(exist_ok=True)
    payload = {
        "summary": result.summary(),
        "cases": [
            {
                "id": c.pr_id,
                "tp": c.tp,
                "fp": c.fp,
                "fn": c.fn,
                "latency_ms": c.latency_ms,
                "missed": c.unmatched_expected,
                "false_positives": c.false_positives,
            }
            for c in result.cases
        ],
    }
    (RESULTS_DIR / "latest.json").write_text(json.dumps(payload, indent=2))
    return result


def test_benchmark_has_twelve_prs(benchmark):
    assert len(benchmark.cases) == 12
    assert sum(1 for c in benchmark.cases if not c.expected) >= 3  # clean PRs measure false alarms


def test_recall_and_precision(benchmark):
    s = benchmark.summary()
    missed = {c.pr_id: c.unmatched_expected for c in benchmark.cases if c.fn}
    fps = {c.pr_id: c.false_positives for c in benchmark.cases if c.fp}
    assert s["recall"] >= MIN_RECALL, f"missed: {missed}"
    assert s["precision"] >= MIN_PRECISION, f"false positives: {fps}"
    assert s["clean_pr_false_alarms"] == 0


def test_latency_budget(benchmark):
    assert benchmark.summary()["p95_latency_ms"] < 15_000


def test_ablations_prove_each_component_matters(benchmark):
    full = benchmark.summary()["recall"]
    for ablation in ("no_impact_graph", "no_sandbox"):
        reduced = run_benchmark(Settings(mock_llm=True, ablation=ablation)).summary()["recall"]
        assert reduced < full, f"{ablation} should lose recall"


@pytest.mark.llm
@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set; DeepEval judge needs an LLM",
)
def test_faithfulness_of_findings(benchmark):
    from deepeval.metrics import FaithfulnessMetric
    from deepeval.test_case import LLMTestCase

    scores = []
    for case in benchmark.cases:
        for f in case.report.findings:
            metric = FaithfulnessMetric(
                threshold=MIN_FAITHFULNESS, model="gpt-4o-mini", include_reason=True
            )
            metric.measure(
                LLMTestCase(
                    input=f"Review finding for {case.pr_id} at {f.location}",
                    actual_output=f.description,
                    retrieval_context=[e.detail for e in f.evidence],
                )
            )
            scores.append(metric.score or 0.0)
    assert scores
    assert sum(scores) / len(scores) >= MIN_FAITHFULNESS
