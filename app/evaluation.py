"""Benchmark scoring shared by `aegispr benchmark` and `evals/test_review_eval.py`.

A finding matches a ground-truth item when category and file agree and, if the ground truth
names a symbol, the finding's enclosing symbol ends with it. Each ground-truth item can be
matched at most once; unmatched findings are false positives.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from app.config import PROJECT_ROOT, Settings, get_settings
from app.schemas import PRReviewReport, ReviewFinding

BENCHMARK_PATH = PROJECT_ROOT / "evals" / "benchmark_prs.json"


@dataclass
class CaseResult:
    pr_id: str
    title: str
    expected: list[dict]
    found: int
    tp: int
    fp: int
    fn: int
    latency_ms: float
    report: PRReviewReport = field(repr=False)
    unmatched_expected: list[dict] = field(default_factory=list)
    false_positives: list[str] = field(default_factory=list)


@dataclass
class BenchmarkResult:
    mode: str
    cases: list[CaseResult]

    def summary(self) -> dict:
        tp = sum(c.tp for c in self.cases)
        fp = sum(c.fp for c in self.cases)
        fn = sum(c.fn for c in self.cases)
        latencies = sorted(c.latency_ms for c in self.cases)
        clean = [c for c in self.cases if not c.expected]
        return {
            "mode": self.mode,
            "prs": len(self.cases),
            "ground_truth": tp + fn,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "recall": tp / (tp + fn) if tp + fn else 1.0,
            "precision": tp / (tp + fp) if tp + fp else 1.0,
            "false_positive_rate": fp / (tp + fp) if tp + fp else 0.0,
            "clean_prs": len(clean),
            "clean_pr_false_alarms": sum(1 for c in clean if c.found),
            "p50_latency_ms": percentile(latencies, 50),
            "p95_latency_ms": percentile(latencies, 95),
        }


def percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    k = (len(sorted_values) - 1) * pct / 100
    lo, hi = math.floor(k), math.ceil(k)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (k - lo)


def matches(finding: ReviewFinding, expected: dict) -> bool:
    if finding.category.value != expected["category"] or finding.file_path != expected["file"]:
        return False
    symbol = expected.get("symbol")
    return symbol is None or (finding.symbol or "").endswith(symbol)


def score(
    findings: list[ReviewFinding], expected: list[dict]
) -> tuple[int, int, list[dict], list[str]]:
    remaining = list(expected)
    tp, false_positives = 0, []
    for f in findings:
        hit = next((e for e in remaining if matches(f, e)), None)
        if hit is not None:
            remaining.remove(hit)
            tp += 1
        else:
            false_positives.append(f"{f.category.value} {f.location} {f.title}")
    return tp, len(false_positives), remaining, false_positives


def load_benchmark(path: Path = BENCHMARK_PATH) -> dict:
    return json.loads(path.read_text())


def run_benchmark(settings: Settings | None = None, path: Path = BENCHMARK_PATH) -> BenchmarkResult:
    from app.graph.review_graph import run_review

    settings = settings or get_settings()
    data = load_benchmark(path)
    repo = PROJECT_ROOT / data["repo"]
    cases = []
    for pr in data["prs"]:
        report = run_review(repo, pr["diff"], settings=settings)
        tp, fp, missing, fps = score(report.findings, pr["expected"])
        cases.append(
            CaseResult(
                pr_id=pr["id"],
                title=pr["title"],
                expected=pr["expected"],
                found=len(report.findings),
                tp=tp,
                fp=fp,
                fn=len(missing),
                latency_ms=report.total_latency_ms,
                report=report,
                unmatched_expected=missing,
                false_positives=fps,
            )
        )
    return BenchmarkResult(mode=settings.mode, cases=cases)
