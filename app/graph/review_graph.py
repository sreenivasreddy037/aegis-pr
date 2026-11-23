"""LangGraph orchestration: deterministic impact analysis -> parallel agents -> critic.

    START -> analyze_impact --Send--> security_auditor  --\
                            --Send--> contract_reviewer --+--> critic_synthesizer -> END
                            --Send--> regression_tester --/

The three reviewers run in the same LangGraph superstep (concurrently); their findings are
merged with an `operator.add` reducer and the critic runs once all of them finish.
"""

from __future__ import annotations

import logging
import operator
import time
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from app.config import Settings, get_settings
from app.graph import agents
from app.graph.llm import ReviewLLM, get_llm
from app.schemas import ImpactAnalysis, PRReviewReport, ReviewFinding, SandboxRun
from app.workspace import ReviewWorkspace, prepare_workspace

logger = logging.getLogger(__name__)

AGENT_NODES = ("security_auditor", "contract_reviewer", "regression_tester")


def merge_dicts(a: dict[str, float], b: dict[str, float]) -> dict[str, float]:
    return {**a, **b}


class ReviewState(TypedDict, total=False):
    workspace: ReviewWorkspace
    impact: ImpactAnalysis
    findings: Annotated[list[ReviewFinding], operator.add]
    sandbox_runs: Annotated[list[SandboxRun], operator.add]
    timings_ms: Annotated[dict[str, float], merge_dicts]
    errors: Annotated[list[str], operator.add]
    report: PRReviewReport


def _timed(name: str, fn: Callable[[ReviewState], dict[str, Any]], *, fatal: bool = False):
    """Wrap a node: record wall time and turn agent crashes into reported errors."""

    def node(state: ReviewState) -> dict[str, Any]:
        start = time.perf_counter()
        try:
            update = fn(state)
        except Exception as exc:
            if fatal:
                raise
            logger.exception("node %s failed", name)
            update = {"errors": [f"{name}: {type(exc).__name__}: {exc}"]}
        update["timings_ms"] = {name: round((time.perf_counter() - start) * 1000, 1)}
        return update

    node.__name__ = name
    return node


def build_review_graph(llm: ReviewLLM, settings: Settings):
    def analyze_impact(state: ReviewState) -> dict[str, Any]:
        return {"impact": agents.analyze_impact(state["workspace"], settings)}

    def fan_out(state: ReviewState) -> list[Send]:
        return [Send(name, state) for name in AGENT_NODES]

    def security_auditor(state: ReviewState) -> dict[str, Any]:
        return {"findings": agents.security_auditor(state["workspace"], state["impact"], llm)}

    def contract_reviewer(state: ReviewState) -> dict[str, Any]:
        return {
            "findings": agents.contract_reviewer(state["workspace"], state["impact"], llm, settings)
        }

    def regression_tester(state: ReviewState) -> dict[str, Any]:
        findings, runs = agents.regression_tester(
            state["workspace"], state["impact"], llm, settings
        )
        return {"findings": findings, "sandbox_runs": runs}

    def critic_synthesizer(state: ReviewState) -> dict[str, Any]:
        impact = state["impact"]
        kept, dropped, summary = agents.critic_synthesizer(
            state["workspace"], impact, state.get("findings", []), llm, settings
        )
        report = PRReviewReport(
            verdict=agents.verdict_for(kept),  # type: ignore[arg-type]
            summary=summary,
            mode=llm.mode,
            findings=kept,
            dropped=dropped,
            impact=impact,
            sandbox_runs=state.get("sandbox_runs", []),
            errors=state.get("errors", []),
        )
        return {"report": report}

    g = StateGraph(ReviewState)
    g.add_node("analyze_impact", _timed("analyze_impact", analyze_impact, fatal=True))
    g.add_node("security_auditor", _timed("security_auditor", security_auditor))
    g.add_node("contract_reviewer", _timed("contract_reviewer", contract_reviewer))
    g.add_node("regression_tester", _timed("regression_tester", regression_tester))
    g.add_node("critic_synthesizer", _timed("critic_synthesizer", critic_synthesizer, fatal=True))
    g.add_edge(START, "analyze_impact")
    g.add_conditional_edges("analyze_impact", fan_out, list(AGENT_NODES))
    for name in AGENT_NODES:
        g.add_edge(name, "critic_synthesizer")
    g.add_edge("critic_synthesizer", END)
    return g.compile()


def run_review(
    repo: Path | str,
    diff_text: str,
    *,
    settings: Settings | None = None,
    llm: ReviewLLM | None = None,
) -> PRReviewReport:
    """Review `diff_text` against the repository at `repo` and return a structured report."""
    settings = settings or get_settings()
    llm = llm or get_llm(settings)
    start = time.perf_counter()
    with prepare_workspace(Path(repo), diff_text) as ws:
        graph = build_review_graph(llm, settings)
        final = graph.invoke(
            {"workspace": ws, "findings": [], "sandbox_runs": [], "timings_ms": {}, "errors": []}
        )
    report: PRReviewReport = final["report"]
    report.timings_ms = final.get("timings_ms", {})
    report.total_latency_ms = round((time.perf_counter() - start) * 1000, 1)
    return report
