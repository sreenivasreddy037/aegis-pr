"""End-to-end LangGraph review, critic behaviour, parallel fan-out, API and CLI."""

import json
import time

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from app.api import app as api_app
from app.cli import app as cli_app
from app.config import FIXTURE_DIFF, FIXTURE_REPO, get_settings
from app.evaluation import load_benchmark
from app.graph.llm import (
    CandidateAssessment,
    LLMReview,
    MockReviewLLM,
    NewFinding,
    merge_llm_review,
)
from app.graph.review_graph import AGENT_NODES, build_review_graph, run_review
from app.schemas import Category, Severity


def _benchmark_diff(pr_id: str) -> str:
    return next(pr["diff"] for pr in load_benchmark()["prs"] if pr["id"] == pr_id)


CLEAN_DIFF = _benchmark_diff("PR-05")  # docstring + named constant refactor


@pytest.fixture(scope="module")
def demo_report():
    return run_review(FIXTURE_REPO, FIXTURE_DIFF.read_text(encoding="utf-8"))


def test_demo_catches_all_three_seeded_bugs(demo_report):
    r = demo_report
    assert r.mode == "mock" and r.verdict == "request_changes" and not r.errors
    by_cat = {f.category: f for f in r.findings}
    assert len(r.findings) == 3

    breaking = by_cat[Category.BREAKING_CHANGE]
    assert (breaking.file_path, breaking.line, breaking.symbol) == (
        "api_routes.py",
        24,
        "api_routes.get_current_user",
    )
    assert breaking.verified and "audience" in breaking.description

    sqli = by_cat[Category.SECURITY]
    assert (sqli.file_path, sqli.cwe, sqli.severity) == ("db.py", "CWE-89", Severity.CRITICAL)
    assert any("ADR-001" in e.detail for e in sqli.evidence)

    regression = by_cat[Category.REGRESSION]
    assert regression.file_path == "payments.py" and regression.verified
    assert any(e.kind == "sandbox" and "ZeroDivisionError" in e.detail for e in regression.evidence)
    assert "security_auditor" in regression.corroborated_by  # static finding merged in

    assert [f.id for f in r.findings] == ["AEG-001", "AEG-002", "AEG-003"]
    assert len(r.sandbox_runs) == 1 and r.sandbox_runs[0].regressions


def test_critic_dedupes_and_filters_low_confidence(demo_report):
    reasons = {d.finding.cwe: d.reason for d in demo_report.dropped}
    assert reasons["CWE-369"].startswith("merged into regression_tester")
    assert "confidence 0.35 < 0.60" in reasons["CWE-532"]


def test_every_node_is_timed(demo_report):
    assert set(demo_report.timings_ms) == {"analyze_impact", *AGENT_NODES, "critic_synthesizer"}
    assert demo_report.total_latency_ms > 0


def test_clean_pr_is_approved():
    r = run_review(FIXTURE_REPO, CLEAN_DIFF)
    assert r.verdict == "approve" and r.findings == []


def test_graph_topology_has_parallel_fan_out_and_fan_in():
    graph = build_review_graph(MockReviewLLM(), get_settings()).get_graph()
    nodes = set(graph.nodes)
    assert {"analyze_impact", *AGENT_NODES, "critic_synthesizer"} <= nodes
    edges = {(e.source, e.target) for e in graph.edges}
    for agent in AGENT_NODES:
        assert ("analyze_impact", agent) in edges
        assert (agent, "critic_synthesizer") in edges


class SlowLLM(MockReviewLLM):
    """Mock LLM with simulated network latency, to prove the agents run concurrently."""

    DELAY = 0.6

    def review(self, agent, system_prompt, context, candidates):
        time.sleep(self.DELAY)
        return super().review(agent, system_prompt, context, candidates)


def test_agents_run_in_parallel():
    r = run_review(FIXTURE_REPO, FIXTURE_DIFF.read_text(encoding="utf-8"), llm=SlowLLM())
    agent_ms = [r.timings_ms[a] for a in AGENT_NODES]
    assert min(agent_ms) >= SlowLLM.DELAY * 1000
    # Sequential execution would take at least the sum of the agent times.
    assert r.total_latency_ms < sum(agent_ms) - SlowLLM.DELAY * 1000


class HallucinatingLLM(MockReviewLLM):
    mode = "mock"

    def review(self, agent, system_prompt, context, candidates):
        result = LLMReview(
            assessments=[
                CandidateAssessment(candidate_id=f"C{i + 1}", keep=c.verified, rationale="checked")
                for i, c in enumerate(candidates)
            ],
            additional_findings=[
                NewFinding(
                    title="Imaginary race condition",
                    description="Made up.",
                    file_path="does_not_exist.py",
                    line=10,
                    category=Category.SECURITY,
                    severity=Severity.HIGH,
                    confidence=0.9,
                )
            ]
            if agent == "security_auditor"
            else [],
        )
        return merge_llm_review(agent, [c.model_copy(deep=True) for c in candidates], result)


def test_critic_rejects_hallucinated_locations_and_llm_cannot_drop_verified():
    r = run_review(FIXTURE_REPO, FIXTURE_DIFF.read_text(encoding="utf-8"), llm=HallucinatingLLM())
    assert any(
        "hallucinated location" in d.reason and d.finding.file_path == "does_not_exist.py"
        for d in r.dropped
    )
    # The LLM rejected every unverified candidate (incl. the SQL injection) but kept verified ones.
    cats = {f.category for f in r.findings}
    assert cats == {Category.BREAKING_CHANGE, Category.REGRESSION}
    assert all(f.verified for f in r.findings)


# ----------------------------------------------------------------------------- API


@pytest.fixture(scope="module")
def client():
    return TestClient(api_app)


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["mode"] == "mock"


def test_review_demo_endpoint(client):
    resp = client.get("/review/demo")
    assert resp.status_code == 200
    body = resp.json()
    assert body["verdict"] == "request_changes"
    assert {f["category"] for f in body["findings"]} == {
        "breaking_change",
        "security",
        "regression",
    }


def test_review_diff_endpoint(client):
    resp = client.post(
        "/review/diff",
        json={"diff": FIXTURE_DIFF.read_text(encoding="utf-8"), "repo_path": "sample_repo"},
    )
    assert resp.status_code == 200 and len(resp.json()["findings"]) == 3


def test_review_diff_rejects_bad_input(client):
    assert (
        client.post(
            "/review/diff", json={"diff": "--- a/db.py\n+++ b/db.py\n@@ -1,1 +1,1 @@\n-nope\n+x\n"}
        ).status_code
        == 422
    )
    assert (
        client.post("/review/diff", json={"diff": CLEAN_DIFF, "repo_path": "../../"}).status_code
        == 400
    )
    assert client.post("/review/diff", json={"diff": ""}).status_code == 422


# ----------------------------------------------------------------------------- CLI


def test_cli_demo_json_and_fail_on():
    runner = CliRunner()
    result = runner.invoke(cli_app, ["review", "--demo", "--json"])
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    assert len(report["findings"]) == 3

    gated = runner.invoke(cli_app, ["review", "--demo", "--json", "--fail-on", "critical"])
    assert gated.exit_code == 1


def test_cli_rich_rendering():
    result = CliRunner().invoke(cli_app, ["review", "--demo", "--no-diff"])
    assert result.exit_code == 0, result.output
    for needle in (
        "REQUEST CHANGES",
        "AST impact graph",
        "AEG-001",
        "Filtered by critic",
        "end-to-end",
    ):
        assert needle in result.output


# ----------------------------------------------------------------------------- Web UI


def test_web_ui_is_served(client):
    page = client.get("/")
    assert page.status_code == 200 and "text/html" in page.headers["content-type"]
    for needle in (
        'id="review-form"',
        'id="graph"',
        'src="static/app.js"',
        'href="static/app.css"',
    ):
        assert needle in page.text
    js = client.get("/static/app.js")
    assert js.status_code == 200 and "renderGraph" in js.text
    assert client.get("/static/app.css").status_code == 200


def test_web_ui_never_uses_inner_html(client):
    # Report text comes from analysed (untrusted) code; the UI must only insert it as text.
    js = client.get("/static/app.js").text
    for sink in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write"):
        assert sink not in js


def test_repos_and_sample_endpoints(client):
    assert client.get("/repos").json() == ["sample_repo"]
    sample = client.get("/sample").json()
    assert sample["repo_path"] == "sample_repo"
    assert sample["diff"] == FIXTURE_DIFF.read_text(encoding="utf-8")
