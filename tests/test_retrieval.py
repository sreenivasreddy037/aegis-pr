"""Policy/runbook retrieval needs no manual indexing step."""

import shutil
from pathlib import Path

from app.config import FIXTURE_DIFF, FIXTURE_REPO
from app.graph.review_graph import run_review
from app.retrieval.policy_store import PolicyStore, discover_repo_docs

RUNBOOK = """# Payments runbook

## ADR-010: Transaction queries go through the query builder

Every SQL statement touching the transactions table, including status filters and LIMIT
clauses, must be built with bound parameters. Raw f-string SQL is an injection incident.
"""


def _repo_with_runbook(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    shutil.copytree(FIXTURE_REPO, repo)
    (repo / "docs" / "runbooks").mkdir(parents=True)
    (repo / "docs" / "runbooks" / "payments.md").write_text(RUNBOOK, encoding="utf-8")
    return repo


def test_repo_docs_are_discovered_automatically(tmp_path):
    repo = _repo_with_runbook(tmp_path)
    sources = {d["source"] for d in discover_repo_docs(repo)}
    assert {"ARCHITECTURE.md", "docs/runbooks/payments.md"} <= sources


def test_new_runbook_is_used_by_the_next_review_without_reindexing(tmp_path):
    repo = _repo_with_runbook(tmp_path)
    report = run_review(repo, FIXTURE_DIFF.read_text(encoding="utf-8"))
    used = {p.source for hits in report.impact.policies.values() for p in hits}
    assert "docs/runbooks/payments.md" in used


def test_hybrid_search_ranks_the_relevant_adr_first():
    store = PolicyStore()
    try:
        store.index_repo(FIXTURE_REPO)
        top = store.search("f-string SQL query status filter execute", top_k=2)
    finally:
        store.close()
    assert any("SQL" in hit.title for hit in top)
