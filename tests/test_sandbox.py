"""Sandboxed pytest execution: isolation, timeouts and differential edge-case testing."""

import time

from app.analysis.edge_cases import synthesize
from app.ast_engine.impact_graph import compute_impact
from app.config import FIXTURE_DIFF, FIXTURE_REPO
from app.sandbox.test_runner import run_pytest
from app.workspace import prepare_workspace


def test_passing_and_failing_cases_are_captured():
    code = (
        "from payments import calculate_split_fee\n\n"
        "def test_ok():\n    assert calculate_split_fee(1000, 2) == 15\n\n"
        "def test_bad():\n    calculate_split_fee(1000, 0)\n"
    )
    result = run_pytest(code, FIXTURE_REPO)
    assert result.exit_code == 1 and not result.timed_out
    status = {c.name: (c.status, c.message) for c in result.cases}
    assert status["test_ok"] == ("passed", "")
    assert status["test_bad"][0] == "failed"
    assert "ValueError: participants must be a positive integer" in status["test_bad"][1]


def test_timeout_is_enforced_and_process_group_killed():
    start = time.perf_counter()
    result = run_pytest("import time\n\ndef test_hang():\n    time.sleep(60)\n", timeout_sec=2)
    elapsed = time.perf_counter() - start
    assert result.timed_out and result.exit_code is None
    assert elapsed < 8


def test_network_is_blocked():
    code = "import socket\n\ndef test_net():\n    socket.create_connection(('203.0.113.1', 80), timeout=1)\n"
    result = run_pytest(code)
    assert result.cases[0].status == "failed"
    assert "network access is disabled" in result.cases[0].message


def test_environment_is_scrubbed(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-should-not-leak")
    code = "import os\n\ndef test_env():\n    assert 'OPENAI_API_KEY' not in os.environ\n    assert os.environ['AEGIS_SANDBOX'] == '1'\n"
    assert run_pytest(code).passed


def test_tests_run_on_a_private_copy(tmp_path):
    project = tmp_path / "proj"
    project.mkdir()
    (project / "mod.py").write_text("VALUE = 1\n")
    code = "from pathlib import Path\n\ndef test_mutate():\n    Path('mod.py').write_text('VALUE = 2\\n')\n"
    assert run_pytest(code, project).passed
    assert (project / "mod.py").read_text() == "VALUE = 1\n"


def test_differential_edge_cases_prove_regression():
    with prepare_workspace(FIXTURE_REPO, FIXTURE_DIFF.read_text()) as ws:
        impact, _, head = compute_impact(ws.base_dir, ws.head_dir, ws.files)
        plans = {p.target: p for p in synthesize(impact.modified_symbols, head)}
        plan = plans["payments.calculate_split_fee"]
        assert [c.case_id for c in plan.cases] == ["participants=0", "participants=<omitted>"]
        base = run_pytest(plan.test_code, ws.base_dir)
        head_result = run_pytest(plan.test_code, ws.head_dir)
    assert base.passed  # base validates input with ValueError, which is acceptable
    zero = head_result.case("[participants=0]")
    omitted = head_result.case("[participants=<omitted>]")
    assert zero is not None and "ZeroDivisionError" in zero.message
    assert omitted is not None and "NoneType" in omitted.message
