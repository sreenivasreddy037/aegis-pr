"""Tree-sitter symbol extraction, diff handling and 2-hop impact resolution."""

from pathlib import Path

import pytest

from app.ast_engine.diff_utils import (
    PatchError,
    apply_diff_to_tree,
    apply_file_diff,
    parse_unified_diff,
)
from app.ast_engine.impact_graph import ImpactGraph, check_call, compare_signatures, compute_impact
from app.ast_engine.parser import parse_source
from app.config import FIXTURE_DIFF, FIXTURE_REPO
from app.schemas import Param
from app.workspace import prepare_workspace

SOURCE = """
import os
from auth import verify_token as vt


class TokenService:
    def __init__(self, secret: str) -> None:
        self.secret = secret

    def check(self, token, *, audience: str = "api"):
        return self._decode(token, audience=audience)

    def _decode(self, token, audience):
        return vt(token, self.secret)


@router.get("/me")
def me(user=Depends(get_user), *args, **kwargs):
    return TokenService(os.environ["S"]).check(user)
"""


def test_extracts_classes_methods_functions_with_signatures():
    mod = parse_source(SOURCE, "svc", "svc.py")
    by_name = {s.qualname: s for s in mod.symbols}
    assert by_name["TokenService"].kind == "class"
    assert by_name["TokenService.check"].kind == "method"
    assert by_name["me"].kind == "function"
    assert by_name["me"].decorators == ['router.get("/me")']
    assert by_name["TokenService.check"].signature == 'check(self, token, *, audience: str = "api")'
    kinds = [p.kind for p in by_name["me"].params]
    assert kinds == ["positional", "var_positional", "var_keyword"]
    assert by_name["TokenService"].start_line == 6


def test_extracts_call_shapes_and_imports():
    mod = parse_source(SOURCE, "svc", "svc.py")
    call = next(c for c in mod.calls if c.callee_expr == "self._decode")
    assert call.caller == "TokenService.check"
    assert (call.positional_args, call.keyword_args) == (1, ["audience"])
    depends = next(c for c in mod.calls if c.callee_expr == "Depends")
    assert depends.caller == "me" and depends.referenced_names == ["get_user"]
    imports = {(i.module, i.name, i.alias) for i in mod.imports}
    assert ("auth", "verify_token", "vt") in imports and ("os", None, "os") in imports


def test_parser_is_error_tolerant():
    mod = parse_source("def ok(a):\n    return a\n\ndef broken(:\n", "m", "m.py")
    assert mod.has_errors
    assert any(s.name == "ok" for s in mod.symbols)


def test_graph_resolves_calls_imports_and_di_references():
    g = ImpactGraph.from_repo(FIXTURE_REPO)
    assert g.graph.edges["api_routes.get_current_user", "auth.verify_token"]["kind"] == "calls"
    assert (
        g.graph.edges["api_routes.split_payment", "api_routes.get_current_user"]["kind"]
        == "references"
    )
    assert g.graph.edges["payments.settle_split", "db.record_transaction"]["kind"] == "calls"
    assert g.graph.edges["api_routes", "auth.verify_token"]["kind"] == "imports"
    callers = {node: hop for node, hop, _, _ in g.upstream("auth.verify_token", 2)}
    assert callers["api_routes.get_current_user"] == 1
    assert callers["api_routes.refresh_session"] == 1
    assert callers["api_routes.list_transactions"] == 2
    assert callers["api_routes.split_payment"] == 2


def test_two_hop_impact_and_contract_violation_for_sample_pr():
    with prepare_workspace(FIXTURE_REPO, FIXTURE_DIFF.read_text()) as ws:
        impact, _, _ = compute_impact(ws.base_dir, ws.head_dir, ws.files, max_hops=2)
    modified = {n.symbol for n in impact.modified_symbols}
    assert {
        "auth.verify_token",
        "db.get_user_transactions",
        "payments.calculate_split_fee",
    } <= modified

    sig = next(s for s in impact.signature_changes if s.symbol == "auth.verify_token")
    assert sig.breaking and sig.reasons == ["new required parameter `audience`"]
    non_breaking = next(
        s for s in impact.signature_changes if s.symbol == "payments.calculate_split_fee"
    )
    assert not non_breaking.breaking  # gaining a default is backwards compatible

    upstream = {n.symbol: n for n in impact.upstream}
    assert upstream["api_routes.get_current_user"].hop == 1
    assert upstream["api_routes.get_current_user"].call_lines == [24]
    assert (
        "verify_token(_bearer(authorization), AUTH_SECRET)"
        in upstream["api_routes.get_current_user"].snippet
    )
    assert upstream["api_routes.split_payment"].hop == 2

    # Only the caller the PR forgot to update is a violation; refresh_session was fixed.
    assert [(v.caller, v.file_path, v.line) for v in impact.contract_violations] == [
        ("api_routes.get_current_user", "api_routes.py", 24)
    ]
    assert impact.contract_violations[0].problems == ["missing required argument `audience`"]


@pytest.mark.parametrize(
    ("old", "new", "expected"),
    [
        (["a", "b"], ["a", "b", "c=1"], []),
        (["a", "b"], ["a", "b", "c"], ["new required parameter `c`"]),
        (["a", "b=2"], ["a"], ["parameter `b` was removed"]),
        (["a", "b=2"], ["a", "b"], ["parameter `b` lost its default and is now required"]),
        (["a", "b"], ["b", "a"], ["positional order changed: (a, b) -> (b, a)"]),
    ],
)
def test_compare_signatures(old, new, expected):
    def params(spec):
        return [
            Param(
                name=s.split("=")[0],
                has_default="=" in s,
                default=s.split("=")[1] if "=" in s else None,
            )
            for s in spec
        ]

    assert compare_signatures(params(old), params(new)) == expected


def test_check_call_detects_arity_and_keyword_errors():
    mod = parse_source("f(1, 2, 3)\nf(1, z=2)\nf(*args)\n", "m", "m.py")
    params = [Param(name="x"), Param(name="y", has_default=True, default="0")]
    too_many, bad_kw, splat = mod.calls
    assert "at most 2" in check_call(params, too_many)[0]
    assert check_call(params, bad_kw) == ["unexpected keyword argument `z`"]
    assert check_call(params, splat) == []


def test_diff_round_trip_and_strict_context(tmp_path: Path):
    files = parse_unified_diff(FIXTURE_DIFF.read_text())
    assert {f.path for f in files} == {"api_routes.py", "auth.py", "db.py", "payments.py"}
    db = next(f for f in files if f.path == "db.py")
    patched = apply_file_diff((FIXTURE_REPO / "db.py").read_text(), db)
    assert 'f"WHERE user_id = {user_id}"' in patched

    with pytest.raises(PatchError, match="out of order/range"):
        apply_file_diff("completely different\ncontent\n", db)
    shifted = "# a new header line\n" + (FIXTURE_REPO / "db.py").read_text()
    with pytest.raises(PatchError, match="context mismatch"):
        apply_file_diff(shifted, db)


def test_patch_cannot_escape_workspace(tmp_path: Path):
    evil = "--- /dev/null\n+++ b/../../escape.py\n@@ -0,0 +1 @@\n+print('x')\n"
    with pytest.raises(PatchError, match="outside the workspace"):
        apply_diff_to_tree(tmp_path, parse_unified_diff(evil))


def test_new_file_diff_is_applied(tmp_path: Path):
    diff = "--- /dev/null\n+++ b/new_mod.py\n@@ -0,0 +1,2 @@\n+def f():\n+    return 1\n"
    apply_diff_to_tree(tmp_path, parse_unified_diff(diff))
    assert (tmp_path / "new_mod.py").read_text() == "def f():\n    return 1\n"
