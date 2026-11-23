"""Synthesise targeted edge-case pytest modules for modified functions.

The synthesiser looks at arithmetic in each modified top-level function and derives the
inputs that would make a divisor zero or an optional operand None:

    x / p            -> p = 0            (p is a parameter)
    x // len(p)      -> p = []
    x // (100 - p)   -> p = 100
    p defaulting to None and used unguarded -> call with p omitted

The resulting test is run *differentially* against the base and head revisions. A case is a
regression only if it crashes on head but passed (or did not apply) on base, which filters
out pre-existing behaviour and produces execution-verified findings rather than guesses.
"""

from __future__ import annotations

import keyword
import re
from dataclasses import dataclass, field

from tree_sitter import Node

from app.ast_engine.impact_graph import ImpactGraph
from app.ast_engine.parser import SymbolDef, parse_tree
from app.schemas import ImpactNode, Param

_DIV_OPS = {"/", "//", "%"}
_ARITH_OPS = _DIV_OPS | {"+", "-", "*", "**"}
ACCEPTABLE_EXCEPTIONS = ("ValueError",)


@dataclass
class EdgeCase:
    case_id: str
    kwargs: dict[str, object]
    reason: str


@dataclass
class TestPlan:
    __test__ = False

    target: str
    module: str
    function: str
    file_path: str
    line: int
    cases: list[EdgeCase] = field(default_factory=list)

    @property
    def test_code(self) -> str:
        return render_test(self)


def _text(node: Node | None) -> str:
    return node.text.decode("utf-8") if node is not None and node.text else ""


def _walk(node: Node):
    stack = [node]
    while stack:
        n = stack.pop()
        yield n
        stack.extend(reversed(n.children))


def _unwrap(node: Node) -> Node:
    while node.type == "parenthesized_expression" and node.named_children:
        node = node.named_children[0]
    return node


def sample_value(p: Param) -> object:
    ann = (p.annotation or "").replace(" ", "").lower()
    name = p.name.lower()
    if "list" in ann or name.endswith("_ids") or (name.endswith("s") and "list" in ann):
        return [1, 2]
    if "dict" in ann:
        return {}
    if ann.startswith("bool"):
        return True
    if ann.startswith("str"):
        return "test"
    if ann.startswith("float") or "rate" in name or "ratio" in name:
        return 0.05
    if any(k in name for k in ("count", "participants", "n_", "num", "size")):
        return 2
    return 1000


def _guarded_none(source: str, param: str) -> bool:
    p = re.escape(param)
    return bool(
        re.search(
            rf"\b{p}\s+is\s+(not\s+)?None|if\s+(not\s+)?{p}\b|{p}\s+or\s+|\b{p}\s*(==|!=)\s*None",
            source,
        )
    )


def _divisor_cases(right: Node, params: dict[str, Param]) -> list[EdgeCase]:
    right = _unwrap(right)
    cases: list[EdgeCase] = []
    if right.type == "identifier" and right.text and _text(right) in params:
        p = _text(right)
        cases.append(EdgeCase(f"{p}=0", {p: 0}, f"`{p}` is used as a divisor"))
    elif right.type == "call" and _text(right.child_by_field_name("function")) == "len":
        args = right.child_by_field_name("arguments")
        inner = args.named_children[0] if args is not None and args.named_children else None
        if inner is not None and inner.type == "identifier" and _text(inner) in params:
            p = _text(inner)
            cases.append(EdgeCase(f"{p}=[]", {p: []}, f"`len({p})` is used as a divisor"))
    elif right.type == "binary_operator" and _text(right.child_by_field_name("operator")) in (
        "-",
        "+",
    ):
        left, rhs = right.child_by_field_name("left"), right.child_by_field_name("right")
        op = _text(right.child_by_field_name("operator"))
        for const, var in ((left, rhs), (rhs, left)):
            if const is None or var is None:
                continue
            const, var = _unwrap(const), _unwrap(var)
            if const.type == "integer" and var.type == "identifier" and _text(var) in params:
                n = int(_text(const))
                value = n if op == "-" else -n
                p = _text(var)
                cases.append(EdgeCase(f"{p}={value}", {p: value}, f"`{_text(right)}` is zero"))
                break
    return cases


def plan_for(sym: SymbolDef) -> TestPlan | None:
    if sym.kind != "function" or "." in sym.qualname or sym.name.startswith("_"):
        return None
    params = {p.name: p for p in sym.params if p.kind in ("positional", "keyword_only")}
    if not params or any(keyword.iskeyword(n) for n in params):
        return None
    tree = parse_tree(sym.source)
    offset = sym.start_line - 1
    cases: list[EdgeCase] = []
    division_line: int | None = None
    arithmetic_operands: set[str] = set()

    for node in _walk(tree.root_node):
        if node.type != "binary_operator":
            continue
        op = _text(node.child_by_field_name("operator"))
        left, right = node.child_by_field_name("left"), node.child_by_field_name("right")
        if op in _ARITH_OPS:
            for side in (left, right):
                if side is not None and _unwrap(side).type == "identifier":
                    arithmetic_operands.add(_text(_unwrap(side)))
        if op not in _DIV_OPS or right is None:
            continue
        if left is not None and left.type in ("string", "concatenated_string"):
            continue
        new = _divisor_cases(right, params)
        if new and division_line is None:
            division_line = node.start_point[0] + 1 + offset
        cases.extend(new)

    for name, p in params.items():
        unguarded = name in arithmetic_operands and not _guarded_none(sym.source, name)
        if p.has_default and p.default == "None" and unguarded:
            cases.append(EdgeCase(f"{name}=<omitted>", {}, f"`{name}` defaults to None"))

    if not cases:
        return None
    required = {n: sample_value(p) for n, p in params.items() if not p.has_default}
    final: dict[str, EdgeCase] = {}
    for c in cases:
        omitted = c.case_id.endswith("=<omitted>")
        target_param = c.case_id.split("=", 1)[0]
        kwargs = {k: v for k, v in required.items() if not (omitted and k == target_param)}
        kwargs.update(c.kwargs)
        final.setdefault(c.case_id, EdgeCase(c.case_id, kwargs, c.reason))
    return TestPlan(
        target=sym.fqname,
        module=sym.module,
        function=sym.name,
        file_path=sym.file_path,
        line=division_line or sym.start_line,
        cases=list(final.values()),
    )


def render_test(plan: TestPlan) -> str:
    cases = ",\n".join(f"    {c.case_id!r}: {c.kwargs!r}" for c in plan.cases)
    return f'''"""Auto-generated by AegisPR regression_tester for `{plan.target}`."""

import inspect

import pytest

try:
    import {plan.module} as target_module
except Exception:  # module absent or unimportable in this revision
    target_module = None

ACCEPTABLE = ({", ".join(ACCEPTABLE_EXCEPTIONS)},)  # explicit validation is the desired contract
CASES = {{
{cases},
}}


@pytest.mark.parametrize("case_id", list(CASES))
def test_{plan.function}_edge_cases(case_id):
    fn = getattr(target_module, "{plan.function}", None)
    if fn is None:
        pytest.skip("target not present in this revision")
    kwargs = CASES[case_id]
    try:
        inspect.signature(fn).bind(**kwargs)
    except TypeError:
        pytest.skip("case not applicable to this signature")
    try:
        fn(**kwargs)
    except ACCEPTABLE:
        pass
'''


def synthesize(modified: list[ImpactNode], head: ImpactGraph) -> list[TestPlan]:
    plans = []
    for node in modified:
        sym = head.symbols.get(node.symbol)
        if sym is None:
            continue
        plan = plan_for(sym)
        if plan is not None:
            plans.append(plan)
    return plans
