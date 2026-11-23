"""NetworkX call/import graph and change-impact resolution.

Nodes are fully-qualified symbols (`auth.verify_token`, `api_routes.get_current_user`) plus
one node per module. Edges point from *dependent* to *dependency*:

    caller   --calls-->      callee        (direct call, call-site shapes stored on the edge)
    function --references--> function      (function passed as a value, e.g. Depends(fn))
    module   --imports-->    symbol/module

Upstream impact of a change is therefore a reverse BFS over `calls`/`references` edges.
Everything here is deterministic and runs before any LLM call: the agents receive exact
caller snippets and pre-computed contract violations instead of guessing from a raw diff.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import networkx as nx

from app.ast_engine.parser import MODULE_SCOPE, CallSite, ParsedModule, SymbolDef, parse_file
from app.schemas import (
    ContractViolation,
    FileDiff,
    ImpactAnalysis,
    ImpactNode,
    Param,
    SignatureChange,
)

EXCLUDED_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "node_modules",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "build",
    "dist",
}
DEPENDENCY_EDGES = ("calls", "references")
MAX_SNIPPET_LINES = 30


class ImpactGraph:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.graph: nx.DiGraph = nx.DiGraph()
        self.modules: dict[str, ParsedModule] = {}
        self.by_path: dict[str, ParsedModule] = {}
        self.symbols: dict[str, SymbolDef] = {}

    # -- construction ----------------------------------------------------------------

    @classmethod
    def from_repo(cls, root: Path) -> ImpactGraph:
        g = cls(root)
        for path in sorted(root.rglob("*.py")):
            if any(part in EXCLUDED_DIRS for part in path.relative_to(root).parts):
                continue
            g.add_module(parse_file(path, root))
        g.link()
        return g

    def add_module(self, mod: ParsedModule) -> None:
        self.modules[mod.module] = mod
        self.by_path[mod.file_path] = mod
        n_lines = max(1, mod.source.count("\n") + 1)
        self.graph.add_node(
            mod.module, kind="module", file_path=mod.file_path, start_line=1, end_line=n_lines
        )
        for s in mod.symbols:
            self.symbols[s.fqname] = s
            self.graph.add_node(
                s.fqname,
                kind=s.kind,
                file_path=s.file_path,
                start_line=s.start_line,
                end_line=s.end_line,
            )

    def _bindings(self, mod: ParsedModule) -> dict[str, str]:
        out: dict[str, str] = {}
        for imp in mod.imports:
            out[imp.alias] = f"{imp.module}.{imp.name}" if imp.name else imp.module
        return out

    def _known(self, fq: str) -> bool:
        return fq in self.symbols or fq in self.modules

    def resolve(self, mod: ParsedModule, scope: str, expr: str) -> str | None:
        """Resolve a call/reference expression to a known fully-qualified symbol."""
        if not expr or "(" in expr or "[" in expr:
            return None
        parts = expr.split(".")
        scope_parts = [] if scope == MODULE_SCOPE else scope.split(".")

        if parts[0] in ("self", "cls") and len(parts) == 2:
            for i in range(len(scope_parts), 0, -1):
                owner = mod.symbol(".".join(scope_parts[:i]))
                if owner is not None and owner.kind == "class":
                    fq = f"{mod.module}.{owner.qualname}.{parts[1]}"
                    return fq if fq in self.symbols else None
            return None

        # Lexically enclosing scopes first (nested functions), then module level.
        for i in range(len(scope_parts), -1, -1):
            prefix = ".".join(scope_parts[:i])
            fq = f"{mod.module}.{prefix + '.' if prefix else ''}{expr}"
            if fq in self.symbols:
                return fq

        bindings = self._bindings(mod)
        if parts[0] in bindings:
            target = bindings[parts[0]]
            fq = ".".join([target, *parts[1:]])
            if self._known(fq):
                return fq
        return None

    def _add_dependency(self, src: str, dst: str, kind: str, site: CallSite) -> None:
        if src == dst:
            return
        if not self.graph.has_edge(src, dst):
            self.graph.add_edge(src, dst, kind=kind, sites=[], refs=[])
        data = self.graph.edges[src, dst]
        if kind == "calls":
            data["kind"] = "calls"  # a direct call dominates a by-value reference
            data["sites"].append(site)
        else:
            data["refs"].append(site)

    def link(self) -> None:
        for mod in self.modules.values():
            for alias_target in self._bindings(mod).values():
                if self._known(alias_target) and not self.graph.has_edge(mod.module, alias_target):
                    self.graph.add_edge(mod.module, alias_target, kind="imports", sites=[], refs=[])
            for call in mod.calls:
                caller = (
                    mod.module if call.caller == MODULE_SCOPE else f"{mod.module}.{call.caller}"
                )
                callee = self.resolve(mod, call.caller, call.callee_expr)
                if callee is not None and callee in self.symbols:
                    self._add_dependency(caller, callee, "calls", call)
                for ref in call.referenced_names:
                    target = self.resolve(mod, call.caller, ref)
                    if target is not None and target in self.symbols:
                        self._add_dependency(caller, target, "references", call)

    # -- queries ---------------------------------------------------------------------

    def upstream(self, fq: str, max_hops: int = 2) -> list[tuple[str, int, str, str]]:
        """Reverse BFS. Returns (node, hop, via, edge_kind) with each node at its min hop."""
        if fq not in self.graph:
            return []
        seen = {fq}
        out: list[tuple[str, int, str, str]] = []
        queue: deque[tuple[str, int]] = deque([(fq, 0)])
        while queue:
            node, hop = queue.popleft()
            if hop >= max_hops:
                continue
            for pred in sorted(self.graph.predecessors(node)):
                kind = self.graph.edges[pred, node]["kind"]
                if kind not in DEPENDENCY_EDGES or pred in seen:
                    continue
                seen.add(pred)
                out.append((pred, hop + 1, node, kind))
                queue.append((pred, hop + 1))
        return out

    def downstream(self, fq: str) -> list[tuple[str, str]]:
        if fq not in self.graph:
            return []
        return [
            (succ, self.graph.edges[fq, succ]["kind"])
            for succ in sorted(self.graph.successors(fq))
            if self.graph.edges[fq, succ]["kind"] in DEPENDENCY_EDGES
        ]

    def call_sites(self, caller: str, callee: str) -> list[CallSite]:
        if not self.graph.has_edge(caller, callee):
            return []
        return list(self.graph.edges[caller, callee]["sites"])

    def reference_sites(self, caller: str, callee: str) -> list[CallSite]:
        if not self.graph.has_edge(caller, callee):
            return []
        return list(self.graph.edges[caller, callee]["refs"])

    def snippet(self, fq: str, max_lines: int = MAX_SNIPPET_LINES) -> str:
        data = self.graph.nodes.get(fq)
        if not data:
            return ""
        mod = self.by_path[data["file_path"]]
        start, end = data["start_line"], data["end_line"]
        if data["kind"] == "module":
            return ""
        text = mod.lines(start, min(end, start + max_lines - 1))
        if end - start + 1 > max_lines:
            text += "\n    # ..."
        return text

    def node(self, fq: str, direction: str, hop: int, **extra) -> ImpactNode:
        data = self.graph.nodes[fq]
        return ImpactNode(
            symbol=fq,
            kind=data["kind"],
            file_path=data["file_path"],
            start_line=data["start_line"],
            end_line=data["end_line"],
            direction=direction,  # type: ignore[arg-type]
            hop=hop,
            snippet=self.snippet(fq),
            **extra,
        )

    def stats(self) -> dict[str, int]:
        kinds = [d["kind"] for _, _, d in self.graph.edges(data=True)]
        return {
            "modules": len(self.modules),
            "symbols": len(self.symbols),
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "call_edges": kinds.count("calls"),
            "reference_edges": kinds.count("references"),
            "import_edges": kinds.count("imports"),
        }


# ---------------------------------------------------------------------------------------
# Signature / contract analysis
# ---------------------------------------------------------------------------------------


def _strip_self(params: list[Param], kind: str) -> list[Param]:
    if kind == "method" and params and params[0].name in ("self", "cls"):
        return params[1:]
    return params


def compare_signatures(old: list[Param], new: list[Param]) -> list[str]:
    """Return human-readable reasons why `new` is not call-compatible with `old`."""
    reasons: list[str] = []
    new_by_name = {p.name: p for p in new}
    old_by_name = {p.name: p for p in old}
    new_varkw = any(p.kind == "var_keyword" for p in new)
    new_varpos = any(p.kind == "var_positional" for p in new)

    for p in old:
        if p.kind == "var_positional" and not new_varpos:
            reasons.append(f"`*{p.name}` was removed")
        elif p.kind == "var_keyword" and not new_varkw:
            reasons.append(f"`**{p.name}` was removed")
        elif p.kind in ("positional", "keyword_only"):
            q = new_by_name.get(p.name)
            if q is None:
                if not new_varkw:
                    reasons.append(f"parameter `{p.name}` was removed")
            else:
                if p.has_default and not q.has_default:
                    reasons.append(f"parameter `{p.name}` lost its default and is now required")
                if p.kind == "positional" and q.kind == "keyword_only":
                    reasons.append(f"parameter `{p.name}` became keyword-only")

    for q in new:
        if (
            q.kind in ("positional", "keyword_only")
            and not q.has_default
            and q.name not in old_by_name
        ):
            reasons.append(f"new required parameter `{q.name}`")

    old_pos = [p.name for p in old if p.kind == "positional" and p.name in new_by_name]
    new_pos = [p.name for p in new if p.kind == "positional" and p.name in old_by_name]
    if old_pos != new_pos and sorted(old_pos) == sorted(new_pos):
        reasons.append(
            f"positional order changed: ({', '.join(old_pos)}) -> ({', '.join(new_pos)})"
        )
    return reasons


def check_call(params: list[Param], call: CallSite) -> list[str]:
    """Statically check a call site's argument shape against a parameter list."""
    if call.has_star_args or call.has_star_kwargs:
        return []  # cannot reason about splatted arguments statically
    positional = [p for p in params if p.kind == "positional"]
    named = {p.name for p in params if p.kind in ("positional", "keyword_only")}
    has_varpos = any(p.kind == "var_positional" for p in params)
    has_varkw = any(p.kind == "var_keyword" for p in params)
    problems: list[str] = []

    if call.positional_args > len(positional) and not has_varpos:
        problems.append(
            f"passes {call.positional_args} positional arguments but at most "
            f"{len(positional)} are accepted"
        )
    bound = {p.name for p in positional[: call.positional_args]}
    for kw in call.keyword_args:
        if kw in bound:
            problems.append(f"got multiple values for argument `{kw}`")
        elif kw not in named and not has_varkw:
            problems.append(f"unexpected keyword argument `{kw}`")
    supplied = bound | set(call.keyword_args)
    for p in params:
        if (
            p.kind in ("positional", "keyword_only")
            and not p.has_default
            and p.name not in supplied
        ):
            problems.append(f"missing required argument `{p.name}`")
    return problems


def _callee_params(graph: ImpactGraph, fq: str, call: CallSite) -> list[Param] | None:
    sym = graph.symbols.get(fq)
    if sym is None:
        return None
    if sym.kind == "class":
        init = graph.symbols.get(f"{fq}.__init__")
        return _strip_self(init.params, "method") if init else None
    if sym.kind == "method":
        # `self.m(...)` / `obj.m(...)` bind self implicitly; `Cls.m(obj, ...)` does not.
        owner = fq.rsplit(".", 1)[0].rsplit(".", 1)[-1]
        head = call.callee_expr.split(".")[0]
        if "." in call.callee_expr and head != owner:
            return _strip_self(sym.params, "method")
    return sym.params


# ---------------------------------------------------------------------------------------
# Change impact
# ---------------------------------------------------------------------------------------


def _touched_symbols(
    base: ImpactGraph, head: ImpactGraph, files: list[FileDiff]
) -> dict[str, tuple[SymbolDef | None, SymbolDef | None]]:
    touched: dict[str, tuple[SymbolDef | None, SymbolDef | None]] = {}
    for fd in files:
        if not fd.is_python:
            continue
        if fd.old_path and fd.old_path in base.by_path:
            mod = base.by_path[fd.old_path]
            for ln in sorted(fd.removed_line_numbers()):
                s = mod.innermost_symbol(ln)
                if s is not None:
                    touched.setdefault(s.fqname, (None, None))
        if fd.new_path and fd.new_path in head.by_path:
            mod = head.by_path[fd.new_path]
            for ln in sorted(fd.added_line_numbers()):
                s = mod.innermost_symbol(ln)
                if s is not None:
                    touched.setdefault(s.fqname, (None, None))
    return {fq: (base.symbols.get(fq), head.symbols.get(fq)) for fq in touched}


def compute_impact(
    base_root: Path, head_root: Path, files: list[FileDiff], max_hops: int = 2
) -> tuple[ImpactAnalysis, ImpactGraph, ImpactGraph]:
    base = ImpactGraph.from_repo(base_root)
    head = ImpactGraph.from_repo(head_root)
    touched = _touched_symbols(base, head, files)
    added_by_file = {fd.new_path: fd.added_line_numbers() for fd in files if fd.new_path}

    analysis = ImpactAnalysis(files=files, graph_stats=head.stats())
    seen_upstream: set[str] = set()
    changed_signatures: set[str] = set()

    for fq, (old, new) in sorted(touched.items()):
        graph = head if new is not None else base
        analysis.modified_symbols.append(graph.node(fq, "modified", 0))

        if old is not None and old.kind in ("function", "method"):
            if new is None:
                analysis.signature_changes.append(
                    SignatureChange(
                        symbol=fq,
                        file_path=old.file_path,
                        line=old.start_line,
                        old_signature=old.signature,
                        new_signature=None,
                        breaking=bool(base.upstream(fq, 1)),
                        reasons=["symbol was removed"],
                    )
                )
            elif old.signature != new.signature:
                reasons = compare_signatures(
                    _strip_self(old.params, old.kind), _strip_self(new.params, new.kind)
                )
                changed_signatures.add(fq)
                analysis.signature_changes.append(
                    SignatureChange(
                        symbol=fq,
                        file_path=new.file_path,
                        line=new.start_line,
                        old_signature=old.signature,
                        new_signature=new.signature,
                        breaking=bool(reasons),
                        reasons=reasons,
                    )
                )

        for node, hop, via, edge_kind in graph.upstream(fq, max_hops):
            if node in seen_upstream or node in touched:
                continue
            seen_upstream.add(node)
            sites = graph.call_sites(node, via) + graph.reference_sites(node, via)
            call_lines = sorted({s.line for s in sites})
            analysis.upstream.append(
                graph.node(
                    node, "upstream", hop, via=via, edge_kind=edge_kind, call_lines=call_lines
                )
            )
        for succ, edge_kind in graph.downstream(fq):
            analysis.downstream.append(
                graph.node(succ, "downstream", 1, via=fq, edge_kind=edge_kind)
            )

    # Contract violations: every call to a symbol whose signature changed, plus every call
    # site introduced by this PR, is checked against the post-change signature.
    violations: dict[tuple[str, int, str], ContractViolation] = {}
    for src, dst, data in head.graph.edges(data=True):
        if data["kind"] != "calls":
            continue
        caller_file = head.graph.nodes[src]["file_path"]
        for site in data["sites"]:
            new_site = site.line in added_by_file.get(caller_file, set())
            if dst not in changed_signatures and not new_site:
                continue
            params = _callee_params(head, dst, site)
            if params is None:
                continue
            problems = check_call(params, site)
            if problems:
                violations[(caller_file, site.line, dst)] = ContractViolation(
                    callee=dst,
                    caller=src,
                    file_path=caller_file,
                    line=site.line,
                    call_text=site.text,
                    problems=problems,
                    caller_in_diff=new_site,
                )
    analysis.contract_violations = sorted(violations.values(), key=lambda v: (v.file_path, v.line))
    return analysis, base, head
