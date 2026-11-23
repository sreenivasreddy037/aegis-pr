"""Agent node implementations: impact analysis, three parallel reviewers, and the critic."""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

from app.analysis.edge_cases import TestPlan, synthesize
from app.analysis.security_rules import scan_diff
from app.ast_engine.impact_graph import compute_impact
from app.config import Settings
from app.graph.llm import ReviewLLM
from app.graph.prompts import CONTRACT_PROMPT, CRITIC_PROMPT, REGRESSION_PROMPT, SECURITY_PROMPT
from app.retrieval.policy_store import Embedder, PolicyStore, make_embedder
from app.sandbox.test_runner import run_pytest
from app.schemas import (
    Category,
    DroppedFinding,
    Evidence,
    ImpactAnalysis,
    PolicySnippet,
    PRReviewReport,
    ReviewFinding,
    SandboxResult,
    SandboxRun,
    Severity,
)
from app.workspace import ReviewWorkspace

logger = logging.getLogger(__name__)

MAX_DIFF_CHARS = 12_000


# ---------------------------------------------------------------------------------------
# Context rendering shared by the LLM-backed agents
# ---------------------------------------------------------------------------------------


def render_diff(ws: ReviewWorkspace) -> str:
    parts = []
    for fd in ws.files:
        parts.append(f"--- {fd.old_path or '/dev/null'}\n+++ {fd.new_path or '/dev/null'}")
        for h in fd.hunks:
            parts.append(f"@@ -{h.old_start},{h.old_len} +{h.new_start},{h.new_len} @@")
            for ln in h.lines:
                tag = {"add": "+", "remove": "-", "context": " "}[ln.kind]
                no = ln.new_lineno if ln.new_lineno is not None else ln.old_lineno
                parts.append(f"{tag}{no:>4} {ln.content}")
    text = "\n".join(parts)
    return text if len(text) <= MAX_DIFF_CHARS else text[:MAX_DIFF_CHARS] + "\n... [diff truncated]"


def render_context(ws: ReviewWorkspace, impact: ImpactAnalysis, agent: str) -> str:
    sections = [
        "## Diff (line numbers refer to the head revision for + and context lines)",
        render_diff(ws),
        "## Modified symbols",
    ]
    sections += [
        f"- {n.symbol} ({n.file_path}:{n.start_line}-{n.end_line})" for n in impact.modified_symbols
    ]
    if impact.signature_changes:
        sections.append("## Signature changes")
        sections += [
            f"- {s.symbol}: `{s.old_signature}` -> `{s.new_signature}`"
            f" breaking={s.breaking} {'; '.join(s.reasons)}"
            for s in impact.signature_changes
        ]
    if impact.upstream:
        sections.append("## Upstream callers (AST impact graph)")
        for n in impact.upstream:
            sections.append(
                f"### {n.symbol} — hop {n.hop}, {n.edge_kind} {n.via} at "
                f"{n.file_path}:{','.join(map(str, n.call_lines))}\n```python\n{n.snippet}\n```"
            )
    policies = impact.policies.get(agent, [])
    if policies:
        sections.append("## Repository policies & ADRs (hybrid retrieval)")
        sections += [f"- [{p.source}] {p.title}: {p.text}" for p in policies]
    return "\n".join(sections)


def _policy_evidence(
    policies: list[PolicySnippet], cwe: str | None, category: Category
) -> list[Evidence]:
    out = []
    for p in policies:
        if (
            (cwe and cwe in p.tags)
            or (p.source.endswith(".md") and category.value in p.text.lower())
            or (category == Category.BREAKING_CHANGE and "contract" in p.tags)
        ):
            out.append(Evidence(kind="policy", detail=f"[{p.source}] {p.title}"))
    adr = {
        "CWE-89": "ADR-001",
        "CWE-798": "ADR-004",
        "CWE-532": "ADR-005",
        "CWE-369": "ADR-003",
        "CWE-476": "ADR-003",
    }.get(cwe or "", "ADR-002" if category == Category.BREAKING_CHANGE else None)
    if adr:
        out += [
            Evidence(kind="policy", detail=f"[{p.source}] {p.title}")
            for p in policies
            if adr in p.title
        ]
    seen, unique = set(), []
    for e in out:
        if e.detail not in seen:
            seen.add(e.detail)
            unique.append(e)
    return unique[:3]


# ---------------------------------------------------------------------------------------
# Node: analyze_impact
# ---------------------------------------------------------------------------------------


@lru_cache(maxsize=4)
def _cached_embedder(backend: str, model: str, offline: bool) -> Embedder:
    s = Settings(embedding_backend=backend, fastembed_model=model, mock_llm=offline)  # type: ignore[arg-type]
    return make_embedder(s)


def analyze_impact(ws: ReviewWorkspace, settings: Settings) -> ImpactAnalysis:
    impact, base_graph, head_graph = compute_impact(
        ws.base_dir, ws.head_dir, ws.files, max_hops=settings.impact_max_hops
    )
    ws.base_graph, ws.head_graph = base_graph, head_graph
    for n in impact.modified_symbols:
        ws.modified_ranges.setdefault(n.file_path, set()).update(
            range(n.start_line, n.end_line + 1)
        )

    added_text = "\n".join(ln.content for fd in ws.files for h in fd.hunks for ln in h.added)[:2000]
    queries = {
        "security_auditor": f"security vulnerability injection secrets logging {added_text}",
        "contract_reviewer": "function signature change backward compatible callers "
        + " ".join(f"{s.symbol} {' '.join(s.reasons)}" for s in impact.signature_changes),
        "regression_tester": "edge case zero None division validation money math "
        + " ".join(n.symbol for n in impact.modified_symbols),
    }
    store = PolicyStore(
        settings,
        embedder=_cached_embedder(
            settings.embedding_backend, settings.fastembed_model, settings.offline
        ),
    )
    try:
        store.index_repo(ws.base_dir)
        impact.policies = {agent: store.search(q) for agent, q in queries.items()}
    finally:
        store.close()
    return impact


# ---------------------------------------------------------------------------------------
# Node: security_auditor
# ---------------------------------------------------------------------------------------


def security_auditor(
    ws: ReviewWorkspace, impact: ImpactAnalysis, llm: ReviewLLM
) -> list[ReviewFinding]:
    assert ws.head_graph is not None
    candidates = scan_diff(ws.files, ws.head_dir, ws.head_graph, ws.modified_ranges)
    policies = impact.policies.get("security_auditor", [])
    for c in candidates:
        c.evidence += _policy_evidence(policies, c.cwe, c.category)
    return llm.review(
        "security_auditor",
        SECURITY_PROMPT,
        render_context(ws, impact, "security_auditor"),
        candidates,
    )


# ---------------------------------------------------------------------------------------
# Node: contract_reviewer
# ---------------------------------------------------------------------------------------


def contract_reviewer(
    ws: ReviewWorkspace, impact: ImpactAnalysis, llm: ReviewLLM, settings: Settings
) -> list[ReviewFinding]:
    policies = impact.policies.get("contract_reviewer", [])
    violations = impact.contract_violations
    if settings.ablation == "no_impact_graph":  # naive baseline: only look at files in the diff
        diff_paths = {fd.path for fd in ws.files}
        violations = [v for v in violations if v.file_path in diff_paths]
    sig_by_symbol = {s.symbol: s for s in impact.signature_changes}
    candidates: list[ReviewFinding] = []
    flagged_callees: set[str] = set()

    for v in violations:
        sig = sig_by_symbol.get(v.callee)
        flagged_callees.add(v.callee)
        blast = [
            (caller, hop, kind)
            for caller, hop, _via, kind in (
                ws.head_graph.upstream(v.caller, 1) if ws.head_graph else []
            )
        ]
        callee_short = v.callee.rsplit(".", 1)[-1]
        change = (
            f"`{sig.old_signature}` → `{sig.new_signature}` ({'; '.join(sig.reasons)})"
            if sig and sig.old_signature
            else f"the current signature of `{v.callee}`"
        )
        blast_txt = (
            " Blast radius: "
            + ", ".join(f"`{c}` ({k}, hop {h + 1})" for c, h, k in blast)
            + " depend on the broken caller."
            if blast
            else ""
        )
        evidence = [
            Evidence(kind="caller", detail=f"{v.file_path}:{v.line}: {v.call_text}"),
            Evidence(kind="static", detail=f"AST contract check: {'; '.join(v.problems)}"),
        ]
        if sig:
            evidence.append(
                Evidence(kind="diff", detail=f"{sig.file_path}:{sig.line}: {sig.new_signature}")
            )
        evidence += [
            Evidence(kind="caller", detail=f"blast radius (hop {h + 1}): {c} {k} {v.caller}")
            for c, h, k in blast
        ]
        evidence += _policy_evidence(policies, None, Category.BREAKING_CHANGE)
        candidates.append(
            ReviewFinding(
                agent="contract_reviewer",
                category=Category.BREAKING_CHANGE,
                severity=Severity.HIGH,
                title=f"Caller `{v.caller}` breaks against new `{callee_short}` signature",
                description=f"`{v.caller}` calls `{v.call_text}`, but the PR changes {change}. "
                f"At runtime this raises `TypeError` ({'; '.join(v.problems)}).{blast_txt}",
                file_path=v.file_path,
                line=v.line,
                symbol=v.caller,
                confidence=0.98,
                verified=True,
                evidence=evidence,
                suggested_fix=f"Update the call site to satisfy the new signature, or keep "
                f"`{callee_short}` backward compatible by giving new parameters defaults.",
            )
        )

    # Breaking public signatures without in-repo callers: may still break external consumers.
    for s in impact.signature_changes:
        if s.breaking and s.symbol not in flagged_callees and s.new_signature is not None:
            short = s.symbol.rsplit(".", 1)[-1]
            if short.startswith("_"):
                continue
            candidates.append(
                ReviewFinding(
                    agent="contract_reviewer",
                    category=Category.BREAKING_CHANGE,
                    severity=Severity.MEDIUM,
                    title=f"Breaking signature change to public `{short}`",
                    description=f"`{s.old_signature}` → `{s.new_signature}` ({'; '.join(s.reasons)}). "
                    "No in-repo callers break, but external consumers may.",
                    file_path=s.file_path,
                    line=s.line,
                    symbol=s.symbol,
                    confidence=0.5,
                    evidence=[
                        Evidence(kind="diff", detail=f"{s.file_path}:{s.line}: {s.new_signature}")
                    ],
                    suggested_fix="Add defaults for new parameters or deprecate the old signature.",
                )
            )

    # Removed symbols that callers in the head revision still reference by name.
    for s in impact.signature_changes:
        if s.new_signature is not None or ws.base_graph is None or ws.head_graph is None:
            continue
        short = s.symbol.rsplit(".", 1)[-1]
        for caller, _hop, _via, _kind in ws.base_graph.upstream(s.symbol, 1):
            node = ws.head_graph.graph.nodes.get(caller)
            if not node or node["kind"] == "module":
                continue
            lines = ws.file_lines(node["file_path"])
            for ln in range(node["start_line"], node["end_line"] + 1):
                if ln <= len(lines) and re.search(rf"\b{re.escape(short)}\s*\(", lines[ln - 1]):
                    candidates.append(
                        ReviewFinding(
                            agent="contract_reviewer",
                            category=Category.BREAKING_CHANGE,
                            severity=Severity.CRITICAL,
                            title=f"`{caller}` still calls removed `{short}`",
                            description=f"The PR removes `{s.symbol}` but `{caller}` still calls it; "
                            "this raises `NameError`/`ImportError` at runtime.",
                            file_path=node["file_path"],
                            line=ln,
                            symbol=caller,
                            confidence=0.95,
                            verified=True,
                            evidence=[
                                Evidence(
                                    kind="caller",
                                    detail=f"{node['file_path']}:{ln}: {lines[ln - 1].strip()}",
                                )
                            ],
                        )
                    )
                    break

    return llm.review(
        "contract_reviewer",
        CONTRACT_PROMPT,
        render_context(ws, impact, "contract_reviewer"),
        candidates,
    )


# ---------------------------------------------------------------------------------------
# Node: regression_tester
# ---------------------------------------------------------------------------------------


def _differential(
    code: str, ws: ReviewWorkspace, timeout: int
) -> tuple[SandboxResult, SandboxResult]:
    with ThreadPoolExecutor(max_workers=2) as pool:
        base_f = pool.submit(run_pytest, code, ws.base_dir, timeout_sec=timeout)
        head_f = pool.submit(run_pytest, code, ws.head_dir, timeout_sec=timeout)
        return base_f.result(), head_f.result()


def _regressions(base: SandboxResult, head: SandboxResult) -> list[tuple[str, str, str]]:
    """(case, head_message, base_status) for cases failing on head but not on base."""
    out = []
    base_by_name = {c.name: c for c in base.cases}
    for c in head.cases:
        if c.status not in ("failed", "error"):
            continue
        b = base_by_name.get(c.name)
        if b is None or b.status in ("passed", "skipped"):
            out.append((c.name, c.message, b.status if b else "absent"))
    return out


def regression_tester(
    ws: ReviewWorkspace, impact: ImpactAnalysis, llm: ReviewLLM, settings: Settings
) -> tuple[list[ReviewFinding], list[SandboxRun]]:
    assert ws.head_graph is not None
    if settings.ablation == "no_sandbox":
        return [], []
    plans: list[TestPlan] = synthesize(impact.modified_symbols, ws.head_graph)
    jobs: list[tuple[str, str, str, int]] = [
        (p.target, p.test_code, p.file_path, p.line) for p in plans
    ]

    for t in llm.propose_tests(REGRESSION_PROMPT, render_context(ws, impact, "regression_tester")):
        sym = ws.head_graph.symbols.get(t.target)
        if sym is not None:
            jobs.append((t.target, t.test_code, sym.file_path, sym.start_line))

    findings: list[ReviewFinding] = []
    runs: list[SandboxRun] = []
    if not jobs:
        return findings, runs
    with ThreadPoolExecutor(max_workers=min(4, len(jobs))) as pool:
        results = list(
            pool.map(lambda j: _differential(j[1], ws, settings.sandbox_timeout_sec), jobs)
        )

    policies = impact.policies.get("regression_tester", [])
    for (target, code, file_path, line), (base, head) in zip(jobs, results, strict=True):
        regs = _regressions(base, head)
        runs.append(
            SandboxRun(
                target=target,
                test_code=code,
                base=base,
                head=head,
                regressions=[r[0] for r in regs],
            )
        )
        if head.timed_out and not base.timed_out:
            regs.append(
                ("timeout", f"head revision exceeded {settings.sandbox_timeout_sec}s", "passed")
            )
        if not regs:
            continue
        short = target.rsplit(".", 1)[-1]
        crashes = "; ".join(
            f"`{_case_label(name)}` → {msg.splitlines()[0] if msg else 'failure'}"
            for name, msg, _ in regs
        )
        cwe = "CWE-369" if any("ZeroDivisionError" in m for _, m, _ in regs) else "CWE-476"
        evidence = [
            Evidence(
                kind="sandbox",
                detail=f"head: {_case_label(name)} → {msg.splitlines()[0] if msg else 'failed'}; "
                f"base: {b_status}",
            )
            for name, msg, b_status in regs
        ]
        evidence.append(
            Evidence(
                kind="sandbox",
                detail=f"differential run: base {base.duration_ms:.0f} ms, "
                f"head {head.duration_ms:.0f} ms, exit codes {base.exit_code}/{head.exit_code}",
            )
        )
        evidence += _policy_evidence(policies, cwe, Category.REGRESSION)
        findings.append(
            ReviewFinding(
                agent="regression_tester",
                category=Category.REGRESSION,
                severity=Severity.HIGH,
                title=f"`{short}` crashes on edge-case inputs (sandbox-verified)",
                description=f"A synthesized pytest run against both revisions shows `{target}` now "
                f"crashes where the base revision did not: {crashes}. Callers passing these values "
                "get an unhandled exception instead of a validation error.",
                file_path=file_path,
                line=line,
                symbol=target,
                cwe=cwe,
                confidence=0.97,
                verified=True,
                evidence=evidence,
                suggested_fix="Restore explicit input validation (e.g. `if not participants or "
                "participants <= 0: raise ValueError(...)`) before the arithmetic.",
            )
        )
    findings = llm.review(
        "regression_tester",
        REGRESSION_PROMPT,
        render_context(ws, impact, "regression_tester"),
        findings,
    )
    return findings, runs


def _case_label(test_name: str) -> str:
    m = re.search(r"\[(.*)\]$", test_name)
    return m.group(1) if m else test_name


# ---------------------------------------------------------------------------------------
# Node: critic_synthesizer
# ---------------------------------------------------------------------------------------


def _location_problem(f: ReviewFinding, ws: ReviewWorkspace, impact: ImpactAnalysis) -> str | None:
    lines = ws.file_lines(f.file_path)
    if not lines:
        return f"file `{f.file_path}` does not exist in the head revision"
    if not 1 <= f.line <= len(lines):
        return f"line {f.line} is outside `{f.file_path}` (1-{len(lines)})"
    if f.category in (Category.SECURITY, Category.REGRESSION):
        fd = next((x for x in ws.files if x.path == f.file_path), None)
        if fd is None:
            return f"`{f.file_path}` is not part of the diff"
        changed = fd.added_line_numbers() | ws.modified_ranges.get(f.file_path, set())
        if f.line not in changed:
            return f"line {f.line} is not in code changed by this PR"
    else:  # breaking change: must point at a known call site or changed signature
        anchors = {(v.file_path, v.line) for v in impact.contract_violations}
        anchors |= {(s.file_path, s.line) for s in impact.signature_changes}
        callee_names = {s.symbol.rsplit(".", 1)[-1] for s in impact.signature_changes}
        line_text = lines[f.line - 1]
        if (f.file_path, f.line) not in anchors and not any(n in line_text for n in callee_names):
            return f"no call site of a changed signature at {f.location}"
    return None


def _same_issue(a: ReviewFinding, b: ReviewFinding) -> bool:
    if a.file_path != b.file_path or abs(a.line - b.line) > 2:
        return False
    if a.cwe and b.cwe:
        return a.cwe == b.cwe or {a.cwe, b.cwe} <= {"CWE-369", "CWE-476"}
    return a.category == b.category


def _rank(f: ReviewFinding) -> tuple:
    return (f.verified, f.severity.rank, f.confidence)


def critic_synthesizer(
    ws: ReviewWorkspace,
    impact: ImpactAnalysis,
    findings: list[ReviewFinding],
    llm: ReviewLLM,
    settings: Settings,
) -> tuple[list[ReviewFinding], list[DroppedFinding], str]:
    dropped: list[DroppedFinding] = []

    # 1. Ground every finding in the diff / impact graph.
    grounded = []
    for f in findings:
        problem = _location_problem(f, ws, impact)
        if problem:
            dropped.append(DroppedFinding(finding=f, reason=f"hallucinated location: {problem}"))
        else:
            grounded.append(f)

    # 2. Deduplicate overlapping findings from different agents; keep the strongest.
    clusters: list[list[ReviewFinding]] = []
    for f in sorted(grounded, key=_rank, reverse=True):
        for cluster in clusters:
            if _same_issue(cluster[0], f):
                cluster.append(f)
                break
        else:
            clusters.append([f])
    merged: list[ReviewFinding] = []
    for cluster in clusters:
        best = cluster[0].model_copy(deep=True)
        for other in cluster[1:]:
            if other.agent != best.agent and other.agent not in best.corroborated_by:
                best.corroborated_by.append(other.agent)
            known = {e.detail for e in best.evidence}
            best.evidence += [e for e in other.evidence if e.detail not in known]
            best.confidence = max(best.confidence, other.confidence)
            dropped.append(
                DroppedFinding(
                    finding=other, reason=f"merged into {best.agent} finding at {best.location}"
                )
            )
        merged.append(best)

    # 3. Filter unverified low-confidence findings.
    kept = []
    for f in merged:
        if not f.verified and f.confidence < settings.confidence_threshold:
            dropped.append(
                DroppedFinding(
                    finding=f,
                    reason=f"unverified and confidence {f.confidence:.2f} < {settings.confidence_threshold:.2f}",
                )
            )
        else:
            kept.append(f)

    kept.sort(key=lambda f: (-f.severity.rank, not f.verified, f.file_path, f.line))
    for i, f in enumerate(kept, start=1):
        f.id = f"AEG-{i:03d}"

    summary = _deterministic_summary(kept, dropped, impact)
    if kept and llm.mode == "live":
        listing = "\n".join(
            f"- {f.id} [{f.severity}] {f.title} at {f.location}: {f.description}" for f in kept
        )
        summary = llm.summarize(CRITIC_PROMPT, f"{summary}\n\nFindings:\n{listing}") or summary
    return kept, dropped, summary


def _deterministic_summary(
    kept: list[ReviewFinding], dropped: list[DroppedFinding], impact: ImpactAnalysis
) -> str:
    if not kept:
        return (
            f"No verified issues. {len(impact.modified_symbols)} modified symbol(s), "
            f"{len(impact.upstream)} upstream caller(s) checked; {len(dropped)} candidate(s) filtered."
        )
    counts = {c: sum(1 for f in kept if f.category == c) for c in Category}
    parts = [f"{n} {c.value.replace('_', ' ')}" for c, n in counts.items() if n]
    top = kept[0]
    verified = sum(f.verified for f in kept)
    return (
        f"{len(kept)} issue(s) found ({', '.join(parts)}); {verified} verified by AST analysis or "
        f"sandbox execution, {len(dropped)} candidate(s) filtered by the critic. Most severe: "
        f"{top.title} at {top.location}. Blast radius: {len(impact.modified_symbols)} modified "
        f"symbol(s), {len(impact.upstream)} upstream caller(s) within 2 hops."
    )


def verdict_for(findings: list[ReviewFinding]) -> str:
    if any(f.severity.rank >= Severity.HIGH.rank for f in findings):
        return "request_changes"
    return "comment" if findings else "approve"


def build_report(**kwargs) -> PRReviewReport:
    return PRReviewReport(**kwargs)
