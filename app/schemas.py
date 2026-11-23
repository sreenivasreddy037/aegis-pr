"""Pydantic v2 domain models shared by every layer of AegisPR."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, computed_field

# ---------------------------------------------------------------------------------------
# Diff model
# ---------------------------------------------------------------------------------------


class DiffLine(BaseModel):
    kind: Literal["add", "remove", "context"]
    content: str
    old_lineno: int | None = None
    new_lineno: int | None = None


class DiffHunk(BaseModel):
    file_path: str
    old_start: int
    old_len: int
    new_start: int
    new_len: int
    section_header: str = ""
    lines: list[DiffLine] = Field(default_factory=list)

    @property
    def added(self) -> list[DiffLine]:
        return [ln for ln in self.lines if ln.kind == "add"]

    @property
    def removed(self) -> list[DiffLine]:
        return [ln for ln in self.lines if ln.kind == "remove"]

    def contains_new_line(self, lineno: int) -> bool:
        return self.new_start <= lineno < self.new_start + max(self.new_len, 1)


class FileDiff(BaseModel):
    old_path: str | None
    new_path: str | None
    hunks: list[DiffHunk] = Field(default_factory=list)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def path(self) -> str:
        return self.new_path or self.old_path or "<unknown>"

    @property
    def is_new_file(self) -> bool:
        return self.old_path is None

    @property
    def is_deleted(self) -> bool:
        return self.new_path is None

    @property
    def is_python(self) -> bool:
        return self.path.endswith(".py")

    def added_line_numbers(self) -> set[int]:
        return {ln.new_lineno for h in self.hunks for ln in h.added if ln.new_lineno is not None}

    def removed_line_numbers(self) -> set[int]:
        return {ln.old_lineno for h in self.hunks for ln in h.removed if ln.old_lineno is not None}


# ---------------------------------------------------------------------------------------
# AST / impact model
# ---------------------------------------------------------------------------------------


class Param(BaseModel):
    name: str
    kind: Literal["positional", "var_positional", "keyword_only", "var_keyword"] = "positional"
    has_default: bool = False
    default: str | None = None
    annotation: str | None = None

    def render(self) -> str:
        prefix = {"var_positional": "*", "var_keyword": "**"}.get(self.kind, "")
        text = f"{prefix}{self.name}"
        if self.annotation:
            text += f": {self.annotation}"
        if self.has_default:
            text += f" = {self.default}" if self.annotation else f"={self.default}"
        return text


class ImpactNode(BaseModel):
    """A symbol touched by, or reachable from, the change."""

    symbol: str = Field(description="Fully-qualified name, e.g. `auth.verify_token`.")
    kind: Literal["function", "method", "class", "module"]
    file_path: str
    start_line: int
    end_line: int
    direction: Literal["modified", "upstream", "downstream"]
    hop: int = Field(description="0 for modified symbols, N for N-hop callers/callees.")
    via: str | None = Field(
        default=None, description="Symbol this node reaches the change through."
    )
    edge_kind: Literal["calls", "references", "imports"] | None = None
    call_lines: list[int] = Field(default_factory=list)
    snippet: str = ""


class SignatureChange(BaseModel):
    symbol: str
    file_path: str
    line: int
    old_signature: str | None
    new_signature: str | None
    breaking: bool
    reasons: list[str] = Field(default_factory=list)


class ContractViolation(BaseModel):
    """A concrete call site whose arguments no longer satisfy the callee's new signature."""

    callee: str
    caller: str
    file_path: str
    line: int
    call_text: str
    problems: list[str]
    caller_in_diff: bool = False


class PolicySnippet(BaseModel):
    source: str
    title: str
    text: str
    score: float = 0.0
    tags: list[str] = Field(default_factory=list)


class ImpactAnalysis(BaseModel):
    files: list[FileDiff] = Field(default_factory=list)
    modified_symbols: list[ImpactNode] = Field(default_factory=list)
    upstream: list[ImpactNode] = Field(default_factory=list)
    downstream: list[ImpactNode] = Field(default_factory=list)
    signature_changes: list[SignatureChange] = Field(default_factory=list)
    contract_violations: list[ContractViolation] = Field(default_factory=list)
    policies: dict[str, list[PolicySnippet]] = Field(
        default_factory=dict, description="Retrieved policy/ADR context keyed by agent name."
    )
    graph_stats: dict[str, int] = Field(default_factory=dict)


# ---------------------------------------------------------------------------------------
# Sandbox model
# ---------------------------------------------------------------------------------------


class TestCaseOutcome(BaseModel):
    __test__ = False  # stop pytest from collecting this model

    name: str
    status: Literal["passed", "failed", "error", "skipped"]
    message: str = ""


class SandboxResult(BaseModel):
    exit_code: int | None
    timed_out: bool = False
    duration_ms: float
    stdout: str = ""
    stderr: str = ""
    cases: list[TestCaseOutcome] = Field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.timed_out and self.exit_code == 0

    def case(self, suffix: str) -> TestCaseOutcome | None:
        for c in self.cases:
            if c.name.endswith(suffix):
                return c
        return None


class SandboxRun(BaseModel):
    target: str
    test_code: str
    base: SandboxResult | None
    head: SandboxResult
    regressions: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------------------
# Review model
# ---------------------------------------------------------------------------------------


class Category(StrEnum):
    BREAKING_CHANGE = "breaking_change"
    SECURITY = "security"
    REGRESSION = "regression"


class Severity(StrEnum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

    @property
    def rank(self) -> int:
        return {"critical": 4, "high": 3, "medium": 2, "low": 1}[self.value]


class Evidence(BaseModel):
    kind: Literal["diff", "caller", "sandbox", "policy", "static", "llm"]
    detail: str


class ReviewFinding(BaseModel):
    id: str = ""
    agent: str
    category: Category
    severity: Severity
    title: str
    description: str
    file_path: str
    line: int
    symbol: str | None = None
    cwe: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    verified: bool = Field(
        default=False, description="True when proven by execution or deterministic AST analysis."
    )
    evidence: list[Evidence] = Field(default_factory=list)
    suggested_fix: str | None = None
    corroborated_by: list[str] = Field(default_factory=list)

    @property
    def location(self) -> str:
        return f"{self.file_path}:{self.line}"


class DroppedFinding(BaseModel):
    finding: ReviewFinding
    reason: str


class PRReviewReport(BaseModel):
    verdict: Literal["approve", "comment", "request_changes"]
    summary: str
    mode: Literal["mock", "live"]
    findings: list[ReviewFinding] = Field(default_factory=list)
    dropped: list[DroppedFinding] = Field(default_factory=list)
    impact: ImpactAnalysis
    sandbox_runs: list[SandboxRun] = Field(default_factory=list)
    timings_ms: dict[str, float] = Field(default_factory=dict)
    total_latency_ms: float = 0.0
    errors: list[str] = Field(default_factory=list)

    def by_category(self, category: Category) -> list[ReviewFinding]:
        return [f for f in self.findings if f.category == category]
