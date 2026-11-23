"""`aegispr` command-line interface (Typer + Rich)."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console, Group
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

from app import __version__
from app.config import FIXTURE_DIFF, FIXTURE_REPO, get_settings, reset_settings_cache
from app.schemas import PRReviewReport, Severity

app = typer.Typer(
    name="aegispr",
    help="AegisPR — multi-agent code review with AST impact graphs and sandbox-verified findings.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()

SEVERITY_STYLE = {
    Severity.CRITICAL: "bold white on red",
    Severity.HIGH: "bold red",
    Severity.MEDIUM: "yellow",
    Severity.LOW: "dim",
}
VERDICT_STYLE = {"approve": "bold green", "comment": "bold yellow", "request_changes": "bold red"}


def _bootstrap() -> None:
    load_dotenv(override=False)
    reset_settings_cache()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")


def render_report(report: PRReviewReport, diff_text: str, show_diff: bool = True) -> None:
    impact = report.impact
    header = Text.assemble(
        ("AegisPR review  ", "bold"),
        (report.verdict.upper().replace("_", " "), VERDICT_STYLE[report.verdict]),
        (f"   mode={report.mode}   {report.total_latency_ms:.0f} ms   ", "dim"),
        (
            f"{len(impact.files)} file(s), {impact.graph_stats.get('symbols', 0)} symbols, "
            f"{impact.graph_stats.get('edges', 0)} graph edges",
            "dim",
        ),
    )
    console.print(Panel(Group(header, Text(report.summary)), border_style="cyan"))

    if show_diff:
        shown = diff_text if len(diff_text) < 6000 else diff_text[:6000] + "\n... (truncated)"
        console.print(
            Panel(
                Syntax(shown, "diff", theme="ansi_dark", word_wrap=True),
                title="Diff",
                border_style="dim",
            )
        )

    tree = Tree("[bold]AST impact graph[/bold] (modified → upstream callers)")
    for m in impact.modified_symbols:
        sig = next((s for s in impact.signature_changes if s.symbol == m.symbol), None)
        label = f"[bold]{m.symbol}[/bold] [dim]{m.file_path}:{m.start_line}[/dim]"
        if sig and sig.breaking:
            label += f"  [red]breaking: {'; '.join(sig.reasons)}[/red]"
        branch = tree.add(label)

        def attach(parent: Tree, via: str) -> None:
            for u in impact.upstream:
                if u.via == via:
                    sub = parent.add(
                        f"[cyan]{u.symbol}[/cyan] [dim]hop {u.hop} · {u.edge_kind} at "
                        f"{u.file_path}:{','.join(map(str, u.call_lines))}[/dim]"
                    )
                    attach(sub, u.symbol)

        attach(branch, m.symbol)
    console.print(tree)

    table = Table(title="Findings", show_lines=True, expand=True)
    for col, kw in (
        ("ID", {"no_wrap": True}),
        ("Severity", {}),
        ("Category", {}),
        ("Location", {"no_wrap": True}),
        ("Finding", {"ratio": 3}),
        ("Agents", {}),
        ("Verified", {"justify": "center"}),
        ("Conf.", {"justify": "right"}),
    ):
        table.add_column(col, **kw)
    for f in report.findings:
        table.add_row(
            f.id,
            Text(f.severity.value, style=SEVERITY_STYLE[f.severity]),
            f.category.value,
            f.location,
            f"[bold]{f.title}[/bold]" + (f"\n[dim]{f.cwe}[/dim]" if f.cwe else ""),
            "\n".join([f.agent, *[f"+{a}" for a in f.corroborated_by]]),
            "[green]✔[/green]" if f.verified else "[yellow]–[/yellow]",
            f"{f.confidence:.2f}",
        )
    if report.findings:
        console.print(table)
    else:
        console.print("[green]No verified findings.[/green]")

    for f in report.findings:
        body = [Text(f.description)]
        if f.evidence:
            body.append(Text("\nEvidence", style="bold"))
            body += [Text(f"  • [{e.kind}] {e.detail}") for e in f.evidence[:7]]
        if f.suggested_fix:
            body.append(Text("\nSuggested fix: ", style="bold green") + Text(f.suggested_fix))
        console.print(
            Panel(
                Group(*body),
                title=f"{f.id} · {f.location}",
                border_style=SEVERITY_STYLE[f.severity].split()[-1],
            )
        )

    if report.dropped:
        dt = Table(title="Filtered by critic", expand=True, title_style="dim")
        dt.add_column("Location", no_wrap=True)
        dt.add_column("Candidate")
        dt.add_column("Reason", ratio=2)
        for d in report.dropped:
            dt.add_row(d.finding.location, f"{d.finding.agent}: {d.finding.title}", d.reason)
        console.print(dt)

    tt = Table(title="Node latency", title_style="dim")
    tt.add_column("Node")
    tt.add_column("ms", justify="right")
    for node, ms in report.timings_ms.items():
        tt.add_row(node, f"{ms:.1f}")
    tt.add_row("[bold]end-to-end[/bold]", f"[bold]{report.total_latency_ms:.1f}[/bold]")
    console.print(tt)
    for err in report.errors:
        console.print(f"[red]agent error:[/red] {err}")


@app.command()
def review(
    repo: Path = typer.Option(None, "--repo", "-r", help="Path to the base repository."),
    diff: Path = typer.Option(None, "--diff", "-d", help="Unified diff file ('-' for stdin)."),
    demo: bool = typer.Option(False, "--demo", help="Review the bundled sample PR."),
    as_json: bool = typer.Option(False, "--json", help="Print the report as JSON."),
    out: Path = typer.Option(None, "--out", "-o", help="Also write the JSON report to a file."),
    no_diff: bool = typer.Option(False, "--no-diff", help="Do not print the diff panel."),
    fail_on: Severity = typer.Option(
        None, "--fail-on", help="Exit 1 if a finding has at least this severity."
    ),
) -> None:
    """Review a diff against a repository."""
    _bootstrap()
    from app.graph.review_graph import run_review

    if demo:
        repo, diff_text = FIXTURE_REPO, FIXTURE_DIFF.read_text()
    else:
        if repo is None or diff is None:
            raise typer.BadParameter("Provide --repo and --diff, or use --demo.")
        diff_text = sys.stdin.read() if str(diff) == "-" else diff.read_text()

    settings = get_settings()
    if not as_json:
        shown = Path(repo).resolve()
        if shown.is_relative_to(Path.cwd()):
            shown = shown.relative_to(Path.cwd())
        console.print(f"[dim]Reviewing {shown} in {settings.mode} mode...[/dim]")
    report = run_review(repo, diff_text, settings=settings)
    payload = report.model_dump_json(indent=2)
    if out:
        out.write_text(payload)
    if as_json:
        print(payload)
    else:
        render_report(report, diff_text, show_diff=not no_diff)
    if fail_on is not None and any(f.severity.rank >= fail_on.rank for f in report.findings):
        raise typer.Exit(code=1)


@app.command()
def benchmark(
    as_json: bool = typer.Option(False, "--json", help="Print metrics as JSON."),
) -> None:
    """Run the 12-PR benchmark and report recall, precision and latency."""
    _bootstrap()
    from app.evaluation import run_benchmark

    result = run_benchmark()
    if as_json:
        print(json.dumps(result.summary(), indent=2))
        return
    table = Table(title=f"AegisPR benchmark ({result.mode} mode, {len(result.cases)} PRs)")
    for col in ("PR", "Expected", "Found", "TP", "FP", "FN", "Latency ms"):
        table.add_column(col, justify="right" if col not in ("PR",) else "left")
    for c in result.cases:
        table.add_row(
            c.pr_id,
            str(len(c.expected)),
            str(c.found),
            str(c.tp),
            str(c.fp),
            str(c.fn),
            f"{c.latency_ms:.0f}",
        )
    console.print(table)
    s = result.summary()
    console.print(
        f"recall [bold]{s['recall']:.2%}[/bold]  precision [bold]{s['precision']:.2%}[/bold]  "
        f"false-positive rate [bold]{s['false_positive_rate']:.2%}[/bold]  "
        f"clean-PR false alarms {s['clean_pr_false_alarms']}/{s['clean_prs']}  "
        f"P50 {s['p50_latency_ms']:.0f} ms  P95 {s['p95_latency_ms']:.0f} ms"
    )


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", help="Bind address."),
    port: int = typer.Option(8000, help="Port."),
) -> None:
    """Run the FastAPI server."""
    _bootstrap()
    import uvicorn

    uvicorn.run("app.api:app", host=host, port=port)


@app.command()
def version() -> None:
    """Print the AegisPR version."""
    console.print(__version__)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
