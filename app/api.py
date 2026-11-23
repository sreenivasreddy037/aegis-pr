"""FastAPI service exposing AegisPR reviews over HTTP."""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app import __version__
from app.ast_engine.diff_utils import PatchError
from app.config import FIXTURE_DIFF, FIXTURE_REPO, get_settings
from app.graph.review_graph import run_review
from app.schemas import PRReviewReport

load_dotenv(override=False)

WEB_DIR = Path(__file__).resolve().parent / "web"

app = FastAPI(
    title="AegisPR",
    version=__version__,
    description="Autonomous multi-agent code review with AST impact graphs and sandbox-verified findings.",
)
app.mount("/static", StaticFiles(directory=WEB_DIR / "static"), name="static")


class DiffReviewRequest(BaseModel):
    diff: str = Field(min_length=1, description="Unified diff (git diff format).")
    repo_path: str | None = Field(
        default=None,
        description="Repository to review against, relative to ALLOWED_REPO_ROOT. "
        "Defaults to the bundled sample repo.",
    )


def _resolve_repo(repo_path: str | None) -> Path:
    if repo_path is None:
        return FIXTURE_REPO
    root = get_settings().allowed_repo_root.resolve()
    candidate = (root / repo_path).resolve()
    if not candidate.is_relative_to(root) or not candidate.is_dir():
        raise HTTPException(
            status_code=400, detail="repo_path must be a directory inside ALLOWED_REPO_ROOT"
        )
    return candidate


class SampleDiff(BaseModel):
    repo_path: str
    title: str
    diff: str


@app.get("/", include_in_schema=False)
def web_ui() -> FileResponse:
    """Browser interface: paste a diff, run a review, explore the impact graph."""
    return FileResponse(WEB_DIR / "index.html")


@app.get("/repos")
def list_repos() -> list[str]:
    """Repositories the API may review: directories directly under ALLOWED_REPO_ROOT."""
    root = get_settings().allowed_repo_root.resolve()
    if not root.is_dir():
        return []
    return sorted(
        p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith((".", "_"))
    )


@app.get("/sample", response_model=SampleDiff)
def sample_diff() -> SampleDiff:
    """The bundled sample PR (three seeded bugs), used by the UI's "Load sample PR" button."""
    return SampleDiff(
        repo_path=FIXTURE_REPO.name,
        title="Audience-bound tokens, status filter, optional split participants",
        diff=FIXTURE_DIFF.read_text(encoding="utf-8"),
    )


@app.get("/health")
def health() -> dict:
    settings = get_settings()
    return {
        "status": "ok",
        "version": __version__,
        "mode": settings.mode,
        "qdrant_mode": settings.qdrant_mode,
    }


@app.post("/review/diff", response_model=PRReviewReport)
def review_diff(request: DiffReviewRequest) -> PRReviewReport:
    # Sync endpoint: FastAPI runs it in a worker thread so the event loop stays responsive.
    settings = get_settings()
    if len(request.diff.encode()) > settings.max_diff_bytes:
        raise HTTPException(status_code=413, detail="diff too large")
    repo = _resolve_repo(request.repo_path)
    try:
        return run_review(repo, request.diff, settings=settings)
    except PatchError as exc:
        raise HTTPException(status_code=422, detail=f"diff does not apply: {exc}") from exc


@app.get("/review/demo", response_model=PRReviewReport)
def review_demo() -> PRReviewReport:
    return run_review(
        FIXTURE_REPO, FIXTURE_DIFF.read_text(encoding="utf-8"), settings=get_settings()
    )
