"""FastAPI service exposing AegisPR reviews over HTTP."""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app import __version__
from app.ast_engine.diff_utils import PatchError
from app.config import FIXTURE_DIFF, FIXTURE_REPO, get_settings
from app.graph.review_graph import run_review
from app.schemas import PRReviewReport

load_dotenv(override=False)

app = FastAPI(
    title="AegisPR",
    version=__version__,
    description="Autonomous multi-agent code review with AST impact graphs and sandbox-verified findings.",
)


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
    return run_review(FIXTURE_REPO, FIXTURE_DIFF.read_text(), settings=get_settings())
