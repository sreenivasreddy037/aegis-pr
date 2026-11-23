"""Per-review scratch workspace holding pristine base and patched head copies of the repo."""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from app.ast_engine.diff_utils import PatchError, apply_diff_to_tree, parse_unified_diff
from app.ast_engine.impact_graph import ImpactGraph
from app.schemas import FileDiff

_IGNORE = shutil.ignore_patterns(
    ".git", ".venv", "venv", "node_modules", "__pycache__", "*.pyc", ".pytest_cache", ".ruff_cache"
)


@dataclass
class ReviewWorkspace:
    repo: Path
    base_dir: Path
    head_dir: Path
    files: list[FileDiff]
    base_graph: ImpactGraph | None = None
    head_graph: ImpactGraph | None = None
    modified_ranges: dict[str, set[int]] = field(default_factory=dict)

    def file_lines(self, rel: str) -> list[str]:
        path = self.head_dir / rel
        return (
            path.read_text(encoding="utf-8", errors="replace").splitlines() if path.exists() else []
        )


@contextmanager
def prepare_workspace(repo: Path, diff_text: str) -> Iterator[ReviewWorkspace]:
    repo = repo.resolve()
    if not repo.is_dir():
        raise FileNotFoundError(f"Repository not found: {repo}")
    files = parse_unified_diff(diff_text)
    if not files:
        raise PatchError("Diff contains no file changes")
    with tempfile.TemporaryDirectory(prefix="aegis-review-") as tmp:
        base = Path(tmp) / "base"
        head = Path(tmp) / "head"
        shutil.copytree(repo, base, ignore=_IGNORE)
        shutil.copytree(repo, head, ignore=_IGNORE)
        apply_diff_to_tree(head, files)
        yield ReviewWorkspace(repo=repo, base_dir=base, head_dir=head, files=files)
