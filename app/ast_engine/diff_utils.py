"""Unified-diff parsing and strict, context-verified patch application.

We implement this ourselves (instead of shelling out to `patch`/`git apply`) so that the
pipeline has no system dependencies and fails loudly on any context mismatch: a review
built on a mis-applied patch is worse than no review at all.
"""

from __future__ import annotations

import re
from pathlib import Path

from app.schemas import DiffHunk, DiffLine, FileDiff

_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@ ?(.*)$")


class PatchError(ValueError):
    """Raised when a diff is malformed or does not apply cleanly to the base tree."""


def _strip_prefix(raw: str) -> str | None:
    path = raw.split("\t", 1)[0].strip()
    if path == "/dev/null":
        return None
    if path.startswith(("a/", "b/")):
        path = path[2:]
    return path


def parse_unified_diff(text: str) -> list[FileDiff]:
    files: list[FileDiff] = []
    current: FileDiff | None = None
    hunk: DiffHunk | None = None
    old_left = new_left = 0
    old_no = new_no = 0
    pending_old: str | bool | None = False  # False = no pending `---` header

    for raw in text.splitlines():
        if hunk is not None and (old_left > 0 or new_left > 0):
            if raw.startswith("\\"):
                continue  # "\ No newline at end of file"
            tag, body = (raw[:1], raw[1:]) if raw else (" ", "")
            if tag == " ":
                hunk.lines.append(
                    DiffLine(kind="context", content=body, old_lineno=old_no, new_lineno=new_no)
                )
                old_no += 1
                new_no += 1
                old_left -= 1
                new_left -= 1
            elif tag == "-":
                hunk.lines.append(DiffLine(kind="remove", content=body, old_lineno=old_no))
                old_no += 1
                old_left -= 1
            elif tag == "+":
                hunk.lines.append(DiffLine(kind="add", content=body, new_lineno=new_no))
                new_no += 1
                new_left -= 1
            else:
                raise PatchError(f"Unexpected line inside hunk: {raw!r}")
            continue

        if raw.startswith("\\"):
            continue
        if raw.startswith("diff --git "):
            current, hunk = None, None
            pending_old = False
        elif raw.startswith("--- "):
            pending_old = _strip_prefix(raw[4:])
            hunk = None
        elif raw.startswith("+++ "):
            if pending_old is False:
                raise PatchError("`+++` header without preceding `---` header")
            current = FileDiff(old_path=pending_old, new_path=_strip_prefix(raw[4:]))  # type: ignore[arg-type]
            files.append(current)
            pending_old = False
        elif raw.startswith("@@"):
            if current is None:
                raise PatchError(f"Hunk header before file header: {raw!r}")
            m = _HUNK_RE.match(raw)
            if not m:
                raise PatchError(f"Malformed hunk header: {raw!r}")
            old_start, old_len = int(m.group(1)), int(m.group(2) or 1)
            new_start, new_len = int(m.group(3)), int(m.group(4) or 1)
            hunk = DiffHunk(
                file_path=current.path,
                old_start=old_start,
                old_len=old_len,
                new_start=new_start,
                new_len=new_len,
                section_header=m.group(5).strip(),
            )
            current.hunks.append(hunk)
            old_left, new_left = old_len, new_len
            old_no, new_no = old_start, new_start
        # Anything else (index lines, mode lines, commit preamble) is ignored.

    if hunk is not None and (old_left > 0 or new_left > 0):
        raise PatchError(f"Truncated hunk in {hunk.file_path}")
    return files


def apply_file_diff(old_text: str, file_diff: FileDiff) -> str:
    old = old_text.splitlines(keepends=True)
    out: list[str] = []
    idx = 0
    for h in file_diff.hunks:
        start = h.old_start - 1 if h.old_len > 0 else h.old_start
        if start < idx or start > len(old):
            raise PatchError(f"{file_diff.path}: hunk @@ -{h.old_start} is out of order/range")
        out.extend(old[idx:start])
        idx = start
        for ln in h.lines:
            if ln.kind == "add":
                out.append(ln.content + "\n")
                continue
            if idx >= len(old) or old[idx].rstrip("\r\n") != ln.content:
                found = old[idx].rstrip("\r\n") if idx < len(old) else "<EOF>"
                raise PatchError(
                    f"{file_diff.path}:{idx + 1}: context mismatch, "
                    f"expected {ln.content!r}, found {found!r}"
                )
            if ln.kind == "context":
                out.append(old[idx])
            idx += 1
    out.extend(old[idx:])
    return "".join(out)


def _safe_join(root: Path, rel: str) -> Path:
    target = (root / rel).resolve()
    if not target.is_relative_to(root.resolve()):
        raise PatchError(f"Refusing to write outside the workspace: {rel}")
    return target


def apply_diff_to_tree(root: Path, files: list[FileDiff]) -> None:
    """Apply parsed diffs in place to the directory tree at `root`."""
    for fd in files:
        if fd.is_deleted:
            target = _safe_join(root, fd.old_path or "")
            if target.exists():
                target.unlink()
            continue
        target = _safe_join(root, fd.new_path or "")
        source = _safe_join(root, fd.old_path) if fd.old_path else None
        old_text = source.read_text() if source and source.exists() else ""
        if fd.old_path and (source is None or not source.exists()):
            raise PatchError(f"Diff modifies {fd.old_path}, which does not exist in the repo")
        new_text = apply_file_diff(old_text, fd)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(new_text)
        if source is not None and source != target and source.exists():
            source.unlink()
