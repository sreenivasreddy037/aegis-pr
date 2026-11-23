"""Isolated pytest execution for agent-synthesised regression tests.

Isolation layers (defence in depth, not a security boundary on its own; run the service in
the provided container for untrusted input):

1. Fresh throwaway copy of the project tree per run, so tests cannot mutate the workspace.
2. Child process in its own session/process group, killed as a group on timeout.
3. Scrubbed environment: no API keys, cloud credentials or user site-packages; HOME and TMPDIR
   point inside the sandbox.
4. POSIX rlimits: CPU seconds, max file size, and (Linux) address space.
5. A pytest plugin that disables outbound sockets for the duration of the run.
6. Its own `pytest.ini`, so the host project's pytest configuration and conftests are ignored.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

from app.config import get_settings
from app.schemas import SandboxResult, TestCaseOutcome

_IGNORE = shutil.ignore_patterns(".git", ".venv", "venv", "__pycache__", "*.pyc", ".pytest_cache")
_GUARD_MODULE = "aegis_sandbox_guard"
_GUARD_SOURCE = '''"""Injected by AegisPR: block network access inside sandboxed tests."""
import socket


def _blocked(*_args, **_kwargs):
    raise RuntimeError("network access is disabled inside the AegisPR sandbox")


socket.socket.connect = _blocked
socket.socket.connect_ex = _blocked
socket.create_connection = _blocked
'''
_PASSTHROUGH_ENV = ("PATH", "LANG", "LC_ALL", "SYSTEMROOT", "TZ")


def _limits(cpu_seconds: int):
    def apply() -> None:  # runs in the child between fork and exec
        try:
            import resource

            resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
            resource.setrlimit(resource.RLIMIT_FSIZE, (50 * 2**20, 50 * 2**20))
            if sys.platform.startswith("linux"):
                resource.setrlimit(resource.RLIMIT_AS, (2 * 2**30, 2 * 2**30))
        except (ImportError, ValueError, OSError):
            pass

    return apply


def _sandbox_env(root: Path) -> dict[str, str]:
    env = {k: os.environ[k] for k in _PASSTHROUGH_ENV if k in os.environ}
    env.update(
        {
            "HOME": str(root),
            "TMPDIR": str(root / ".tmp"),
            "PYTHONPATH": str(root),
            "PYTHONNOUSERSITE": "1",
            "PYTHONHASHSEED": "0",
            "AEGIS_SANDBOX": "1",
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        }
    )
    return env


def _truncate(text: str, limit: int) -> str:
    return (
        text
        if len(text) <= limit
        else text[:limit] + f"\n... [truncated {len(text) - limit} bytes]"
    )


def parse_junit(path: Path) -> list[TestCaseOutcome]:
    if not path.exists():
        return []
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError:
        return []
    cases = []
    for tc in root.iter("testcase"):
        status, message = "passed", ""
        for tag in ("failure", "error", "skipped"):
            el = tc.find(tag)
            if el is not None:
                status = "failed" if tag == "failure" else tag
                message = (el.get("message") or el.text or "").strip()
                break
        cases.append(TestCaseOutcome(name=tc.get("name", ""), status=status, message=message[:500]))  # type: ignore[arg-type]
    return cases


def run_pytest(
    test_code: str,
    project_dir: Path | None = None,
    *,
    timeout_sec: int | None = None,
    test_filename: str = "test_aegis_generated.py",
) -> SandboxResult:
    """Run `test_code` with pytest against a private copy of `project_dir`."""
    settings = get_settings()
    timeout = timeout_sec or settings.sandbox_timeout_sec
    with tempfile.TemporaryDirectory(prefix="aegis-sandbox-") as tmp:
        root = Path(tmp) / "project"
        if project_dir is not None:
            shutil.copytree(project_dir, root, ignore=_IGNORE)
        else:
            root.mkdir()
        (root / ".tmp").mkdir(exist_ok=True)
        (root / "pytest.ini").write_text("[pytest]\naddopts = -p no:cacheprovider\n")
        (root / f"{_GUARD_MODULE}.py").write_text(_GUARD_SOURCE)
        (root / test_filename).write_text(test_code)
        junit = Path(tmp) / "junit.xml"

        cmd = [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-c",
            str(root / "pytest.ini"),
            "--rootdir",
            str(root),
            "-p",
            _GUARD_MODULE,
            f"--junitxml={junit}",
            "-o",
            "junit_family=xunit2",
            test_filename,
        ]
        start = time.perf_counter()
        proc = subprocess.Popen(
            cmd,
            cwd=root,
            env=_sandbox_env(root),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
            preexec_fn=_limits(timeout + 1) if os.name == "posix" else None,
        )
        timed_out = False
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            killpg = getattr(os, "killpg", None)  # POSIX only; Windows falls back to kill()
            try:
                if killpg is None:
                    raise ProcessLookupError
                killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                proc.kill()
            stdout, stderr = proc.communicate()
        duration_ms = (time.perf_counter() - start) * 1000
        limit = settings.sandbox_max_output_bytes
        return SandboxResult(
            exit_code=None if timed_out else proc.returncode,
            timed_out=timed_out,
            duration_ms=round(duration_ms, 1),
            stdout=_truncate(stdout or "", limit),
            stderr=_truncate(stderr or "", limit),
            cases=[] if timed_out else parse_junit(junit),
        )
