"""Regenerate `fixtures/sample_repo/sample_pr.diff` and `evals/benchmark_prs.json`.

Each benchmark PR is declared as a list of literal edits against the fixture repo, and the
unified diff is produced with `difflib`. This keeps hunk headers and line numbers exact and
makes the benchmark reproducible:  `uv run python evals/build_benchmark.py`.
"""

from __future__ import annotations

import difflib
import json
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO = ROOT / "fixtures" / "sample_repo"
OUT_JSON = ROOT / "evals" / "benchmark_prs.json"
OUT_DIFF = REPO / "sample_pr.diff"


@dataclass
class Edit:
    file: str
    old: str | None  # None => create file with `new` as content
    new: str


@dataclass
class BenchmarkPR:
    id: str
    title: str
    edits: list[Edit]
    expected: list[dict] = field(default_factory=list)


def render_diff(edits: list[Edit]) -> str:
    originals: dict[str, str | None] = {}
    current: dict[str, str] = {}
    for e in edits:
        if e.file not in current:
            path = REPO / e.file
            originals[e.file] = path.read_text(encoding="utf-8") if path.exists() else None
            current[e.file] = originals[e.file] or ""
        if e.old is None:
            current[e.file] = e.new
        else:
            if current[e.file].count(e.old) != 1:
                raise SystemExit(f"Edit anchor not unique/absent in {e.file}: {e.old[:60]!r}")
            current[e.file] = current[e.file].replace(e.old, e.new)

    chunks: list[str] = []
    for name in sorted(current):
        before = originals[name]
        diff = difflib.unified_diff(
            (before or "").splitlines(keepends=True),
            current[name].splitlines(keepends=True),
            fromfile="/dev/null" if before is None else f"a/{name}",
            tofile=f"b/{name}",
            n=3,
        )
        body = "".join(diff)
        if body:
            chunks.append(f"diff --git a/{name} b/{name}\n{body}")
    return "".join(chunks)


# ---------------------------------------------------------------------------------------
# PR-01: the seeded showcase PR (three senior-level bugs)
# ---------------------------------------------------------------------------------------

SAMPLE_PR = BenchmarkPR(
    id="PR-01",
    title="Audience-bound tokens, status filter for transactions, optional split participants",
    edits=[
        Edit(
            "auth.py",
            "DEFAULT_TTL_SECONDS = 3600\n",
            'DEFAULT_TTL_SECONDS = 3600\nDEFAULT_AUDIENCE = "payments-api"\n',
        ),
        Edit(
            "auth.py",
            """def issue_token(
    user_id: int, scopes: list[str], secret: str, ttl_seconds: int = DEFAULT_TTL_SECONDS
) -> str:
    expires_at = int(time.time()) + ttl_seconds
    payload = f"{user_id}:{','.join(scopes)}:{expires_at}"
""",
            """def issue_token(
    user_id: int,
    scopes: list[str],
    secret: str,
    ttl_seconds: int = DEFAULT_TTL_SECONDS,
    audience: str = DEFAULT_AUDIENCE,
) -> str:
    expires_at = int(time.time()) + ttl_seconds
    payload = f"{user_id}:{','.join(scopes)}:{expires_at}:{audience}"
""",
        ),
        Edit(
            "auth.py",
            '''def verify_token(token: str, secret: str) -> TokenClaims:
    """Verify an HMAC-signed token and return its claims."""
    payload, _, signature = token.rpartition(".")
    if not payload or not hmac.compare_digest(_sign(payload, secret), signature):
        raise TokenError("invalid signature")
    user_id, scopes, expires_at = payload.split(":")
''',
            '''def verify_token(token: str, secret: str, audience: str) -> TokenClaims:
    """Verify an HMAC-signed token, enforcing the audience it was issued for."""
    logger.debug("verifying token for audience %s", audience)
    payload, _, signature = token.rpartition(".")
    if not payload or not hmac.compare_digest(_sign(payload, secret), signature):
        raise TokenError("invalid signature")
    user_id, scopes, expires_at, token_audience = payload.split(":")
    if token_audience != audience:
        raise TokenError("audience mismatch")
''',
        ),
        Edit(
            "api_routes.py",
            'AUTH_SECRET = os.environ.get("AUTH_SECRET", "")\n',
            'AUTH_SECRET = os.environ.get("AUTH_SECRET", "")\nSERVICE_AUDIENCE = "payments-api"\n',
        ),
        Edit(
            "api_routes.py",
            "    claims = verify_token(_bearer(authorization), AUTH_SECRET)\n",
            "    claims = verify_token(_bearer(authorization), AUTH_SECRET, audience=SERVICE_AUDIENCE)\n",
        ),
        Edit(
            "api_routes.py",
            """def list_transactions(limit: int = 50, user: TokenClaims = Depends(get_current_user)) -> list:
    require_scope(user, "transactions:read")
    conn = get_connection()
    return get_user_transactions(conn, user.user_id, limit=limit)
""",
            """def list_transactions(
    limit: int = 50,
    status: str | None = None,
    user: TokenClaims = Depends(get_current_user),
) -> list:
    require_scope(user, "transactions:read")
    conn = get_connection()
    return get_user_transactions(conn, user.user_id, limit=limit, status=status)
""",
        ),
        Edit(
            "db.py",
            """def get_user_transactions(conn: sqlite3.Connection, user_id: int, limit: int = 50) -> list[dict]:
    rows = conn.execute(
        "SELECT id, amount_cents, status, created_at FROM transactions "
        "WHERE user_id = ? ORDER BY id DESC LIMIT ?",
        (user_id, limit),
    ).fetchall()
    return [dict(row) for row in rows]
""",
            """def get_user_transactions(
    conn: sqlite3.Connection, user_id: int, limit: int = 50, status: str | None = None
) -> list[dict]:
    query = (
        "SELECT id, amount_cents, status, created_at FROM transactions "
        f"WHERE user_id = {user_id}"
    )
    if status:
        query += f" AND status = '{status}'"
    query += f" ORDER BY id DESC LIMIT {limit}"
    rows = conn.execute(query).fetchall()
    return [dict(row) for row in rows]
""",
        ),
        Edit(
            "payments.py",
            '''def calculate_split_fee(
    amount_cents: int, participants: int, fee_rate: float = PLATFORM_FEE_RATE
) -> int:
    """Return the per-participant platform fee in cents."""
    if participants <= 0:
        raise ValueError("participants must be a positive integer")
    total_fee = max(round(amount_cents * fee_rate), MIN_FEE_CENTS)
''',
            '''def calculate_split_fee(
    amount_cents: int,
    participants: int | None = None,
    fee_rate: float = PLATFORM_FEE_RATE,
) -> int:
    """Return the per-participant platform fee in cents.

    ``participants`` is optional so single-payer checkouts can reuse this helper.
    """
    total_fee = max(round(amount_cents * fee_rate), MIN_FEE_CENTS)
''',
        ),
    ],
    expected=[
        {"category": "breaking_change", "file": "api_routes.py", "symbol": "get_current_user"},
        {"category": "security", "file": "db.py", "symbol": "get_user_transactions"},
        {"category": "regression", "file": "payments.py", "symbol": "calculate_split_fee"},
    ],
)

BENCHMARK: list[BenchmarkPR] = [
    SAMPLE_PR,
    BenchmarkPR(
        id="PR-02",
        title="Add status lookup helper (SQL injection via % formatting)",
        edits=[
            Edit(
                "db.py",
                "def get_user_transactions(",
                """def find_transactions_by_status(conn: sqlite3.Connection, status: str) -> list[dict]:
    rows = conn.execute(
        "SELECT id, user_id, amount_cents FROM transactions WHERE status = '%s'" % status
    ).fetchall()
    return [dict(row) for row in rows]


def get_user_transactions(""",
            )
        ],
        expected=[
            {"category": "security", "file": "db.py", "symbol": "find_transactions_by_status"}
        ],
    ),
    BenchmarkPR(
        id="PR-03",
        title="Multi-tenant transactions (new required parameter breaks route)",
        edits=[
            Edit(
                "db.py",
                """def get_user_transactions(conn: sqlite3.Connection, user_id: int, limit: int = 50) -> list[dict]:
    rows = conn.execute(
        "SELECT id, amount_cents, status, created_at FROM transactions "
        "WHERE user_id = ? ORDER BY id DESC LIMIT ?",
        (user_id, limit),
    ).fetchall()
""",
                """def get_user_transactions(
    conn: sqlite3.Connection, user_id: int, tenant_id: int, limit: int = 50
) -> list[dict]:
    rows = conn.execute(
        "SELECT id, amount_cents, status, created_at FROM transactions "
        "WHERE user_id = ? AND tenant_id = ? ORDER BY id DESC LIMIT ?",
        (user_id, tenant_id, limit),
    ).fetchall()
""",
            )
        ],
        expected=[
            {"category": "breaking_change", "file": "api_routes.py", "symbol": "list_transactions"}
        ],
    ),
    BenchmarkPR(
        id="PR-04",
        title="Simplify average_ticket (drops zero guard)",
        edits=[
            Edit(
                "payments.py",
                '''    """Average transaction size in cents; 0 when there are no transactions."""
    if count == 0:
        return 0
    return total_cents // count
''',
                '''    """Average transaction size in cents."""
    return total_cents // count
''',
            )
        ],
        expected=[{"category": "regression", "file": "payments.py", "symbol": "average_ticket"}],
    ),
    BenchmarkPR(
        id="PR-05",
        title="Clean refactor: named constant and docstring for format_amount",
        edits=[
            Edit(
                "payments.py",
                """def format_amount(amount_cents: int) -> str:
    return f"${amount_cents / 100:,.2f}"
""",
                '''CENTS_PER_UNIT = 100


def format_amount(amount_cents: int) -> str:
    """Render integer cents as a dollar string, e.g. 123456 -> "$1,234.56"."""
    return f"${amount_cents / CENTS_PER_UNIT:,.2f}"
''',
            )
        ],
        expected=[],
    ),
    BenchmarkPR(
        id="PR-06",
        title="Local dev convenience: fallback signing secret (hardcoded credential)",
        edits=[
            Edit(
                "api_routes.py",
                'AUTH_SECRET = os.environ.get("AUTH_SECRET", "")\n',
                'AUTH_SECRET = os.environ.get("AUTH_SECRET", "sk_live_51HxQe2LkP9vR3tYzWq")\n',
            )
        ],
        expected=[{"category": "security", "file": "api_routes.py"}],
    ),
    BenchmarkPR(
        id="PR-07",
        title="Clean: optional currency for format_amount (backwards compatible)",
        edits=[
            Edit(
                "payments.py",
                """def format_amount(amount_cents: int) -> str:
    return f"${amount_cents / 100:,.2f}"
""",
                """def format_amount(amount_cents: int, currency: str = "USD") -> str:
    symbol = "$" if currency == "USD" else f"{currency} "
    return f"{symbol}{amount_cents / 100:,.2f}"
""",
            )
        ],
        expected=[],
    ),
    BenchmarkPR(
        id="PR-08",
        title="Nightly report export (shell injection)",
        edits=[
            Edit(
                "reports.py",
                None,
                '''"""Nightly export of settled transactions."""

from __future__ import annotations

import subprocess


def export_report(day: str, bucket: str) -> int:
    archive = f"/tmp/{day}.sql.gz"
    cmd = f"sqlite3 payments.db .dump | gzip > {archive} && aws s3 cp {archive} s3://{bucket}/"
    return subprocess.call(cmd, shell=True)
''',
            )
        ],
        expected=[{"category": "security", "file": "reports.py", "symbol": "export_report"}],
    ),
    BenchmarkPR(
        id="PR-09",
        title="Rename record_transaction(status=) to state= (breaks keyword caller)",
        edits=[
            Edit(
                "db.py",
                """    conn: sqlite3.Connection, user_id: int, amount_cents: int, status: str = "pending"
) -> int:
    cursor = conn.execute(
        "INSERT INTO transactions (user_id, amount_cents, status) VALUES (?, ?, ?)",
        (user_id, amount_cents, status),
    )
""",
                """    conn: sqlite3.Connection, user_id: int, amount_cents: int, state: str = "pending"
) -> int:
    cursor = conn.execute(
        "INSERT INTO transactions (user_id, amount_cents, status) VALUES (?, ?, ?)",
        (user_id, amount_cents, state),
    )
""",
            )
        ],
        expected=[{"category": "breaking_change", "file": "payments.py", "symbol": "settle_split"}],
    ),
    BenchmarkPR(
        id="PR-10",
        title="Gross-up discount helper (division by zero at percent=100)",
        edits=[
            Edit(
                "payments.py",
                "def format_amount(",
                '''def apply_discount(amount_cents: int, percent: int) -> int:
    """Amount to charge so that after `percent`% off the merchant still nets `amount_cents`."""
    if percent < 0:
        raise ValueError("percent must be non-negative")
    return amount_cents * 100 // (100 - percent)


def format_amount(''',
            )
        ],
        expected=[{"category": "regression", "file": "payments.py", "symbol": "apply_discount"}],
    ),
    BenchmarkPR(
        id="PR-11",
        title="Clean: structured logging in settle_split",
        edits=[
            Edit(
                "payments.py",
                "import sqlite3\n",
                "import logging\nimport sqlite3\n",
            ),
            Edit(
                "payments.py",
                "MIN_FEE_CENTS = 30\n",
                "MIN_FEE_CENTS = 30\n\nlogger = logging.getLogger(__name__)\n",
            ),
            Edit(
                "payments.py",
                "    fee = calculate_split_fee(amount_cents, len(payer_ids))\n    share",
                '    logger.info("settling split for %d payers", len(payer_ids))\n'
                "    fee = calculate_split_fee(amount_cents, len(payer_ids))\n    share",
            ),
        ],
        expected=[],
    ),
    BenchmarkPR(
        id="PR-12",
        title="Keyword-only scopes for issue_token + dynamic fee rules (eval)",
        edits=[
            Edit(
                "auth.py",
                """def issue_token(
    user_id: int, scopes: list[str], secret: str, ttl_seconds: int = DEFAULT_TTL_SECONDS
) -> str:""",
                """def issue_token(
    user_id: int, secret: str, *, scopes: list[str], ttl_seconds: int = DEFAULT_TTL_SECONDS
) -> str:""",
            ),
            Edit(
                "payments.py",
                "def format_amount(",
                '''def parse_fee_rule(expression: str) -> float:
    """Evaluate a merchant-configured fee rule such as "0.029 + 0.001"."""
    return float(eval(expression))


def format_amount(''',
            ),
        ],
        expected=[
            {"category": "breaking_change", "file": "api_routes.py", "symbol": "refresh_session"},
            {"category": "security", "file": "payments.py", "symbol": "parse_fee_rule"},
        ],
    ),
]


def main() -> None:
    entries = []
    for pr in BENCHMARK:
        entries.append(
            {"id": pr.id, "title": pr.title, "diff": render_diff(pr.edits), "expected": pr.expected}
        )
    OUT_JSON.write_text(
        json.dumps({"repo": "fixtures/sample_repo", "prs": entries}, indent=2) + "\n",
        encoding="utf-8",
    )
    OUT_DIFF.write_text(entries[0]["diff"], encoding="utf-8")
    print(f"wrote {len(entries)} PRs -> {OUT_JSON.relative_to(ROOT)}")
    print(f"wrote sample diff -> {OUT_DIFF.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
