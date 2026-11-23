"""SQLite persistence for transactions."""

from __future__ import annotations

import os
import sqlite3

DB_PATH = os.environ.get("PAYMENTS_DB", "payments.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS transactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    amount_cents INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""


def get_connection(path: str | None = None) -> sqlite3.Connection:
    conn = sqlite3.connect(path or DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def record_transaction(
    conn: sqlite3.Connection, user_id: int, amount_cents: int, status: str = "pending"
) -> int:
    cursor = conn.execute(
        "INSERT INTO transactions (user_id, amount_cents, status) VALUES (?, ?, ?)",
        (user_id, amount_cents, status),
    )
    conn.commit()
    return int(cursor.lastrowid or 0)


def get_user_transactions(conn: sqlite3.Connection, user_id: int, limit: int = 50) -> list[dict]:
    rows = conn.execute(
        "SELECT id, amount_cents, status, created_at FROM transactions "
        "WHERE user_id = ? ORDER BY id DESC LIMIT ?",
        (user_id, limit),
    ).fetchall()
    return [dict(row) for row in rows]
