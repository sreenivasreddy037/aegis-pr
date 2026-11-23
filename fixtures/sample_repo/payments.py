"""Fee calculation and settlement for split payments."""

from __future__ import annotations

import sqlite3

from db import record_transaction

PLATFORM_FEE_RATE = 0.029
MIN_FEE_CENTS = 30


def calculate_split_fee(
    amount_cents: int, participants: int, fee_rate: float = PLATFORM_FEE_RATE
) -> int:
    """Return the per-participant platform fee in cents."""
    if participants <= 0:
        raise ValueError("participants must be a positive integer")
    total_fee = max(round(amount_cents * fee_rate), MIN_FEE_CENTS)
    return total_fee // participants


def average_ticket(total_cents: int, count: int) -> int:
    """Average transaction size in cents; 0 when there are no transactions."""
    if count == 0:
        return 0
    return total_cents // count


def settle_split(conn: sqlite3.Connection, payer_ids: list[int], amount_cents: int) -> list[int]:
    """Charge every payer their share plus the platform fee and persist the transactions."""
    if not payer_ids:
        raise ValueError("at least one payer is required")
    fee = calculate_split_fee(amount_cents, len(payer_ids))
    share = amount_cents // len(payer_ids) + fee
    return [record_transaction(conn, payer, share, status="settled") for payer in payer_ids]


def format_amount(amount_cents: int) -> str:
    return f"${amount_cents / 100:,.2f}"
