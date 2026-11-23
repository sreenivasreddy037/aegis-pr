"""HTTP routes for the payments service."""

from __future__ import annotations

import os

from fastapi import APIRouter, Depends, Header, HTTPException

from auth import TokenClaims, TokenError, issue_token, require_scope, verify_token
from db import get_connection, get_user_transactions
from payments import calculate_split_fee, format_amount, settle_split

router = APIRouter()
AUTH_SECRET = os.environ.get("AUTH_SECRET", "")


def _bearer(authorization: str) -> str:
    return authorization.removeprefix("Bearer ").strip()


def get_current_user(authorization: str = Header(...)) -> TokenClaims:
    try:
        return verify_token(_bearer(authorization), AUTH_SECRET)
    except TokenError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc


@router.post("/session/refresh")
def refresh_session(authorization: str = Header(...)) -> dict:
    claims = verify_token(_bearer(authorization), AUTH_SECRET)
    return {"token": issue_token(claims.user_id, claims.scopes, AUTH_SECRET)}


@router.get("/transactions")
def list_transactions(limit: int = 50, user: TokenClaims = Depends(get_current_user)) -> list:
    require_scope(user, "transactions:read")
    conn = get_connection()
    return get_user_transactions(conn, user.user_id, limit=limit)


@router.post("/payments/split")
def split_payment(
    amount_cents: int, payer_ids: list[int], user: TokenClaims = Depends(get_current_user)
) -> dict:
    require_scope(user, "payments:write")
    fee = calculate_split_fee(amount_cents, len(payer_ids))
    transaction_ids = settle_split(get_connection(), payer_ids, amount_cents)
    return {"fee": format_amount(fee), "transactions": transaction_ids}
