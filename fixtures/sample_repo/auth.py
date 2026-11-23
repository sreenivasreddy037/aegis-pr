"""Token issuing and verification for the payments service.

Tokens are compact HMAC-SHA256 signed strings: ``<user_id>:<scopes>:<expires_at>.<sig>``.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

DEFAULT_TTL_SECONDS = 3600


class TokenError(Exception):
    """Raised when a token is malformed, forged or expired."""


@dataclass(frozen=True)
class TokenClaims:
    user_id: int
    expires_at: float
    scopes: list[str] = field(default_factory=list)


def _sign(payload: str, secret: str) -> str:
    return hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()


def issue_token(
    user_id: int, scopes: list[str], secret: str, ttl_seconds: int = DEFAULT_TTL_SECONDS
) -> str:
    expires_at = int(time.time()) + ttl_seconds
    payload = f"{user_id}:{','.join(scopes)}:{expires_at}"
    return f"{payload}.{_sign(payload, secret)}"


def verify_token(token: str, secret: str) -> TokenClaims:
    """Verify an HMAC-signed token and return its claims."""
    payload, _, signature = token.rpartition(".")
    if not payload or not hmac.compare_digest(_sign(payload, secret), signature):
        raise TokenError("invalid signature")
    user_id, scopes, expires_at = payload.split(":")
    if float(expires_at) < time.time():
        raise TokenError("token expired")
    return TokenClaims(
        user_id=int(user_id),
        expires_at=float(expires_at),
        scopes=scopes.split(",") if scopes else [],
    )


def require_scope(claims: TokenClaims, scope: str) -> None:
    if scope not in claims.scopes:
        raise TokenError(f"missing scope: {scope}")
