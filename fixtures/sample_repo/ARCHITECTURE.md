# Payments Service — Architecture & Decision Records

## Overview

`api_routes.py` exposes the HTTP surface. Every authenticated route resolves the caller
through `get_current_user`, which delegates to `auth.verify_token`. Persistence lives in
`db.py` (SQLite), and money movement in `payments.py`.

## ADR-001: Parameterised SQL only

All SQL executed through `db.py` MUST use DB-API placeholders (`?`). Building queries with
f-strings, `%` formatting, `.format()` or string concatenation of user-controlled values is
forbidden, including for optional filters such as `status` or sort order.

## ADR-002: Stable auth contract

`auth.verify_token(token, secret)` is called by every route dependency. Any change to its
signature must remain backwards compatible (new parameters need defaults) or update every
caller in the same change. Audience checks must default to the service audience.

## ADR-003: Money math is integer cents and total

Fee helpers in `payments.py` operate on integer cents and must be total over their documented
domain: invalid inputs (zero or missing participants, negative amounts) raise `ValueError`
explicitly instead of surfacing `ZeroDivisionError` or `TypeError` from arithmetic.

## ADR-004: Secrets come from the environment

Signing keys, database credentials and API tokens are read from environment variables.
Literal secrets in source code are rejected in review.

## ADR-005: No sensitive data in logs

Tokens, signatures and secrets must never be written to logs, even at DEBUG level.
