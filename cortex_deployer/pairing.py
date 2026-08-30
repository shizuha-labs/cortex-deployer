"""Short-lived, revocable pairing tokens for connecting this local backend
to the Cortex catalog (CTX-733).

A pairing token is scoped to the local deployer instance, expires after a
bounded TTL, and can be revoked at any time. It is what a user pastes into
the Cortex catalog (or the catalog client uses) to register this local
backend — never a caller-supplied destination (the gateway/token for the
outbound connect still comes from server-side config, see ``connect_ctl``).
"""

from __future__ import annotations

import hmac
import secrets
import time
from typing import Any

from . import store

TOKEN_TTL_SECONDS = 15 * 60  # 15 minutes, bounded and short-lived.

_PAIRING_KEY = "pairing"


def generate() -> dict[str, Any]:
    """Create a fresh pairing token, atomically revoking any previous one."""
    raw = secrets.token_urlsafe(32)
    now = int(time.time())
    record = {
        "token": raw,
        "created_at": now,
        "expires_at": now + TOKEN_TTL_SECONDS,
        "revoked": False,
    }
    state = store.load_state()
    state[_PAIRING_KEY] = record
    store.save_state(state)
    return _public(record)


def current() -> dict[str, Any] | None:
    """The active (unrevoked, unexpired) pairing token, or None."""
    record = _record()
    if record is None:
        return None
    return _public(record)


def revoke() -> None:
    """Revoke the current pairing token (idempotent)."""
    state = store.load_state()
    record = state.get(_PAIRING_KEY)
    if record:
        record["revoked"] = True
        store.save_state(state)


def validate(token: str) -> bool:
    """Constant-time check that ``token`` is the current, unrevoked,
    unexpired pairing token."""
    record = _record()
    if record is None:
        return False
    expected = str(record.get("token") or "")
    return hmac.compare_digest(expected, token)


def _record() -> dict[str, Any] | None:
    state = store.load_state()
    record = state.get(_PAIRING_KEY)
    if not record:
        return None
    if record.get("revoked"):
        return None
    if int(time.time()) > int(record.get("expires_at") or 0):
        return None
    return record


def _public(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "token": record["token"],
        "created_at": record["created_at"],
        "expires_at": record["expires_at"],
        "ttl_seconds": TOKEN_TTL_SECONDS,
    }
