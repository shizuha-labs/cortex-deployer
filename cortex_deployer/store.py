"""Durable local backend registry (JSON). No Django."""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .paths import ensure_home, state_path


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_state() -> dict[str, Any]:
    ensure_home()
    path = state_path()
    if not path.exists():
        return {"backends": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"backends": []}
    if not isinstance(data, dict) or not isinstance(data.get("backends"), list):
        return {"backends": []}
    return data


def save_state(state: dict[str, Any]) -> None:
    ensure_home()
    path = state_path()
    payload = json.dumps(state, indent=2, sort_keys=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".state.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.write("\n")
        Path(tmp).replace(path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def new_backend(fields: dict[str, Any]) -> dict[str, Any]:
    backend = {
        "id": str(uuid.uuid4()),
        "created_at": _now(),
        "updated_at": _now(),
        "state": "stopped",
        "healthy": False,
        "pid": None,
        "log_path": "",
        "argv": [],
        "env": {},
        "connect": {},
    }
    backend.update(fields)
    backend["updated_at"] = _now()
    state = load_state()
    state["backends"].append(backend)
    save_state(state)
    return backend


def update_backend(backend_id: str, **fields: Any) -> dict[str, Any] | None:
    state = load_state()
    found = None
    for item in state["backends"]:
        if item.get("id") == backend_id:
            item.update(fields)
            item["updated_at"] = _now()
            found = item
            break
    if found is None:
        return None
    save_state(state)
    return found


def delete_backend(backend_id: str) -> bool:
    state = load_state()
    before = len(state["backends"])
    state["backends"] = [b for b in state["backends"] if b.get("id") != backend_id]
    if len(state["backends"]) == before:
        return False
    save_state(state)
    return True


def get_backend(backend_id: str) -> dict[str, Any] | None:
    for item in load_state()["backends"]:
        if item.get("id") == backend_id:
            return item
    return None


def list_backends() -> list[dict[str, Any]]:
    return list(load_state()["backends"])
