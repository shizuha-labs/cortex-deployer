"""Wire protocol for the outbound Deployer ↔ Cortex gateway channel.

The Mac (or any NAT host) dials out. Frames are JSON. Bodies travel as
standard base64. Streaming inference uses kind=start / chunk / end;
/models and non-stream JSON use kind=response.
"""

from __future__ import annotations

import base64
from typing import Any


STREAM_KINDS = frozenset({"start", "chunk", "end"})
CLIENT_REPLY_KINDS = STREAM_KINDS | {"response"}


def b64e(data: bytes) -> str:
    return base64.b64encode(data).decode()


def b64d(text: str | None) -> bytes:
    return base64.b64decode(text) if text else b""


def hello_frame(
    model: str,
    aliases: list[str] | None = None,
    *,
    max_model_len: int | None = None,
    engine: str | None = None,
    quant: str | None = None,
    max_concurrent: int | None = None,
) -> dict[str, Any]:
    """First client frame after the WebSocket opens."""
    names: list[str] = []
    for raw in [model, *(aliases or [])]:
        name = str(raw).strip()
        if name and name not in names:
            names.append(name)
    if not names:
        raise ValueError("hello requires a non-empty model id")
    frame: dict[str, Any] = {
        "model": names[0],
        "aliases": names,
    }
    if max_model_len is not None:
        frame["max_model_len"] = int(max_model_len)
    if engine:
        frame["engine"] = str(engine)
    if quant:
        frame["quant"] = str(quant)
    if max_concurrent is not None:
        frame["max_concurrent"] = int(max_concurrent)
    return frame


def proxy_request_frame(
    request_id: str,
    method: str,
    path: str,
    *,
    headers: dict[str, str] | None = None,
    body: bytes | None = None,
) -> dict[str, Any]:
    return {
        "id": request_id,
        "method": method.upper(),
        "path": path,
        "headers": dict(headers or {}),
        "body_b64": b64e(body or b""),
    }


def start_frame(request_id: str, status: int, headers: dict[str, str]) -> dict[str, Any]:
    return {
        "kind": "start",
        "id": request_id,
        "status": int(status),
        "headers": dict(headers),
    }


def chunk_frame(request_id: str, body: bytes) -> dict[str, Any]:
    return {"kind": "chunk", "id": request_id, "body_b64": b64e(body)}


def end_frame(request_id: str) -> dict[str, Any]:
    return {"kind": "end", "id": request_id}


def response_frame(
    request_id: str,
    status: int,
    headers: dict[str, str],
    body: bytes,
) -> dict[str, Any]:
    return {
        "kind": "response",
        "id": request_id,
        "status": int(status),
        "headers": dict(headers),
        "body_b64": b64e(body),
    }


def request_wants_stream(body: bytes | None) -> bool:
    if not body:
        return False
    try:
        import json

        payload = json.loads(body)
    except Exception:
        return False
    return bool(isinstance(payload, dict) and payload.get("stream"))


def models_with_max_model_len(payload: Any) -> tuple[Any, bool]:
    """Copy context_window-style fields onto max_model_len for Cortex health_poll."""
    if not isinstance(payload, dict):
        return payload, False
    changed = False
    for item in payload.get("data") or []:
        if not isinstance(item, dict):
            continue
        ctx = None
        for key in (
            "max_model_len",
            "context_window",
            "max_context_length",
            "context_length",
        ):
            raw = item.get(key)
            if isinstance(raw, int) and raw > 0:
                ctx = raw
                break
        if ctx is None:
            continue
        if item.get("max_model_len") != ctx:
            item["max_model_len"] = ctx
            changed = True
        if item.get("context_window") != ctx:
            item["context_window"] = ctx
            changed = True
    return payload, changed
