"""Attach an already-running local OpenAI-compatible server.

LM Studio, Ollama, vLLM, llama.cpp, SGLang, etc. Deployer does not start
the engine — it probes the URL, records it, and ``connect`` tunnels it to
Cortex Router so a box without a public HTTPS URL can still earn Hane.
"""

from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from . import store
from .runtime import _loopback_url, deploy_from_spec

KNOWN_LOCAL = (
    {
        "id": "lmstudio",
        "label": "LM Studio",
        "engine": "lmstudio",
        "urls": ("http://127.0.0.1:1234/v1",),
    },
    {
        "id": "ollama",
        "label": "Ollama",
        "engine": "ollama",
        "urls": ("http://127.0.0.1:11434/v1",),
    },
    {
        "id": "vllm",
        "label": "vLLM",
        "engine": "vllm",
        "urls": ("http://127.0.0.1:8000/v1",),
    },
    {
        "id": "llamacpp",
        "label": "llama.cpp",
        "engine": "llamacpp",
        "urls": ("http://127.0.0.1:8080/v1",),
    },
    {
        "id": "sglang",
        "label": "SGLang",
        "engine": "sglang",
        "urls": ("http://127.0.0.1:30000/v1",),
    },
    {
        "id": "textgen",
        "label": "text-generation-webui",
        "engine": "external",
        "urls": ("http://127.0.0.1:5000/v1",),
    },
)


def normalize_openai_url(url: str) -> str:
    raw = (url or "").strip()
    if not raw:
        raise ValueError("url required")
    if "://" not in raw:
        raw = "http://" + raw
    raw = raw.rstrip("/")
    if raw.endswith("/v1/models"):
        raw = raw[: -len("/models")]
    elif raw.endswith("/models"):
        raw = raw[: -len("/models")]
    if not raw.endswith("/v1"):
        raw = raw + "/v1"
    return raw


def infer_engine(url: str) -> str:
    lowered = (url or "").lower()
    for row in KNOWN_LOCAL:
        for candidate in row["urls"]:
            if candidate.split("://", 1)[-1] in lowered:
                return str(row["engine"])
    return "external"


def _models_from_payload(payload: Any) -> list[str]:
    if isinstance(payload, dict):
        data = payload.get("data")
        if isinstance(data, list):
            names = []
            for item in data:
                if isinstance(item, dict) and item.get("id"):
                    names.append(str(item["id"]))
                elif isinstance(item, str):
                    names.append(item)
            return names
        models = payload.get("models")
        if isinstance(models, list):
            names = []
            for item in models:
                if isinstance(item, dict):
                    name = item.get("name") or item.get("model") or item.get("id")
                    if name:
                        names.append(str(name))
                elif isinstance(item, str):
                    names.append(item)
            return names
    return []


def probe_openai(url: str, api_key: str = "", timeout: float = 2.0) -> dict[str, Any]:
    """GET {url}/models. Returns ok/status/models/error."""
    base = normalize_openai_url(url)
    target = _loopback_url(base) + "/models"
    headers = {"accept": "application/json"}
    if api_key:
        headers["authorization"] = "Bearer " + api_key
    try:
        req = Request(target, headers=headers, method="GET")
        with urlopen(req, timeout=timeout) as resp:
            status = int(getattr(resp, "status", 200) or 200)
            raw = resp.read()
        try:
            payload = json.loads(raw.decode("utf-8", errors="replace") or "{}")
        except json.JSONDecodeError:
            payload = {}
        models = _models_from_payload(payload)
        return {"ok": 200 <= status < 300, "status": status, "url": base, "models": models}
    except HTTPError as exc:
        return {
            "ok": False,
            "status": int(exc.code),
            "url": base,
            "models": [],
            "error": f"HTTP {exc.code}",
        }
    except (URLError, TimeoutError, OSError, ValueError) as exc:
        return {"ok": False, "status": 0, "url": base, "models": [], "error": str(exc) or exc.__class__.__name__}


def scan_local(timeout: float = 0.8) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in KNOWN_LOCAL:
        for candidate in row["urls"]:
            if candidate in seen:
                continue
            seen.add(candidate)
            probed = probe_openai(candidate, timeout=timeout)
            if not probed.get("ok"):
                continue
            hits.append(
                {
                    "id": row["id"],
                    "label": row["label"],
                    "engine": row["engine"],
                    "url": probed["url"],
                    "models": list(probed.get("models") or []),
                }
            )
    return hits


def attach(
    url: str,
    *,
    model: str = "",
    api_key: str = "",
    engine: str = "",
    name: str = "",
    require_probe: bool = True,
) -> dict[str, Any]:
    """Register an existing local /v1 as a Deployer backend (kind=adopt)."""
    base = normalize_openai_url(url)
    probed = probe_openai(base, api_key=api_key, timeout=3.0)
    models = [str(m) for m in (probed.get("models") or [])]
    model_id = (model or "").strip() or (models[0] if models else "")
    if require_probe and not probed.get("ok"):
        raise ValueError(
            f"nothing answering at {base}/models"
            + (f" ({probed.get('error')})" if probed.get("error") else "")
            + " — start LM Studio / Ollama / vLLM or pass --force"
        )
    if not model_id:
        raise ValueError("no model id on /v1/models — pass --model")
    return deploy_from_spec(
        {
            "kind": "adopt",
            "model_id": model_id,
            "name": name or model_id,
            "base_url": base,
            "engine": engine or infer_engine(base),
            "api_key": api_key,
            "models": models,
        }
    )


def already_attached(url: str) -> dict[str, Any] | None:
    try:
        want = normalize_openai_url(url)
    except ValueError:
        return None
    for row in store.list_backends():
        have = str(row.get("base_url") or "").rstrip("/")
        if have == want or have + "/v1" == want or have == want + "/v1":
            return row
    return None
