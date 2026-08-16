"""Shared hardware×model×quant catalog. Remote Cortex page, bundled fallback."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

DEFAULT_CATALOG_URL = "https://cortex.shizuha.com/deployer/catalog.json"
BUNDLED = Path(__file__).resolve().parent / "data" / "catalog.v1.json"


def bundled_catalog() -> dict[str, Any]:
    return json.loads(BUNDLED.read_text(encoding="utf-8"))


def fetch_catalog(url: str = "", timeout: float = 3.0, *, force: bool = False) -> dict[str, Any]:
    """Prefer the live Cortex catalog; fall back to the package copy."""
    env_url = os.environ.get("CORTEX_DEPLOYER_CATALOG_URL")
    if (url or env_url or "") == "bundled":
        local = bundled_catalog()
        local["fetched"] = False
        local["source"] = "bundled"
        return local
    target = url or env_url or DEFAULT_CATALOG_URL
    if force and target:
        sep = "&" if "?" in target else "?"
        target = f"{target}{sep}t={int(time.time())}"
    try:
        req = Request(
            target,
            headers={
                "user-agent": "cortex-deployer",
                "accept": "application/json",
                "cache-control": "no-cache",
            },
        )
        with urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode())
        if isinstance(payload, dict) and payload.get("schema") == "cortex.deployer.catalog.v1":
            payload.setdefault("source", url or env_url or DEFAULT_CATALOG_URL)
            payload["fetched"] = True
            return payload
    except (URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError):
        pass
    local = bundled_catalog()
    local["fetched"] = False
    local["source"] = str(BUNDLED)
    return local
