#!/usr/bin/env python3
"""CTX-702 smoke: server + UI + adopt + recommend. No GPU required."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cortex_deployer.httpapi import serve_in_thread


def _get(url: str):
    with urllib.request.urlopen(url, timeout=5) as resp:
        return resp.status, resp.read()


def main() -> int:
    tmp = tempfile.TemporaryDirectory()
    os.environ["CORTEX_DEPLOYER_HOME"] = tmp.name
    httpd = serve_in_thread("127.0.0.1", 0)
    port = httpd.server_address[1]
    base = f"http://127.0.0.1:{port}"
    try:
        st, body = _get(base + "/api/health")
        assert st == 200 and b'"ok"' in body, body
        st, host = _get(base + "/api/host")
        assert st == 200, host
        st, html = _get(base + "/")
        assert st == 200 and b"Deploy model" in html, html[:200]
        st, rec = _get(base + "/api/recommend")
        data = json.loads(rec)
        assert "recipes" in data and data["recipes"], data
        files = {r["file"] for r in data["recipes"]}
        assert "qwen38-27b-q3-llamacpp.yaml" in files
        assert "mlx-macos.yaml" in files
        st, dls = _get(base + "/api/downloads")
        assert st == 200 and b"jobs" in dls
        req = urllib.request.Request(
            base + "/api/backends",
            data=json.dumps(
                {"kind": "adopt", "model_id": "smoke", "base_url": "http://127.0.0.1:9/v1"}
            ).encode(),
            method="POST",
            headers={"content-type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            created = json.loads(resp.read())
        assert created["served_name"] == "smoke"
        st, listed = _get(base + "/api/backends")
        rows = json.loads(listed)["backends"]
        assert any(r["served_name"] == "smoke" for r in rows)
        print(f"SMOKE_OK os={json.loads(host).get('os')} port={port} recipes={len(data['recipes'])}")
        return 0
    finally:
        httpd.shutdown()
        httpd.server_close()
        time.sleep(0.05)
        tmp.cleanup()


if __name__ == "__main__":
    sys.exit(main())
