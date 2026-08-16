"""Start/stop outbound Cortex connect for a managed backend."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from typing import Any

from . import store
from .paths import logs_dir


def start_connect(backend_id: str, gateway: str, token: str, model: str = "") -> dict[str, Any]:
    backend = store.get_backend(backend_id)
    if backend is None:
        raise KeyError(backend_id)
    if not gateway or not token:
        raise ValueError("gateway and token are required")
    model = model or str(backend.get("served_name") or backend.get("name") or "")
    upstream = str(backend.get("base_url") or "")
    if not upstream:
        raise ValueError("backend has no base_url")
    log_path = logs_dir() / f"{backend_id}.connect.log"
    log_f = open(log_path, "ab", buffering=0)
    argv = [
        sys.executable,
        "-m",
        "cortex_deployer",
        "connect",
        "--gateway",
        gateway,
        "--token",
        token,
        "--model",
        model,
        "--upstream",
        upstream,
    ]
    kw: dict[str, Any] = {
        "args": argv,
        "stdout": log_f,
        "stderr": subprocess.STDOUT,
        "stdin": subprocess.DEVNULL,
    }
    if os.name == "nt":
        kw["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    else:
        kw["start_new_session"] = True
    proc = subprocess.Popen(**kw)
    updated = store.update_backend(
        backend_id,
        connect={
            "gateway": gateway,
            "model": model,
            "pid": proc.pid,
            "log_path": str(log_path),
            "state": "connected",
        },
    )
    return updated or backend


def stop_connect(backend_id: str) -> dict[str, Any]:
    backend = store.get_backend(backend_id)
    if backend is None:
        raise KeyError(backend_id)
    info = dict(backend.get("connect") or {})
    pid = info.get("pid")
    if pid:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                capture_output=True,
                check=False,
            )
        else:
            try:
                os.kill(int(pid), signal.SIGTERM)
            except OSError:
                pass
    info["pid"] = None
    info["state"] = "stopped"
    updated = store.update_backend(backend_id, connect=info)
    return updated or backend
