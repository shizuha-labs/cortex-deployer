"""Best-effort host inventory for the local UI (Windows / Linux / macOS)."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from typing import Any


def _run(cmd: list[str], timeout: float = 3.0) -> str:
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return (proc.stdout or "") + (proc.stderr or "")


def detect_gpus() -> list[dict[str, Any]]:
    gpus: list[dict[str, Any]] = []
    smi = shutil.which("nvidia-smi")
    if smi:
        out = _run(
            [smi, "--query-gpu=index,name,memory.total,driver_version", "--format=csv,noheader,nounits"]
        )
        for line in out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3 and parts[0].isdigit():
                try:
                    mem = int(float(parts[2]))
                except ValueError:
                    mem = 0
                gpus.append(
                    {
                        "index": int(parts[0]),
                        "name": parts[1],
                        "memory_mb": mem,
                        "driver": parts[3] if len(parts) > 3 else "",
                        "vendor": "nvidia",
                    }
                )
    if platform.system() == "Darwin":
        out = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        gpus.append(
            {
                "index": 0,
                "name": (out.strip() or "Apple Silicon") + " (Metal)",
                "memory_mb": 0,
                "driver": "metal",
                "vendor": "apple",
            }
        )
    return gpus


def detect_binaries() -> dict[str, str | None]:
    names = {
        "llamacpp": ["llama-server", "llama-server.exe"],
        "vllm": ["vllm"],
        "sglang": [],
        "mlx": ["rapid-mlx"],
    }
    found: dict[str, str | None] = {}
    for engine, candidates in names.items():
        path = None
        for cand in candidates:
            path = shutil.which(cand)
            if path:
                break
        found[engine] = path
    return found


def snapshot() -> dict[str, Any]:
    return {
        "os": platform.system(),
        "os_release": platform.release(),
        "arch": platform.machine(),
        "python": platform.python_version(),
        "hostname": platform.node(),
        "gpus": detect_gpus(),
        "binaries": detect_binaries(),
        "home": str(os.path.expanduser("~")),
    }
