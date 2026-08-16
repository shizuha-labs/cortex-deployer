"""Best-effort host inventory for the local UI (Windows / Linux / macOS)."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from pathlib import Path
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


def find_llama_server() -> str | None:
    env = os.environ.get("CORTEX_DEPLOYER_LLAMA_SERVER")
    if env and Path(env).exists():
        return env
    for name in ("llama-server", "llama-server.exe"):
        found = shutil.which(name)
        if found:
            return found
    extras: list[Path] = []
    home = Path.home()
    extras.extend(
        [
            home / "llama.cpp" / "llama-server",
            home / "llama.cpp" / "llama-server.exe",
            home / "llama.cpp" / "build" / "bin" / "llama-server",
            home / "bin" / "llama-server",
            home / "bin" / "llama-server.exe",
        ]
    )
    if os.name == "nt":
        local = Path(os.environ.get("LOCALAPPDATA") or (home / "AppData" / "Local"))
        pf = Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
        extras.extend(
            [
                local / "llama.cpp" / "llama-server.exe",
                local / "Programs" / "llama.cpp" / "llama-server.exe",
                home / "scoop" / "shims" / "llama-server.exe",
                home / "scoop" / "apps" / "llama.cpp" / "current" / "llama-server.exe",
                pf / "llama.cpp" / "llama-server.exe",
                Path(r"C:\tools\llama.cpp\llama-server.exe"),
                Path(r"C:\llama.cpp\llama-server.exe"),
            ]
        )
    else:
        extras.extend(
            [
                Path("/usr/local/bin/llama-server"),
                Path("/opt/homebrew/bin/llama-server"),
                Path("/usr/bin/llama-server"),
            ]
        )
    for path in extras:
        if path.is_file():
            return str(path)
    return None


def detect_binaries() -> dict[str, str | None]:
    return {
        "llamacpp": find_llama_server(),
        "vllm": shutil.which("vllm"),
        "sglang": shutil.which("python3") or shutil.which("python"),
        "mlx": shutil.which("rapid-mlx"),
    }


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
