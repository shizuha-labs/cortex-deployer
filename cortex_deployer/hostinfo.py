"""Best-effort host inventory for the local UI (Windows / Linux / macOS)."""

from __future__ import annotations

import os
import platform
import shutil
import socket
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
        # Apple silicon shares unified memory between CPU and GPU. Report the
        # total system memory (bytes -> MB) so the picker can give honest
        # VRAM guidance instead of a bare 0 (CTX-754).
        mem_bytes = _run(["sysctl", "-n", "hw.memsize"]).strip()
        try:
            mem_mb = int(int(mem_bytes) / (1024 * 1024))
        except (TypeError, ValueError):
            mem_mb = 0
        gpus.append(
            {
                "index": 0,
                "name": (out.strip() or "Apple Silicon") + " (Metal)",
                "memory_mb": mem_mb,
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
    try:
        from .enginebin import cached_server

        cached = cached_server()
        if cached:
            return cached
    except Exception:  # noqa: BLE001 — discovery must not raise
        pass
    return None


def find_comfyui() -> str | None:
    env = os.environ.get("CORTEX_DEPLOYER_COMFYUI")
    if env and Path(env).exists():
        return env
    for path in (
        Path.home() / "opt" / "ComfyUI" / "main.py",
        Path("/opt/ComfyUI/main.py"),
    ):
        if path.is_file():
            return str(path.parent)
    return None


def detect_binaries() -> dict[str, str | None]:
    return {
        "llamacpp": find_llama_server(),
        "vllm": shutil.which("vllm"),
        "sglang": shutil.which("python3") or shutil.which("python"),
        "mlx": shutil.which("rapid-mlx"),
        "comfyui": find_comfyui(),
    }


def is_wsl() -> bool:
    if os.environ.get("WSL_DISTRO_NAME"):
        return True
    try:
        text = Path("/proc/sys/kernel/osrelease").read_text(encoding="utf-8").lower()
    except OSError:
        try:
            text = Path("/proc/version").read_text(encoding="utf-8").lower()
        except OSError:
            return False
    return "microsoft" in text or "wsl" in text


def default_bind_host() -> str:
    """Linux/WSL listen on all interfaces so the distro IP is reachable.

    Native Windows stays loopback (the usual local UI). Override with --host.
    """
    if os.name == "nt":
        return "127.0.0.1"
    return "0.0.0.0"


def ipv4_addrs() -> list[str]:
    found: list[str] = []

    def add(ip: str) -> None:
        if not ip or ip.startswith("127.") or ip.startswith("169.254.") or ip in found:
            return
        found.append(ip)

    try:
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        probe.connect(("8.8.8.8", 80))
        add(probe.getsockname()[0])
        probe.close()
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            add(info[4][0])
    except OSError:
        pass
    return found


def advertise_urls(host: str, port: int) -> list[str]:
    port = int(port)
    if host not in {"0.0.0.0", "::", ""}:
        shown = "127.0.0.1" if host in {"::1"} else host
        return [f"http://{shown}:{port}/"]
    urls = [f"http://127.0.0.1:{port}/"]
    for ip in ipv4_addrs():
        urls.append(f"http://{ip}:{port}/")
    return urls


def snapshot() -> dict[str, Any]:
    return {
        "os": platform.system(),
        "os_release": platform.release(),
        "arch": platform.machine(),
        "python": platform.python_version(),
        "hostname": platform.node(),
        "wsl": is_wsl(),
        "ipv4": ipv4_addrs(),
        "gpus": detect_gpus(),
        "binaries": detect_binaries(),
        "home": str(os.path.expanduser("~")),
    }
