"""Install official llama.cpp release binaries (Windows CUDA first)."""

from __future__ import annotations

import json
import os
import platform
import tarfile
import zipfile
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from .hostinfo import find_llama_server
from .paths import engines_dir

LLAMA_RELEASES = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"

# Prefer CUDA 13 on Windows x64 — Blackwell (RTX 5080) needs it. 12.4 is fallback.
_WIN_X64_CUDA = (
    "win-cuda-13.3-x64",
    "win-cuda-13.4-x64",
    "win-cuda-13",
    "win-cuda-12.4-x64",
    "win-cuda-12",
)
_WIN_ARM_CUDA = (
    "win-cuda-13.4-arm64",
    "win-cuda-13-arm64",
)


def _is_llama_bin(name: str) -> bool:
    base = name.lower()
    return "llama-" in base and "-bin-" in base and not base.startswith("cudart")


def _is_arm(machine: str) -> bool:
    return machine.lower() in {"arm64", "aarch64"}


def pick_release_assets(
    names: list[str],
    *,
    system: str,
    machine: str,
) -> tuple[str | None, str | None]:
    """Return (llama_zip, optional_cudart_zip) from a GitHub release asset list."""
    arm = _is_arm(machine)
    if system == "Windows":
        tags = _WIN_ARM_CUDA if arm else _WIN_X64_CUDA
        for tag in tags:
            llama = next((n for n in names if _is_llama_bin(n) and tag in n.lower()), None)
            if not llama:
                continue
            cudart = next(
                (
                    n
                    for n in names
                    if n.lower().startswith("cudart-")
                    and tag.replace("win-", "") in n.lower()
                ),
                None,
            )
            if cudart is None:
                # Match just the cuda-X.Y token when tag is a prefix like win-cuda-13
                token = tag.replace("win-", "")
                cudart = next(
                    (
                        n
                        for n in names
                        if n.lower().startswith("cudart-")
                        and token in n.lower()
                        and (("arm64" in n.lower()) == arm)
                    ),
                    None,
                )
            return llama, cudart
        cpu = "win-cpu-arm64" if arm else "win-cpu-x64"
        llama = next((n for n in names if _is_llama_bin(n) and cpu in n.lower()), None)
        return llama, None
    if system == "Darwin":
        tag = "macos-arm64" if arm else "macos-x64"
        llama = next((n for n in names if _is_llama_bin(n) and tag in n.lower()), None)
        return llama, None
    # Official Linux CUDA zips are not published; CPU/Vulkan only.
    tag = "ubuntu-arm64" if arm else "ubuntu-x64"
    llama = next(
        (
            n
            for n in names
            if _is_llama_bin(n) and tag in n.lower() and "sycl" not in n.lower()
            and "vulkan" not in n.lower() and "openvino" not in n.lower()
        ),
        None,
    )
    return llama, None


def _http_json(url: str) -> dict[str, Any]:
    req = Request(url, headers={"user-agent": "cortex-deployer", "accept": "application/json"})
    with urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = Request(url, headers={"user-agent": "cortex-deployer"})
    with urlopen(req, timeout=60) as resp, open(tmp, "wb") as handle:
        while True:
            chunk = resp.read(1024 * 256)
            if not chunk:
                break
            handle.write(chunk)
    tmp.replace(dest)


def _extract(archive: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    name = archive.name.lower()
    if name.endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest)
        return
    if name.endswith(".tar.gz") or name.endswith(".tgz"):
        with tarfile.open(archive, "r:gz") as tf:
            tf.extractall(dest)
        return
    raise ValueError(f"unsupported archive {archive.name}")


def find_extracted_server(root: Path) -> Path | None:
    names = {"llama-server", "llama-server.exe"}
    hits = [p for p in root.rglob("*") if p.is_file() and p.name in names]
    if not hits:
        return None
    hits.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return hits[0]


def cached_server() -> str | None:
    root = engines_dir()
    if not root.exists():
        return None
    found = find_extracted_server(root)
    return str(found) if found else None


def ensure_llama_server(*, force: bool = False) -> dict[str, Any]:
    """Return a usable llama-server path, downloading an official build if needed."""
    existing = None if force else (find_llama_server() or cached_server())
    if existing and Path(existing).is_file():
        return {"binary": existing, "installed": False, "source": "local"}

    release = _http_json(LLAMA_RELEASES)
    assets = release.get("assets") or []
    names = [str(a.get("name") or "") for a in assets if isinstance(a, dict)]
    urls = {
        str(a.get("name") or ""): str(a.get("browser_download_url") or "")
        for a in assets
        if isinstance(a, dict)
    }
    llama_name, cudart_name = pick_release_assets(
        names,
        system=platform.system(),
        machine=platform.machine(),
    )
    if not llama_name or not urls.get(llama_name):
        raise ValueError(
            "no official llama.cpp binary for this OS/arch; "
            "install llama-server and set CORTEX_DEPLOYER_LLAMA_SERVER"
        )
    tag = str(release.get("tag_name") or "latest")
    dest = engines_dir() / tag
    dest.mkdir(parents=True, exist_ok=True)
    llama_zip = dest / llama_name
    _download(urls[llama_name], llama_zip)
    _extract(llama_zip, dest)
    if cudart_name and urls.get(cudart_name):
        crt = dest / cudart_name
        _download(urls[cudart_name], crt)
        _extract(crt, dest)
    binary = find_extracted_server(dest)
    if binary is None:
        raise ValueError(f"archive {llama_name} had no llama-server")
    marker = dest / "installed.json"
    marker.write_text(
        json.dumps({"tag": tag, "binary": str(binary), "asset": llama_name}, indent=2),
        encoding="utf-8",
    )
    os.environ.setdefault("CORTEX_DEPLOYER_LLAMA_SERVER", str(binary))
    return {
        "binary": str(binary),
        "installed": True,
        "source": "github",
        "tag": tag,
        "asset": llama_name,
        "cudart": cudart_name or "",
    }
