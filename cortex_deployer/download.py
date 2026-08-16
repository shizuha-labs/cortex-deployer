"""Download Hugging Face weight files into the local models dir."""

from __future__ import annotations

import fnmatch
import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .paths import home_dir

_jobs: dict[str, dict[str, Any]] = {}
_lock = threading.Lock()

_SKIP_NAMES = {".gitattributes", "README.md", "config.json"}
_SKIP_SUFFIXES = {".md", ".json"}


def models_dir() -> Path:
    override = os.environ.get("CORTEX_DEPLOYER_MODELS")
    if override:
        path = Path(override).expanduser()
    else:
        path = home_dir() / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _hf_headers() -> dict[str, str]:
    headers = {"user-agent": "cortex-deployer"}
    token = os.environ.get("HF_TOKEN") or os.environ.get("CORTEX_DEPLOYER_HF_TOKEN")
    if token:
        headers["authorization"] = f"Bearer {token}"
    return headers


def list_local_models() -> list[dict[str, Any]]:
    root = models_dir()
    out: list[dict[str, Any]] = []
    if not root.exists():
        return out
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".gguf", ".safetensors", ".bin"}:
            out.append(
                {
                    "path": str(path),
                    "name": path.name,
                    "bytes": path.stat().st_size,
                }
            )
    return out


def select_weight_files(names: list[str], glob: str = "") -> list[str]:
    """Keep GGUF / shard files; drop READMEs, mmproj unless the glob asks for it."""
    out: list[str] = []
    for name in names:
        base = Path(name).name
        if not name or base in _SKIP_NAMES:
            continue
        if any(base.endswith(suf) for suf in _SKIP_SUFFIXES):
            continue
        if "mmproj" in base.lower() and (not glob or "mmproj" not in glob.lower()):
            continue
        if glob and not (
            fnmatch.fnmatch(name, glob) or fnmatch.fnmatch(base, glob)
        ):
            continue
        if not glob and not base.lower().endswith((".gguf", ".safetensors")):
            continue
        out.append(name)
    return out


def list_hf_files(repo: str, glob: str = "") -> list[str]:
    url = f"https://huggingface.co/api/models/{repo}"
    req = Request(url, headers=_hf_headers())
    with urlopen(req, timeout=30) as resp:
        payload = json.loads(resp.read().decode())
    siblings = payload.get("siblings") or []
    names: list[str] = []
    for item in siblings:
        if not isinstance(item, dict):
            continue
        name = str(item.get("rfilename") or item.get("path") or "")
        if name:
            names.append(name)
    return select_weight_files(names, glob)


def _download_one(url: str, dest: Path, job: dict[str, Any]) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = Request(url, headers=_hf_headers())
    with urlopen(req, timeout=60) as resp, open(tmp, "wb") as handle:
        total = int(resp.headers.get("content-length") or 0)
        job["total"] = total
        done = 0
        while True:
            chunk = resp.read(1024 * 256)
            if not chunk:
                break
            handle.write(chunk)
            done += len(chunk)
            job["bytes"] = done
    tmp.replace(dest)
    job["path"] = str(dest)


def start_download(repo: str, filename: str = "", glob: str = "") -> dict[str, Any]:
    if not repo or "/" not in repo:
        raise ValueError("repo must look like org/name")
    if not filename and not glob:
        raise ValueError("filename or glob is required (refusing to pull an entire repo)")
    job_id = str(uuid.uuid4())
    job: dict[str, Any] = {
        "id": job_id,
        "repo": repo,
        "filename": filename,
        "glob": glob,
        "state": "starting",
        "bytes": 0,
        "total": 0,
        "path": "",
        "error": "",
        "files": [],
    }
    with _lock:
        _jobs[job_id] = job

    def worker() -> None:
        try:
            names = [filename] if filename else list_hf_files(repo, glob)
            names = select_weight_files(names, glob or filename or "")
            if not names:
                raise ValueError("no matching weight files in repo")
            saved: list[str] = []
            for name in names:
                job["filename"] = name
                job["state"] = "downloading"
                url = f"https://huggingface.co/{repo}/resolve/main/{name}?download=true"
                dest = models_dir() / repo.replace("/", "__") / Path(name)
                _download_one(url, dest, job)
                saved.append(str(dest))
            job["files"] = saved
            ggufs = [p for p in saved if p.lower().endswith(".gguf") and "mmproj" not in p.lower()]
            job["path"] = ggufs[0] if ggufs else (saved[-1] if saved else "")
            job["state"] = "done"
        except (HTTPError, URLError, OSError, ValueError) as exc:
            job["state"] = "error"
            job["error"] = str(exc)

    threading.Thread(target=worker, daemon=True).start()
    return job


def list_jobs() -> list[dict[str, Any]]:
    with _lock:
        return list(_jobs.values())


def get_job(job_id: str) -> dict[str, Any] | None:
    with _lock:
        return _jobs.get(job_id)


def wait_job(job_id: str, timeout: float = 3600.0, poll: float = 0.2) -> dict[str, Any]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = get_job(job_id)
        if job is None:
            raise KeyError(job_id)
        if job.get("state") in {"done", "error"}:
            return job
        time.sleep(poll)
    raise TimeoutError(f"download {job_id} still running after {timeout}s")
