"""One-click: pick a fitting recipe, install the engine, pull weights, start."""

from __future__ import annotations

import threading
import time
import uuid
from typing import Any

from . import download, enginebin, recommend, runtime
from .recipes import list_examples, load_recipe

_jobs: dict[str, dict[str, Any]] = {}
_lock = threading.Lock()


def list_jobs() -> list[dict[str, Any]]:
    with _lock:
        return list(_jobs.values())


def get_job(job_id: str) -> dict[str, Any] | None:
    with _lock:
        return _jobs.get(job_id)


def _local_weight(glob: str) -> str:
    needle = glob.replace("*", "").lower()
    for row in download.list_local_models():
        name = str(row.get("name") or "").lower()
        path = str(row.get("path") or "")
        if needle and needle in name:
            return path
        if path.lower().endswith(".gguf") and not glob:
            return path
    return ""


def start_setup(recipe_file: str = "") -> dict[str, Any]:
    recs = recommend.recommend()
    chosen = recipe_file or recs.get("best") or "qwen38-27b-q3-llamacpp.yaml"
    match = next((p for p in list_examples() if p.name == chosen), None)
    if match is None:
        raise ValueError(f"unknown recipe {chosen}")
    recipe = load_recipe(match)
    job_id = str(uuid.uuid4())
    job: dict[str, Any] = {
        "id": job_id,
        "state": "starting",
        "step": "engine",
        "recipe": chosen,
        "served_name": recipe.model.served_name,
        "error": "",
        "binary": "",
        "weights": "",
        "backend_id": "",
        "base_url": "",
    }
    with _lock:
        _jobs[job_id] = job

    def worker() -> None:
        try:
            job["step"] = "engine"
            job["state"] = "running"
            if recipe.engine == "llamacpp":
                info = enginebin.ensure_llama_server()
                job["binary"] = info["binary"]
            job["step"] = "weights"
            weights = _local_weight(recipe.download_glob)
            if recipe.engine == "llamacpp" and not weights:
                if not recipe.model.repo:
                    raise ValueError("recipe has no download repo and no local GGUF")
                dl = download.start_download(
                    recipe.model.repo,
                    filename=recipe.model.filename or "",
                    glob=recipe.download_glob or "",
                )
                done = download.wait_job(dl["id"])
                if done.get("state") != "done":
                    raise ValueError(done.get("error") or "weight download failed")
                weights = str(done.get("path") or "")
            if recipe.engine == "llamacpp" and not weights:
                raise ValueError("no GGUF after download")
            job["weights"] = weights
            job["step"] = "start"
            backend = runtime.deploy_from_spec(
                {
                    "kind": "recipe",
                    "recipe": chosen,
                    "model_path": weights,
                    "served_name": recipe.model.served_name,
                    "context_length": recipe.launch.context_length,
                    "autostart": True,
                }
            )
            backend = runtime.wait_started(backend["id"], timeout=8.0)
            job["backend_id"] = backend.get("id") or ""
            job["base_url"] = backend.get("base_url") or ""
            job["state"] = "done"
            job["step"] = "done"
        except Exception as exc:  # noqa: BLE001 — surface to UI
            job["state"] = "error"
            job["error"] = str(exc)

    threading.Thread(target=worker, daemon=True).start()
    return job


def wait_job(job_id: str, timeout: float = 7200.0, poll: float = 0.4) -> dict[str, Any]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = get_job(job_id)
        if job is None:
            raise KeyError(job_id)
        if job.get("state") in {"done", "error"}:
            return job
        time.sleep(poll)
    raise TimeoutError(f"setup {job_id} still running after {timeout}s")
