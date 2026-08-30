"""Spawn / stop / probe local inference processes on Windows, Linux, and macOS."""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

from .engines import render_process
from .paths import logs_dir
from .ports import (
    argv_port,
    bind_error_in_log,
    pick_free_port,
    set_argv_port,
    summarize_crash,
)
from .recipes import list_examples, load_recipe
from .spec import recipe_from_dict
from . import store


def _load_recipe_arg(spec: dict[str, Any]):
    recipe_dict = spec.get("recipe")
    if isinstance(recipe_dict, str):
        match = next((p for p in list_examples() if p.name == recipe_dict), None)
        if match is None:
            raise ValueError(f"unknown recipe {recipe_dict!r}")
        return load_recipe(match)
    if isinstance(recipe_dict, dict):
        return recipe_from_dict(recipe_dict)
    raise ValueError("deploy requires kind=adopt or a recipe object / bundled filename")


def _pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    if os.name == "nt":
        try:
            proc = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}"],
                capture_output=True,
                text=True,
                timeout=3,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return False
        return str(pid) in (proc.stdout or "")
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _loopback_url(url: str) -> str:
    # Binding 0.0.0.0 is correct for listen, but clients must hit loopback.
    return url.replace("://0.0.0.0:", "://127.0.0.1:").replace("://[::]:", "://[::1]:")


def _probe(url: str, timeout: float = 2.0, api_key: str = "") -> bool:
    try:
        headers = {"accept": "application/json"}
        if api_key:
            headers["authorization"] = "Bearer " + api_key
        req = Request(
            _loopback_url(url).rstrip("/") + "/models",
            headers=headers,
            method="GET",
        )
        with urlopen(req, timeout=timeout) as resp:
            return 200 <= getattr(resp, "status", 200) < 300
    except (URLError, TimeoutError, OSError, ValueError):
        return False


def reconcile() -> list[dict[str, Any]]:
    """Mark dead PIDs stopped; refresh healthy from /v1/models."""
    out: list[dict[str, Any]] = []
    for backend in store.list_backends():
        pid = backend.get("pid")
        base = backend.get("base_url") or ""
        if backend.get("state") == "running" and not _pid_alive(pid) and backend.get("kind") != "adopt":
            backend = store.update_backend(backend["id"], state="stopped", pid=None, healthy=False) or backend
        elif base:
            healthy = _probe(base, api_key=str(backend.get("api_key") or ""))
            if healthy != bool(backend.get("healthy")):
                backend = store.update_backend(backend["id"], healthy=healthy) or backend
            else:
                backend["healthy"] = healthy
        out.append(backend)
    return out


def pick_port(preferred: int | None = None, host: str = "127.0.0.1") -> int:
    """Preferred port if it binds, else the next free unprivileged port.

    Recipes default to 8080. On Windows that port is often excluded (Hyper-V)
    or taken — llama-server then exits with "couldn't bind HTTP server socket".
    """
    return pick_free_port(int(preferred or 0), host=host)


def resolve_binary(engine: str, override: str = "") -> str:
    # Recipe argv always starts with a generic name ("llama-server"). That is a
    # hint, not a resolved path — only use it when the file actually exists
    # or is on PATH (so mlx_lm.server wins over a stale CORTEX_DEPLOYER_MLX).
    if override:
        if os.path.isfile(override):
            return override
        from .engines.mlx import is_mlx_lm

        if is_mlx_lm(override):
            return shutil.which(override) or override
    env_key = {
        "llamacpp": "CORTEX_DEPLOYER_LLAMA_SERVER",
        "vllm": "CORTEX_DEPLOYER_VLLM",
        "sglang": "CORTEX_DEPLOYER_SGLANG",
        "mlx": "CORTEX_DEPLOYER_MLX",
        "comfyui": "CORTEX_DEPLOYER_PYTHON",
    }.get(engine)
    if env_key and os.environ.get(env_key):
        return os.environ[env_key]
    if engine == "comfyui":
        if override:
            found = shutil.which(override) if not os.path.isfile(override) else override
            if found:
                return found
        return sys.executable
    if engine == "llamacpp":
        from .hostinfo import find_llama_server

        found = find_llama_server()
        if found:
            return found
        return "llama-server.exe" if os.name == "nt" else "llama-server"
    candidates = {
        "vllm": ["python3", "python"],
        "sglang": ["python3", "python"],
        "mlx": ["rapid-mlx"],
    }.get(engine, [])
    for name in candidates:
        found = shutil.which(name)
        if found:
            return found
    return candidates[0] if candidates else engine


def _spawn(argv: list[str], env: dict[str, str], log_path: Path) -> subprocess.Popen[Any]:
    log_f = open(log_path, "ab", buffering=0)
    popen_kw: dict[str, Any] = {
        "args": argv,
        "stdout": log_f,
        "stderr": subprocess.STDOUT,
        "stdin": subprocess.DEVNULL,
        "env": env,
    }
    if os.name == "nt":
        popen_kw["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    else:
        popen_kw["start_new_session"] = True
    try:
        return subprocess.Popen(**popen_kw)
    except OSError as exc:
        log_f.close()
        raise ValueError(f"failed to spawn {argv[0]}: {exc}") from exc


def start_backend(backend_id: str) -> dict[str, Any]:
    backend = store.get_backend(backend_id)
    if backend is None:
        raise KeyError(backend_id)
    if backend.get("kind") == "adopt":
        healthy = _probe(backend.get("base_url") or "", api_key=str(backend.get("api_key") or ""))
        updated = store.update_backend(
            backend_id,
            state="running" if healthy else "stopped",
            healthy=healthy,
        )
        return updated or backend
    argv = list(backend.get("argv") or [])
    if not argv:
        raise ValueError("backend has no argv to start")
    argv[0] = resolve_binary(backend.get("engine") or "llamacpp", argv[0])
    host = str(backend.get("host") or "127.0.0.1")
    preferred = argv_port(argv) or int(backend.get("port") or 0)
    log_path = Path(backend.get("log_path") or (logs_dir() / f"{backend_id}.log"))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({str(k): str(v) for k, v in (backend.get("env") or {}).items()})
    bin_dir = str(Path(argv[0]).resolve().parent)
    if bin_dir and bin_dir not in {".", ""}:
        env["PATH"] = bin_dir + os.pathsep + env.get("PATH", "")

    last_log = ""
    skip: set[int] = set()
    want = preferred
    for _attempt in range(6):
        port = pick_port(want, host=host)
        if port in skip:
            port = pick_port(port + 1, host=host)
        skip.add(port)
        argv = set_argv_port(argv, port)
        before = log_path.stat().st_size if log_path.exists() else 0
        proc = _spawn(argv, env, log_path)
        store.update_backend(
            backend_id,
            state="starting",
            pid=proc.pid,
            log_path=str(log_path),
            argv=argv,
            host=host,
            port=port,
            base_url=f"http://{host}:{port}/v1",
            healthy=False,
        )
        died = False
        for _ in range(12):
            time.sleep(0.1)
            if proc.poll() is not None:
                died = True
                break
        if not died:
            updated = store.get_backend(backend_id)
            return updated or backend
        try:
            raw = log_path.read_bytes()[before:]
            last_log = raw.decode("utf-8", errors="replace")
        except OSError:
            last_log = tail_log(backend_id)
        if not bind_error_in_log(last_log):
            store.update_backend(backend_id, state="stopped", pid=None, healthy=False)
            raise ValueError(summarize_crash(last_log))
        want = port + 1
    store.update_backend(backend_id, state="stopped", pid=None, healthy=False)
    raise ValueError(summarize_crash(last_log or "couldn't bind HTTP server socket"))


def wait_started(backend_id: str, timeout: float = 8.0) -> dict[str, Any]:
    """Fail fast if the engine dies on bind; do not wait out a long GGUF load."""
    deadline = time.time() + timeout
    last = store.get_backend(backend_id)
    if last is None:
        raise KeyError(backend_id)
    while time.time() < deadline:
        last = store.get_backend(backend_id) or last
        pid = last.get("pid")
        if last.get("kind") == "adopt":
            return last
        if pid and not _pid_alive(int(pid)):
            raise ValueError(summarize_crash(tail_log(backend_id)))
        if _probe(str(last.get("base_url") or "")):
            return store.update_backend(backend_id, state="running", healthy=True) or last
        time.sleep(0.2)
    return last


def stop_backend(backend_id: str) -> dict[str, Any]:
    backend = store.get_backend(backend_id)
    if backend is None:
        raise KeyError(backend_id)
    pid = backend.get("pid")
    if pid and _pid_alive(int(pid)):
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                capture_output=True,
                check=False,
                timeout=10,
            )
        else:
            try:
                os.kill(int(pid), signal.SIGTERM)
            except OSError:
                pass
            deadline = time.time() + 5
            while time.time() < deadline and _pid_alive(int(pid)):
                time.sleep(0.1)
            if _pid_alive(int(pid)):
                try:
                    os.kill(int(pid), signal.SIGKILL)
                except OSError:
                    pass
    updated = store.update_backend(backend_id, state="stopped", pid=None, healthy=False)
    return updated or backend


def autostart_persisted() -> list[dict[str, Any]]:
    """Restart managed backends that were running last time the server stopped."""
    started: list[dict[str, Any]] = []
    for backend in store.list_backends():
        if backend.get("kind") != "managed":
            continue
        if backend.get("state") not in {"running", "starting"} and not backend.get("healthy"):
            continue
        try:
            started.append(start_backend(backend["id"]))
        except (KeyError, ValueError, OSError):
            continue
    return started


def deploy_from_spec(spec: dict[str, Any]) -> dict[str, Any]:
    """Create + optionally start a backend from a recipe or adopt an existing URL."""
    kind = str(spec.get("kind") or "recipe")
    if kind == "adopt":
        base = str(spec.get("base_url") or "").rstrip("/")
        if not base:
            raise ValueError("adopt requires base_url")
        model = str(spec.get("model_id") or spec.get("served_name") or "adopted")
        backend = store.new_backend(
            {
                "kind": "adopt",
                "name": spec.get("name") or model,
                "engine": spec.get("engine") or "external",
                "served_name": model,
                "base_url": base,
                "api_key": spec.get("api_key") or "",
                "host": "",
                "port": 0,
                "context_length": spec.get("context_length"),
                "state": "running",
            }
        )
        return reconcile_one(backend["id"])

    recipe = _load_recipe_arg(spec)

    host = str(spec.get("host") or recipe.launch.host or "127.0.0.1")
    preferred = spec.get("port")
    if preferred in (None, ""):
        preferred = recipe.launch.port
    port = pick_port(int(preferred or 0), host=host)
    model_path = str(spec.get("model_path") or recipe.model.path or "").strip()
    if recipe.engine == "llamacpp" and not model_path:
        raise ValueError(
            "llama.cpp needs a local GGUF path — use Download recipe weights or set Weights path"
        )
    source_kind = "local_path" if model_path else recipe.model.source_kind
    # Rebuild launch with chosen port.
    data = {
        "schema_version": recipe.schema_version,
        "name": recipe.name,
        "engine": recipe.engine,
        "executor": recipe.executor,
        "quant": recipe.quant,
        "model": {
            "id": recipe.model.id,
            "served_name": recipe.model.served_name,
            "source": {
                "kind": source_kind,
                "path": model_path,
                "repo": recipe.model.repo,
                "revision": recipe.model.revision,
                "filename": recipe.model.filename,
            },
        },
        "launch": {
            "host": host,
            "port": port,
            "context_length": spec.get("context_length") or recipe.launch.context_length,
            "extra_args": list(spec.get("extra_args") or recipe.launch.extra_args),
            "env": dict(recipe.launch.env),
        },
        "connect": {
            "aliases": list(recipe.connect.aliases),
            "max_concurrent": recipe.connect.max_concurrent,
        },
    }
    if spec.get("served_name"):
        data["model"]["served_name"] = spec["served_name"]
        data["model"]["id"] = spec.get("model_id") or spec["served_name"]
    if spec.get("extra_args") is None and recipe.engine == "llamacpp":
        from . import hostinfo, recommend as recmod
        from .placement import apply_ngl_args, fit_label

        snap = hostinfo.snapshot()
        have = recmod.nvidia_vram_mb(snap)
        apple = any(g.get("vendor") == "apple" for g in snap.get("gpus") or [])
        ctx = int(data["launch"]["context_length"] or 0)
        fit = fit_label(recipe.min_vram_mb, have, apple=apple, engine=recipe.engine, context_length=ctx)
        data["launch"]["extra_args"] = apply_ngl_args(data["launch"]["extra_args"], fit, ctx)
        data["notes"] = f"placement={fit}"
    recipe = recipe_from_dict(data)
    launch = render_process(recipe)
    argv = list(launch.argv)
    argv[0] = resolve_binary(recipe.engine, argv[0])
    backend = store.new_backend(
        {
            "kind": "managed",
            "name": spec.get("name") or recipe.model.served_name,
            "engine": recipe.engine,
            "served_name": recipe.model.served_name,
            "base_url": recipe.upstream_url(),
            "host": launch.host,
            "port": launch.port,
            "context_length": recipe.launch.context_length,
            "quant": recipe.quant,
            "argv": argv,
            "env": dict(launch.env),
            "model_path": spec.get("model_path") or recipe.model.path,
        }
    )
    if spec.get("autostart", True):
        start_backend(backend["id"])
    return store.get_backend(backend["id"]) or backend


def reconcile_one(backend_id: str) -> dict[str, Any]:
    reconcile()
    backend = store.get_backend(backend_id)
    if backend is None:
        raise KeyError(backend_id)
    return backend


def tail_log(backend_id: str, n: int = 80) -> str:
    backend = store.get_backend(backend_id)
    if backend is None:
        raise KeyError(backend_id)
    path = backend.get("log_path")
    if not path or not Path(path).exists():
        return ""
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    lines = text.splitlines()
    return "\n".join(lines[-n:])
