"""Local control-plane HTTP server.

Same shape as DeepSeek Harness `dsh web`: one process, browser UI on localhost,
JSON API for backends, plus an OpenAI-compatible /v1 that fans out to them.
"""

from __future__ import annotations

import json
import posixpath
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from . import __version__, download, hostinfo, recommend, store
from . import connect_ctl
from .recipes import list_examples, load_recipe
from .runtime import (
    _loopback_url,
    deploy_from_spec,
    reconcile,
    start_backend,
    stop_backend,
    tail_log,
)
from .spec import ENGINE_KINDS

WEB_DIR = Path(__file__).resolve().parent / "web"


def _json_bytes(payload: Any, status: int = 200) -> tuple[int, bytes, str]:
    body = json.dumps(payload).encode("utf-8")
    return status, body, "application/json; charset=utf-8"


def handle(method: str, path: str, body: bytes) -> tuple[int, bytes, str]:
    parsed = urlparse(path)
    route = posixpath.normpath(parsed.path)
    payload: Any = {}
    if body:
        try:
            payload = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError:
            return _json_bytes({"error": "invalid json"}, 400)

    if method == "GET" and route in {"/", "/index.html"}:
        index = WEB_DIR / "index.html"
        return 200, index.read_bytes(), "text/html; charset=utf-8"
    if method == "GET" and route.startswith("/static/"):
        name = route[len("/static/") :]
        if ".." in name or name.startswith("/"):
            return _json_bytes({"error": "not found"}, 404)
        target = WEB_DIR / name
        if not target.is_file():
            return _json_bytes({"error": "not found"}, 404)
        ctype = "text/css" if name.endswith(".css") else "text/javascript"
        return 200, target.read_bytes(), ctype

    if method == "GET" and route == "/api/health":
        return _json_bytes({"ok": True, "version": __version__})
    if method == "GET" and route == "/api/host":
        return _json_bytes(hostinfo.snapshot())
    if method == "GET" and route == "/api/engines":
        return _json_bytes({"engines": list(ENGINE_KINDS)})
    if method == "GET" and route == "/api/recipes":
        recipes = []
        for path in list_examples():
            rec = load_recipe(path)
            recipes.append(
                {
                    "file": path.name,
                    "name": rec.name,
                    "engine": rec.engine,
                    "served_name": rec.model.served_name,
                    "context_length": rec.launch.context_length,
                    "quant": rec.quant,
                    "path": rec.model.path,
                    "min_vram_mb": rec.min_vram_mb,
                    "notes": rec.notes,
                    "download_repo": rec.model.repo,
                    "download_glob": rec.download_glob,
                }
            )
        return _json_bytes({"recipes": recipes})
    if method == "GET" and route == "/api/recommend":
        return _json_bytes(recommend.recommend())
    if method == "GET" and route == "/api/downloads":
        return _json_bytes({"jobs": download.list_jobs(), "local": download.list_local_models()})
    if method == "POST" and route == "/api/downloads":
        if not isinstance(payload, dict):
            return _json_bytes({"error": "object required"}, 400)
        try:
            job = download.start_download(
                str(payload.get("repo") or ""),
                filename=str(payload.get("filename") or ""),
                glob=str(payload.get("glob") or ""),
            )
        except ValueError as exc:
            return _json_bytes({"error": str(exc)}, 400)
        return _json_bytes(job, 202)
    if method == "GET" and route.startswith("/api/downloads/"):
        job_id = route.rsplit("/", 1)[-1]
        job = download.get_job(job_id)
        if job is None:
            return _json_bytes({"error": "not found"}, 404)
        return _json_bytes(job)
    if method == "GET" and route == "/api/local-models":
        return _json_bytes({"models": download.list_local_models()})
    if method == "GET" and route == "/api/backends":
        return _json_bytes({"backends": reconcile()})
    if method == "POST" and route == "/api/backends":
        if not isinstance(payload, dict):
            return _json_bytes({"error": "object required"}, 400)
        try:
            backend = deploy_from_spec(payload)
        except (ValueError, KeyError) as exc:
            return _json_bytes({"error": str(exc)}, 400)
        return _json_bytes(backend, 201)

    parts = route.strip("/").split("/")
    if len(parts) >= 3 and parts[0] == "api" and parts[1] == "backends":
        backend_id = parts[2]
        action = parts[3] if len(parts) > 3 else ""
        if method == "GET" and not action:
            row = store.get_backend(backend_id)
            if row is None:
                return _json_bytes({"error": "not found"}, 404)
            return _json_bytes(row)
        if method == "DELETE" and not action:
            try:
                stop_backend(backend_id)
            except KeyError:
                return _json_bytes({"error": "not found"}, 404)
            store.delete_backend(backend_id)
            return _json_bytes({"ok": True})
        if method == "POST" and action == "start":
            try:
                return _json_bytes(start_backend(backend_id))
            except KeyError:
                return _json_bytes({"error": "not found"}, 404)
            except ValueError as exc:
                return _json_bytes({"error": str(exc)}, 400)
        if method == "POST" and action == "stop":
            try:
                return _json_bytes(stop_backend(backend_id))
            except KeyError:
                return _json_bytes({"error": "not found"}, 404)
        if method == "GET" and action == "logs":
            try:
                return _json_bytes({"log": tail_log(backend_id)})
            except KeyError:
                return _json_bytes({"error": "not found"}, 404)
        if method == "POST" and action == "connect":
            if not isinstance(payload, dict):
                return _json_bytes({"error": "object required"}, 400)
            try:
                return _json_bytes(
                    connect_ctl.start_connect(
                        backend_id,
                        gateway=str(payload.get("gateway") or ""),
                        token=str(payload.get("token") or ""),
                        model=str(payload.get("model") or ""),
                    )
                )
            except KeyError:
                return _json_bytes({"error": "not found"}, 404)
            except ValueError as exc:
                return _json_bytes({"error": str(exc)}, 400)
        if method == "POST" and action == "disconnect":
            try:
                return _json_bytes(connect_ctl.stop_connect(backend_id))
            except KeyError:
                return _json_bytes({"error": "not found"}, 404)

    if method == "GET" and route in {"/v1/models", "/models"}:
        models = []
        for backend in reconcile():
            if backend.get("healthy") or backend.get("state") == "running":
                models.append(
                    {
                        "id": backend.get("served_name") or backend.get("name"),
                        "object": "model",
                        "owned_by": "cortex-deployer",
                    }
                )
        return _json_bytes({"object": "list", "data": models})

    if method == "POST" and route in {"/v1/chat/completions", "/chat/completions"}:
        return _proxy_openai(payload)

    return _json_bytes({"error": f"no route {method} {route}"}, 404)


def _pick_backend(model: str) -> dict[str, Any] | None:
    target = None
    for backend in reconcile():
        if model and model in {
            backend.get("served_name"),
            backend.get("name"),
            backend.get("id"),
        }:
            return backend
        if not model and backend.get("healthy"):
            target = backend
            break
    return target


def _proxy_openai(payload: dict[str, Any]) -> tuple[int, bytes, str]:
    model = str(payload.get("model") or "")
    target = _pick_backend(model)
    if target is None:
        return _json_bytes({"error": "no healthy backend for model"}, 404)
    base = _loopback_url(str(target.get("base_url") or "")).rstrip("/")
    req = Request(
        base + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"content-type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(req, timeout=600) as resp:
            raw = resp.read()
            status = getattr(resp, "status", 200)
            ctype = resp.headers.get("content-type") or "application/json"
            return status, raw, ctype
    except Exception as exc:  # noqa: BLE001
        return _json_bytes({"error": f"upstream: {exc}"}, 502)


class Handler(BaseHTTPRequestHandler):
    server_version = "cortex-deployer/" + __version__

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def _cors(self) -> None:
        self.send_header("access-control-allow-origin", "*")
        self.send_header("access-control-allow-headers", "content-type, authorization")
        self.send_header("access-control-allow-methods", "GET, POST, DELETE, OPTIONS, HEAD")

    def _send(self, status: int, body: bytes, ctype: str) -> None:
        self.send_response(status)
        self.send_header("content-type", ctype)
        self.send_header("content-length", str(len(body)))
        self.send_header("cache-control", "no-store")
        self._cors()
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_HEAD(self) -> None:  # noqa: N802
        status, body, ctype = handle("GET", self.path, b"")
        self._send(status, body, ctype)

    def do_GET(self) -> None:  # noqa: N802
        status, body, ctype = handle("GET", self.path, b"")
        self._send(status, body, ctype)

    def do_DELETE(self) -> None:  # noqa: N802
        status, body, ctype = handle("DELETE", self.path, b"")
        self._send(status, body, ctype)

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("content-length") or 0)
        body = self.rfile.read(length) if length else b""
        parsed = urlparse(self.path)
        route = posixpath.normpath(parsed.path)
        payload: Any = {}
        if body:
            try:
                payload = json.loads(body.decode("utf-8"))
            except json.JSONDecodeError:
                payload = {}
        if (
            route in {"/v1/chat/completions", "/chat/completions"}
            and isinstance(payload, dict)
            and payload.get("stream")
        ):
            self._stream_openai(payload)
            return
        status, out, ctype = handle("POST", self.path, body)
        self._send(status, out, ctype)

    def _stream_openai(self, payload: dict[str, Any]) -> None:
        target = _pick_backend(str(payload.get("model") or ""))
        if target is None:
            self._send(*_json_bytes({"error": "no healthy backend for model"}, 404))
            return
        base = str(target.get("base_url") or "").rstrip("/")
        req = Request(
            base + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"content-type": "application/json"},
            method="POST",
        )
        try:
            resp = urlopen(req, timeout=600)
        except Exception as exc:  # noqa: BLE001
            self._send(*_json_bytes({"error": f"upstream: {exc}"}, 502))
            return
        try:
            self.send_response(getattr(resp, "status", 200))
            ctype = resp.headers.get("content-type") or "text/event-stream"
            self.send_header("content-type", ctype)
            self.send_header("cache-control", "no-cache")
            self._cors()
            self.end_headers()
            while True:
                chunk = resp.read(1024)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
        finally:
            resp.close()


def serve(host: str = "127.0.0.1", port: int = 7480) -> ThreadingHTTPServer:
    httpd = ThreadingHTTPServer((host, port), Handler)
    httpd.daemon_threads = True
    return httpd


def serve_in_thread(host: str = "127.0.0.1", port: int = 7480) -> ThreadingHTTPServer:
    httpd = serve(host, port)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd
