"""Outbound WebSocket client: register a local OpenAI server with Cortex."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
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

log = logging.getLogger("cortex-deployer")

# read=None: 256K prefills and long thinking turns exceed any idle-read
# budget; the gateway sends kind=cancel when the consumer is gone.
UPSTREAM_TIMEOUT = httpx.Timeout(connect=10.0, read=None, write=600.0, pool=10.0)
DEFAULT_CONCURRENCY = 2

SendFn = Callable[[dict], Awaitable[None]]

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

    url = join_upstream(upstream, path)
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
            await send(response_frame(rid, resp.status_code, out_headers, content))
    except asyncio.CancelledError:
        log.info("upstream cancelled rid=%s %s %s", rid, method, url)
        raise
    except Exception as exc:  # noqa: BLE001 — report upstream failure to the router
        log.exception("upstream error for %s %s", method, url)
        err = json.dumps({"error": f"deployer upstream error: {exc}"}).encode()
        await send(
            response_frame(rid, 502, {"content-type": "application/json"}, err)
        )


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
                await relay_request(client, args.upstream, msg, send)

        async with httpx.AsyncClient(timeout=UPSTREAM_TIMEOUT) as client:
            try:
                probe = await client.get(args.upstream.rstrip("/") + "/models")
                log.info("upstream /models -> %s", probe.status_code)
            except Exception as exc:  # noqa: BLE001
                log.warning("upstream /models probe failed: %s", exc)

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
                for task in list(inflight):
                    task.cancel()
                if inflight:
                    await asyncio.gather(*inflight, return_exceptions=True)
    log.info("disconnected from gateway")


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
