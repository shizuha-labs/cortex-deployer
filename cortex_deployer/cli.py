"""cortex-deployer CLI."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

from . import __version__
from .client import add_connect_arguments, run_connect
from .hostinfo import advertise_urls, default_bind_host
from .engines import render_process
from .recipes import list_examples, load_recipe
from .runtime import resolve_binary
from .spec import ENGINE_KINDS


def _cmd_engines(_args: argparse.Namespace) -> int:
    for kind in ENGINE_KINDS:
        print(kind)
    return 0


def _cmd_recipes(_args: argparse.Namespace) -> int:
    for path in list_examples():
        print(path.name)
    return 0


def _cmd_render(args: argparse.Namespace) -> int:
    recipe = load_recipe(args.recipe)
    launch = render_process(recipe)
    if args.json:
        print(
            json.dumps(
                {
                    "engine": launch.engine,
                    "argv": list(launch.argv),
                    "env": {k: v for k, v in launch.env},
                    "host": launch.host,
                    "port": launch.port,
                    "upstream": recipe.upstream_url(),
                },
                indent=2,
            )
        )
        return 0
    print(" ".join(launch.argv))
    return 0


def _cmd_attach(args: argparse.Namespace) -> int:
    from . import attach

    if args.scan:
        hits = attach.scan_local()
        if args.json:
            print(json.dumps({"hits": hits}, indent=2))
            return 0 if hits else 1
        if not hits:
            print("no local OpenAI-compatible server found")
            print("tried LM Studio :1234, Ollama :11434, vLLM :8000, llama.cpp :8080, SGLang :30000")
            return 1
        for hit in hits:
            models = ", ".join(hit.get("models") or []) or "(no /v1/models ids)"
            print(f"{hit['label']:22} {hit['url']}  {models}")
        return 0

    url = (args.url or "").strip()
    if not url:
        print("attach requires a URL, or --scan", file=sys.stderr)
        return 2
    existing = attach.already_attached(url)
    if existing and not args.force:
        print(f"already attached {existing.get('served_name')} @ {existing.get('base_url')} ({existing.get('id')})")
        return 0
    try:
        backend = attach.attach(
            url,
            model=args.model or "",
            api_key=args.api_key or "",
            engine=args.engine or "",
            require_probe=not args.force,
        )
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(backend, indent=2))
    else:
        print(
            f"attached {backend.get('served_name')} @ {backend.get('base_url')}"
            + (" healthy" if backend.get("healthy") else " (probe pending)")
        )
        print("tunnel: cortex-deployer connect --upstream", backend.get("base_url"), "--model", backend.get("served_name"), "--token <pairing>")
    if args.connect:
        if not args.token:
            print("attach --connect needs --token (pairing from Cortex)", file=sys.stderr)
            return 2
        from .client import parse_connect_args, run_connect

        conn = parse_connect_args(
            [
                "--gateway",
                args.gateway,
                "--token",
                args.token,
                "--model",
                args.model or str(backend.get("served_name") or ""),
                "--upstream",
                str(backend.get("base_url") or url),
            ]
        )
        if args.api_key:
            os.environ["CORTEX_DEPLOYER_UPSTREAM_KEY"] = args.api_key
        run_connect(conn)
    return 0


def _cmd_connect(args: argparse.Namespace) -> int:
    from . import selfupdate

    if getattr(args, "auto_update", False):
        selfupdate.set_auto_update(True)
    selfupdate.apply_on_connect()
    run_connect(args)
    return 0


def _cmd_recommend(args: argparse.Namespace) -> int:
    from .recommend import recommend

    out = recommend()
    if args.json:
        print(json.dumps(out, indent=2))
        return 0
    print(f"best={out.get('best') or 'none'} vram_mb={out.get('vram_mb')} apple={out.get('apple')}")
    for row in out.get("recipes") or []:
        print(f"{row.get('fit', '?'):12} {row.get('file')}")
    return 0


def _cmd_setup(args: argparse.Namespace) -> int:
    from .setup import start_setup, wait_job

    job = start_setup(args.recipe or "")
    print(f"setup {job['id']} recipe={job.get('recipe')}", flush=True)
    done = wait_job(job["id"])
    if done.get("state") == "done":
        print(done.get("base_url") or "done")
        return 0
    print(done.get("error") or "setup failed", flush=True)
    return 1


def _cmd_download(args: argparse.Namespace) -> int:
    from .download import start_download, wait_job

    job = start_download(args.repo, filename=args.filename or "", glob=args.glob or "")
    print(f"download {job['id']} starting", flush=True)
    done = wait_job(job["id"])
    if done.get("state") == "done":
        print(done.get("path") or "done")
        return 0
    print(done.get("error") or "download failed", flush=True)
    return 1


def _cmd_run(args: argparse.Namespace) -> int:
    """Render a recipe and exec the engine in the foreground (launchd/systemd)."""
    recipe = load_recipe(args.recipe)
    launch = render_process(recipe)
    argv = list(launch.argv)
    argv[0] = resolve_binary(recipe.engine, argv[0])
    env = os.environ.copy()
    env.update({str(k): str(v) for k, v in launch.env})
    if args.dry_run:
        print(" ".join(argv))
        return 0
    if os.name == "nt":
        return int(subprocess.call(argv, env=env))
    try:
        os.execvpe(argv[0], argv, env)
    except OSError as exc:
        print(f"failed to exec {argv[0]}: {exc}", file=sys.stderr)
        return 1
    return 0

def _cmd_server(args: argparse.Namespace) -> int:
    from . import selfupdate
    from .httpapi import serve
    from .runtime import autostart_persisted

    host = args.host
    port = int(args.port)
    if args.auto_update:
        selfupdate.set_auto_update(True)
    auto = bool(args.auto_update) or selfupdate.auto_update_enabled()
    selfupdate.apply_on_start(auto=auto, host=host, port=port)
    httpd = serve(host, port)
    bound_host, bound_port = httpd.server_address[:2]
    started = autostart_persisted()
    urls = advertise_urls(str(bound_host), int(bound_port))
    if bound_port != port:
        print(f"port {port} unavailable; listening on {bound_port}", flush=True)
    print(f"Cortex Deployer UI  {urls[0]}")
    for extra in urls[1:]:
        print(f"                    {extra}")
    print("API                 /api/backends  /api/recommend  /api/setup  /api/downloads  /v1/models")
    if started:
        print(f"autostart           {len(started)} backend(s)")
    print("Ctrl+C to stop.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        httpd.server_close()
    return 0


def _cmd_version(_args: argparse.Namespace) -> int:
    print(__version__)
    return 0


def _cmd_update(args: argparse.Namespace) -> int:
    from . import catalog, selfupdate

    cat = catalog.fetch_catalog(force=True, timeout=8.0)
    latest = selfupdate.latest_from_catalog(cat)
    print(f"current={__version__} latest={latest or '?'}")
    if args.check:
        return 0 if not selfupdate.update_available(latest) else 2
    if latest and not selfupdate.update_available(latest) and not args.force:
        print("already current — models stay, no reinstall needed")
        return 0
    try:
        result = selfupdate.run_update(selfupdate.tarball_from_catalog(cat))
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1
    print("updated in place (models kept)")
    if result.get("detail"):
        print(result["detail"])
    if args.restart:
        host = default_bind_host()
        os.execv(
            sys.executable,
            [sys.executable, "-m", "cortex_deployer", "server", "--host", host],
        )
    print("restart: cortex-deployer server")
    return 0


def _cmd_auto_update(args: argparse.Namespace) -> int:
    from . import selfupdate

    if args.off:
        selfupdate.set_auto_update(False)
        print("auto-update off")
        return 0
    selfupdate.set_auto_update(True)
    print("auto-update on — server upgrades in place on start")
    args.check = False
    args.force = bool(getattr(args, "force", False))
    args.restart = bool(getattr(args, "restart", False))
    return _cmd_update(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cortex-deployer",
        description="Deploy a local model and connect it to a Cortex catalog",
    )
    parser.add_argument("--version", action="store_true")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("engines", help="list supported engines")
    sub.add_parser("recipes", help="list bundled example recipes")
    upd = sub.add_parser(
        "update",
        aliases=["upgrade"],
        help="upgrade this install in place from GitHub main (keeps models)",
    )
    upd.add_argument("--check", action="store_true", help="print versions and exit 2 if newer")
    upd.add_argument("--force", action="store_true", help="reinstall current latest anyway")
    upd.add_argument("--restart", action="store_true", help="exec server after a successful update")
    au = sub.add_parser(
        "auto-update",
        help="enable upgrade-on-start (persisted) and update now",
    )
    au.add_argument("--off", action="store_true", help="disable upgrade-on-start")

    render = sub.add_parser("render", help="print argv for a recipe (no spawn)")
    render.add_argument("recipe")
    render.add_argument("--json", action="store_true")

    run = sub.add_parser(
        "run",
        help="exec a recipe in the foreground (launchd/systemd KeepAlive)",
    )
    run.add_argument("recipe")
    run.add_argument(
        "--dry-run",
        action="store_true",
        help="print resolved argv and exit",
    )

    connect = sub.add_parser(
        "connect",
        help="dial out to a Cortex deployer gateway and relay /v1",
    )
    add_connect_arguments(connect)

    att = sub.add_parser(
        "attach",
        help="point at an already-running local /v1 (LM Studio, Ollama, vLLM) and tunnel it to Cortex",
    )
    att.add_argument("url", nargs="?", default="", help="OpenAI-compatible base, e.g. http://127.0.0.1:1234/v1")
    att.add_argument("--scan", action="store_true", help="probe well-known local ports and print what is up")
    att.add_argument("--model", default="", help="catalog/served name (default: first id from /v1/models)")
    att.add_argument("--api-key", default="", help="upstream key if the local server requires one")
    att.add_argument("--engine", default="", help="lmstudio / ollama / vllm / llamacpp / sglang / external")
    att.add_argument("--force", action="store_true", help="register even if /v1/models does not answer")
    att.add_argument("--json", action="store_true")
    att.add_argument("--connect", action="store_true", help="after attach, open the Cortex tunnel (needs --token)")
    att.add_argument(
        "--gateway",
        default=os.environ.get("CORTEX_DEPLOYER_GATEWAY")
        or "wss://cortex.shizuha.com/cortex/deployer/ws/register",
    )
    att.add_argument("--token", default=os.environ.get("CORTEX_DEPLOYER_TOKEN") or "")

    rec = sub.add_parser("recommend", help="rank bundled recipes against detected VRAM")
    rec.add_argument("--json", action="store_true")

    su = sub.add_parser(
        "setup",
        help="one-click: recommended recipe + official llama-server + weights + start",
    )
    su.add_argument("--recipe", default="", help="bundled recipe filename (default: GPU fit)")

    dl = sub.add_parser("download", help="download Hugging Face GGUF weights")
    dl.add_argument("--repo", required=True, help="org/name")
    dl.add_argument("--filename", default="", help="single file under the repo")
    dl.add_argument("--glob", default="", help="fnmatch, e.g. *UD-Q3_K_XL.gguf")

    server = sub.add_parser(
        "server",
        aliases=["up", "web"],
        help="start the local control-plane UI (DeepSeek Harness-style)",
    )
    server.add_argument(
        "--host",
        default=default_bind_host(),
        help="listen address (Linux/WSL default 0.0.0.0 so the distro IP is reachable; Windows default 127.0.0.1)",
    )
    server.add_argument(
        "--port",
        type=int,
        default=7480,
        help="preferred listen port (>=1024); if busy or forbidden, the next free high port is used",
    )
    server.add_argument(
        "--auto-update",
        action="store_true",
        help="if the catalog lists a newer release, upgrade in place then re-exec (or set CORTEX_DEPLOYER_AUTO_UPDATE=1)",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.version or args.command is None:
        if args.command is None and not args.version:
            parser.print_help()
            raise SystemExit(0 if args.version else 0)
        print(__version__)
        return
    handlers = {
        "engines": _cmd_engines,
        "recipes": _cmd_recipes,
        "render": _cmd_render,
        "run": _cmd_run,
        "connect": _cmd_connect,
        "attach": _cmd_attach,
        "recommend": _cmd_recommend,
        "setup": _cmd_setup,
        "download": _cmd_download,
        "update": _cmd_update,
        "upgrade": _cmd_update,
        "auto-update": _cmd_auto_update,
        "server": _cmd_server,
        "up": _cmd_server,
        "web": _cmd_server,
        "version": _cmd_version,
    }
    raise SystemExit(handlers[args.command](args))


if __name__ == "__main__":
    main()
