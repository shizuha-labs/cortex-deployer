"""cortex-deployer CLI."""

from __future__ import annotations

import argparse
import json

from . import __version__
from .client import add_connect_arguments, run_connect
from .engines import render_process
from .recipes import list_examples, load_recipe
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


def _cmd_connect(args: argparse.Namespace) -> int:
    run_connect(args)
    return 0


def _cmd_server(args: argparse.Namespace) -> int:
    from .httpapi import serve

    host = args.host
    port = int(args.port)
    httpd = serve(host, port)
    url = f"http://{host}:{port}/"
    print(f"Cortex Deployer UI  {url}")
    print("API                 /api/backends  /api/host  /v1/models")
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cortex-deployer",
        description="Deploy a local model and connect it to a Cortex catalog",
    )
    parser.add_argument("--version", action="store_true")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("engines", help="list supported engines")
    sub.add_parser("recipes", help="list bundled example recipes")

    render = sub.add_parser("render", help="print argv for a recipe (no spawn)")
    render.add_argument("recipe")
    render.add_argument("--json", action="store_true")

    connect = sub.add_parser(
        "connect",
        help="dial out to a Cortex deployer gateway and relay /v1",
    )
    add_connect_arguments(connect)

    server = sub.add_parser(
        "server",
        aliases=["up", "web"],
        help="start the local control-plane UI (DeepSeek Harness-style)",
    )
    server.add_argument("--host", default="127.0.0.1")
    server.add_argument("--port", type=int, default=7480)
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
        "connect": _cmd_connect,
        "server": _cmd_server,
        "up": _cmd_server,
        "web": _cmd_server,
        "version": _cmd_version,
    }
    raise SystemExit(handlers[args.command](args))


if __name__ == "__main__":
    main()
