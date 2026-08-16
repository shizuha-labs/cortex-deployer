from __future__ import annotations

from .base import ProcessLaunch, _model_path
from ..spec import Recipe


def render_sglang(recipe: Recipe) -> ProcessLaunch:
    if recipe.engine != "sglang":
        raise ValueError("sglang renderer received a different engine")
    extra = " ".join(recipe.launch.extra_args)
    # Multi-node / explicit TP>local is a reviewed adapter later. Fail closed
    # on the flags that would need rendezvous we do not render.
    for banned in ("--dist-init-addr", "--nnodes", "--node-rank"):
        if banned in extra or banned in recipe.launch.extra_args:
            raise ValueError(
                "sglang multi-node flags are unsupported until a reviewed "
                "rank/rendezvous adapter exists"
            )
    path = _model_path(recipe)
    argv = [
        "python3",
        "-m",
        "sglang.launch_server",
        "--model-path",
        path,
        "--served-model-name",
        recipe.model.served_name,
        "--host",
        recipe.launch.host,
        "--port",
        str(recipe.launch.port),
    ]
    if recipe.launch.context_length:
        argv.extend(["--context-length", str(recipe.launch.context_length)])
    argv.extend(recipe.launch.extra_args)
    return ProcessLaunch(
        argv=tuple(argv),
        env=recipe.launch.env,
        host=recipe.launch.host,
        port=recipe.launch.port,
        engine="sglang",
    )
