from __future__ import annotations

from .base import ProcessLaunch, _model_path
from ..spec import Recipe


def render_llamacpp(recipe: Recipe) -> ProcessLaunch:
    if recipe.engine != "llamacpp":
        raise ValueError("llamacpp renderer received a different engine")
    path = _model_path(recipe)
    argv = [
        "llama-server",
        "-m",
        path,
        "--host",
        recipe.launch.host,
        "--port",
        str(recipe.launch.port),
        "--alias",
        recipe.model.served_name,
    ]
    if recipe.launch.context_length:
        argv.extend(["-c", str(recipe.launch.context_length)])
    argv.extend(recipe.launch.extra_args)
    return ProcessLaunch(
        argv=tuple(argv),
        env=recipe.launch.env,
        host=recipe.launch.host,
        port=recipe.launch.port,
        engine="llamacpp",
    )
