from __future__ import annotations

from .base import ProcessLaunch, _model_path
from ..spec import Recipe


def render_vllm(recipe: Recipe) -> ProcessLaunch:
    if recipe.engine != "vllm":
        raise ValueError("vllm renderer received a different engine")
    path = _model_path(recipe)
    argv = [
        "python3",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        path,
        "--served-model-name",
        recipe.model.served_name,
        "--host",
        recipe.launch.host,
        "--port",
        str(recipe.launch.port),
    ]
    if recipe.launch.context_length:
        argv.extend(["--max-model-len", str(recipe.launch.context_length)])
    argv.extend(recipe.launch.extra_args)
    return ProcessLaunch(
        argv=tuple(argv),
        env=recipe.launch.env,
        host=recipe.launch.host,
        port=recipe.launch.port,
        engine="vllm",
    )
