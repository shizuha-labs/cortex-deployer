from __future__ import annotations

from pathlib import Path

from .base import ProcessLaunch, _model_path
from ..spec import Recipe


def _binary_name(binary: str) -> str:
    return Path(binary).name.lower()


def is_mlx_lm(binary: str) -> bool:
    """True for Apple mlx-lm's HTTP server (not Rapid-MLX / vllm-mlx)."""
    name = _binary_name(binary)
    return name in {"mlx_lm.server", "mlx_lm"} or name.startswith("mlx_lm.")


def render_mlx(recipe: Recipe) -> ProcessLaunch:
    if recipe.engine != "mlx":
        raise ValueError("mlx renderer received a different engine")
    path = _model_path(recipe)
    # rapid-mlx is the production-shaped default; extra_args can switch binary.
    binary = "rapid-mlx"
    rest = list(recipe.launch.extra_args)
    if rest and not rest[0].startswith("-"):
        binary = rest.pop(0)
    if is_mlx_lm(binary):
        argv = [
            binary,
            "--model",
            path,
            "--host",
            recipe.launch.host,
            "--port",
            str(recipe.launch.port),
        ]
        argv.extend(rest)
    else:
        argv = [
            binary,
            "serve",
            path,
            "--host",
            recipe.launch.host,
            "--port",
            str(recipe.launch.port),
            "--served-model-name",
            recipe.model.served_name,
        ]
        argv.extend(rest)
    return ProcessLaunch(
        argv=tuple(argv),
        env=recipe.launch.env,
        host=recipe.launch.host,
        port=recipe.launch.port,
        engine="mlx",
    )
