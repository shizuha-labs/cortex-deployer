from __future__ import annotations

from dataclasses import dataclass

from ..spec import Recipe


@dataclass(frozen=True)
class ProcessLaunch:
    argv: tuple[str, ...]
    env: tuple[tuple[str, str], ...]
    host: str
    port: int
    engine: str


def _model_path(recipe: Recipe) -> str:
    if recipe.model.source_kind == "local_path":
        if not recipe.model.path:
            raise ValueError("local_path source requires model.source.path")
        return recipe.model.path
    if recipe.model.source_kind == "huggingface":
        if not recipe.model.repo:
            raise ValueError("huggingface source requires model.source.repo")
        return recipe.model.repo
    raise ValueError(f"unsupported source {recipe.model.source_kind!r}")


def render_process(recipe: Recipe) -> ProcessLaunch:
    if recipe.executor != "process":
        raise ValueError(
            f"executor {recipe.executor!r} is not implemented; use process"
        )
    if recipe.engine == "llamacpp":
        from .llamacpp import render_llamacpp

        return render_llamacpp(recipe)
    if recipe.engine == "sglang":
        from .sglang import render_sglang

        return render_sglang(recipe)
    if recipe.engine == "vllm":
        from .vllm import render_vllm

        return render_vllm(recipe)
    if recipe.engine == "mlx":
        from .mlx import render_mlx

        return render_mlx(recipe)
    if recipe.engine == "comfyui":
        from .comfyui import render_comfyui

        return render_comfyui(recipe)
    raise ValueError(f"unknown engine {recipe.engine!r}")
