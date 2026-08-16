"""Versioned recipe / inventory values. Pure: no I/O, no Django, no Kubernetes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


ENGINE_KINDS = ("llamacpp", "sglang", "vllm", "mlx")
EngineKind = Literal["llamacpp", "sglang", "vllm", "mlx"]
ExecutorKind = Literal["process", "docker"]
RECIPE_SCHEMA = "deployer.recipe.v1"


@dataclass(frozen=True)
class ModelRef:
    id: str
    served_name: str
    source_kind: Literal["local_path", "huggingface"] = "local_path"
    path: str = ""
    repo: str = ""
    revision: str = ""
    filename: str = ""


@dataclass(frozen=True)
class LaunchSpec:
    host: str = "127.0.0.1"
    port: int = 8080
    context_length: int | None = None
    extra_args: tuple[str, ...] = ()
    env: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class ConnectSpec:
    aliases: tuple[str, ...] = ()
    max_concurrent: int = 1
    gateway: str = ""


@dataclass(frozen=True)
class Recipe:
    schema_version: str
    name: str
    engine: EngineKind
    executor: ExecutorKind
    model: ModelRef
    launch: LaunchSpec = field(default_factory=LaunchSpec)
    connect: ConnectSpec = field(default_factory=ConnectSpec)
    quant: str = ""
    min_vram_mb: int = 0
    notes: str = ""
    download_glob: str = ""

    def upstream_url(self) -> str:
        return f"http://{self.launch.host}:{int(self.launch.port)}/v1"


def _require_str(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"recipe field {key!r} must be a non-empty string")
    return value.strip()


def recipe_from_dict(data: dict[str, Any]) -> Recipe:
    if not isinstance(data, dict):
        raise ValueError("recipe must be a mapping")
    schema = data.get("schema_version") or RECIPE_SCHEMA
    if schema != RECIPE_SCHEMA:
        raise ValueError(f"unsupported recipe schema {schema!r}")
    engine = _require_str(data, "engine")
    if engine not in ENGINE_KINDS:
        raise ValueError(f"unknown engine {engine!r}; expected one of {ENGINE_KINDS}")
    executor = str(data.get("executor") or "process")
    if executor not in ("process", "docker"):
        raise ValueError(f"unsupported executor {executor!r}")
    model_raw = data.get("model") or {}
    if not isinstance(model_raw, dict):
        raise ValueError("recipe.model must be a mapping")
    model_id = str(model_raw.get("id") or "").strip()
    if not model_id:
        raise ValueError("recipe.model.id is required")
    source = model_raw.get("source") or {}
    if not isinstance(source, dict):
        source = {}
    source_kind = str(source.get("kind") or "local_path")
    if source_kind not in ("local_path", "huggingface"):
        raise ValueError(f"unsupported model source {source_kind!r}")
    launch_raw = data.get("launch") or {}
    if not isinstance(launch_raw, dict):
        raise ValueError("recipe.launch must be a mapping")
    extra = launch_raw.get("extra_args") or []
    if not isinstance(extra, list) or any(not isinstance(x, str) for x in extra):
        raise ValueError("launch.extra_args must be a list of strings")
    env_raw = launch_raw.get("env") or {}
    if not isinstance(env_raw, dict):
        raise ValueError("launch.env must be a mapping")
    env = tuple((str(k), str(v)) for k, v in env_raw.items())
    ctx = launch_raw.get("context_length")
    if ctx is not None:
        ctx = int(ctx)
        if ctx <= 0:
            raise ValueError("launch.context_length must be positive")
    connect_raw = data.get("connect") or {}
    if not isinstance(connect_raw, dict):
        raise ValueError("recipe.connect must be a mapping")
    aliases = connect_raw.get("aliases") or []
    if not isinstance(aliases, list) or any(not isinstance(x, str) for x in aliases):
        raise ValueError("connect.aliases must be a list of strings")
    max_concurrent = int(connect_raw.get("max_concurrent") or 1)
    if max_concurrent < 1:
        raise ValueError("connect.max_concurrent must be >= 1")
    return Recipe(
        schema_version=schema,
        name=_require_str(data, "name"),
        engine=engine,  # type: ignore[arg-type]
        executor=executor,  # type: ignore[arg-type]
        model=ModelRef(
            id=model_id,
            served_name=str(model_raw.get("served_name") or model_id).strip(),
            source_kind=source_kind,  # type: ignore[arg-type]
            path=str(source.get("path") or ""),
            repo=str(source.get("repo") or ""),
            revision=str(source.get("revision") or ""),
            filename=str(source.get("filename") or ""),
        ),
        launch=LaunchSpec(
            host=str(launch_raw.get("host") or "127.0.0.1"),
            port=int(launch_raw.get("port") or 8080),
            context_length=ctx,
            extra_args=tuple(extra),
            env=env,
        ),
        connect=ConnectSpec(
            aliases=tuple(a.strip() for a in aliases if str(a).strip()),
            max_concurrent=max_concurrent,
            gateway=str(connect_raw.get("gateway") or ""),
        ),
        quant=str(data.get("quant") or ""),
        min_vram_mb=int(data.get("min_vram_mb") or 0),
        notes=str(data.get("notes") or ""),
        download_glob=str(data.get("download_glob") or source.get("glob") or ""),
    )
