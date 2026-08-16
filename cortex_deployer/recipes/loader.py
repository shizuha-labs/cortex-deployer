from __future__ import annotations

from pathlib import Path

import yaml

from ..spec import Recipe, recipe_from_dict


def example_dir() -> Path:
    return Path(__file__).resolve().parent / "examples"


def list_examples() -> list[Path]:
    return sorted(example_dir().glob("*.yaml"))


def load_recipe(path: str | Path) -> Recipe:
    raw = Path(path).read_text(encoding="utf-8")
    data = yaml.safe_load(raw)
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not contain a YAML mapping")
    return recipe_from_dict(data)
