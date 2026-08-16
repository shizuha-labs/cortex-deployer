"""Pick recipes that fit detected VRAM (LM Studio-style GPU fit)."""

from __future__ import annotations

from typing import Any

from . import hostinfo
from .recipes import list_examples, load_recipe

# Generic templates must not beat a real model recipe on the same card.
_EXAMPLE_REPOS = {"", "org/example-model"}
_EXAMPLE_FILES = {
    "llamacpp-cuda.yaml",
    "vllm-openai.yaml",
    "sglang-openai.yaml",
    "mlx-process.yaml",
}


def nvidia_vram_mb(snap: dict[str, Any] | None = None) -> int:
    snap = snap or hostinfo.snapshot()
    nvs = [g for g in snap.get("gpus") or [] if g.get("vendor") == "nvidia"]
    if nvs:
        return max(int(g.get("memory_mb") or 0) for g in nvs)
    return 0


def fit_label(min_vram_mb: int, have_mb: int, *, apple: bool) -> str:
    if min_vram_mb <= 0:
        return "unknown"
    if apple:
        return "ok"
    if have_mb <= 0:
        return "cpu"
    if have_mb >= min_vram_mb:
        return "recommended"
    if have_mb >= int(min_vram_mb * 0.72):
        return "tight"
    return "skip"


def is_example_recipe(row: dict[str, Any], *, apple: bool) -> bool:
    fname = str(row.get("file") or "")
    if fname.startswith("qwen"):
        return False
    if apple and fname.startswith("mlx-macos"):
        return False
    if fname in _EXAMPLE_FILES:
        return True
    repo = str(row.get("download_repo") or "")
    name = str(row.get("name") or "")
    return repo in _EXAMPLE_REPOS or name.startswith("example-")


def recommend() -> dict[str, Any]:
    snap = hostinfo.snapshot()
    have = nvidia_vram_mb(snap)
    apple = any(g.get("vendor") == "apple" for g in snap.get("gpus") or [])
    rows = []
    for path in list_examples():
        rec = load_recipe(path)
        label = fit_label(rec.min_vram_mb, have, apple=apple)
        if rec.engine == "mlx" and not apple:
            label = "skip"
        if rec.engine == "mlx" and apple:
            label = "recommended"
        if rec.engine in {"vllm", "sglang"} and have < 8000 and not apple:
            label = "tight" if have else "cpu"
        row = {
            "file": path.name,
            "name": rec.name,
            "engine": rec.engine,
            "served_name": rec.model.served_name,
            "quant": rec.quant,
            "min_vram_mb": rec.min_vram_mb,
            "notes": rec.notes,
            "download_repo": rec.model.repo,
            "download_glob": rec.download_glob,
            "path": rec.model.path,
            "context_length": rec.launch.context_length,
            "fit": label,
        }
        row["example"] = is_example_recipe(row, apple=apple)
        rows.append(row)
    order = {"recommended": 0, "ok": 1, "tight": 2, "unknown": 3, "cpu": 4, "skip": 5}
    rows.sort(
        key=lambda r: (
            order.get(r["fit"], 9),
            1 if r["example"] else 0,
            -int(r["min_vram_mb"] or 0),
            r["file"],
        )
    )
    best = next((r for r in rows if r["fit"] == "recommended"), None)
    return {
        "vram_mb": have,
        "apple": apple,
        "best": best["file"] if best else None,
        "recipes": rows,
    }
