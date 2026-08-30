"""Pick recipes that fit detected VRAM (LM Studio-style GPU fit)."""

from __future__ import annotations

from typing import Any

from . import catalog as catalog_mod
from . import hostinfo
from .placement import annotate_quant, fit_label
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


def is_example_recipe(row: dict[str, Any], *, apple: bool) -> bool:
    fname = str(row.get("file") or "")
    if fname.startswith("qwen") or fname.startswith("minimax"):
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
        label = fit_label(rec.min_vram_mb, have, apple=apple, engine=rec.engine)
        if rec.engine in {"vllm", "sglang"} and have < 8000 and not apple:
            label = "skip" if have < 6000 else "tight"
        row = {
            "file": path.name,
            "name": rec.name,
            "engine": rec.engine,
            "served_name": rec.model.served_name,
            "quant": rec.quant,
            "min_vram_mb": rec.min_vram_mb,
            "notes": rec.notes,
            "download_repo": rec.model.repo,
            "download_filename": rec.model.filename,
            "download_glob": rec.download_glob,
            "path": rec.model.path,
            "context_length": rec.launch.context_length,
            "fit": label,
        }
        row["example"] = is_example_recipe(row, apple=apple)
        rows.append(row)
    cat = catalog_mod.fetch_catalog()
    tier = pick_tier(cat.get("hardware_tiers") or [], have, apple=apple)
    wanted = ((tier or {}).get("recommend") or {})
    default_id = str(wanted.get("default") or "")
    models: list[dict[str, Any]] = []
    for model in cat.get("models") or []:
        if not isinstance(model, dict):
            continue
        quants: list[dict[str, Any]] = []
        for quant in model.get("quants") or []:
            if not isinstance(quant, dict):
                continue
            quants.append(annotate_quant(quant, have, apple=apple))
        prefer = str(wanted.get(str(model.get("id") or "")) or "")
        pick = next((q for q in quants if q.get("id") == prefer), None)
        if pick is None or pick.get("fit") in {"skip", "cpu"}:
            fits = [q for q in quants if q.get("fit") == "recommended"]
            pick = max(fits, key=lambda q: int(q.get("min_vram_mb") or 0)) if fits else None
        if pick is None:
            offs = [q for q in quants if q.get("fit") == "offload"]
            pick = min(offs, key=lambda q: int(q.get("min_vram_mb") or 0)) if offs else None
        for quant in quants:
            if (
                quant.get("fit") == "recommended"
                and pick
                and quant.get("id") != pick.get("id")
            ):
                quant["fit"] = "ok"
        models.append({
            "id": model.get("id"),
            "name": model.get("name"),
            "summary": model.get("summary"),
            "params": model.get("params"),
            "role": model.get("role") or "",
            "quants": quants,
            "recommended_quant": (pick or {}).get("id"),
            "recommended_recipe": (pick or {}).get("recipe"),
        })
    default_recipe = ""
    for model in models:
        if model.get("recommended_quant") == default_id:
            default_recipe = str(model.get("recommended_recipe") or "")
            break
    order = {"recommended": 0, "ok": 1, "offload": 2, "tight": 3, "unknown": 4, "cpu": 5, "skip": 6}
    rows.sort(
        key=lambda r: (
            0 if default_recipe and r["file"] == default_recipe else 1,
            order.get(r["fit"], 9),
            1 if r["example"] else 0,
            -int(r["min_vram_mb"] or 0),
            r["file"],
        )
    )
    best = next((r for r in rows if default_recipe and r["file"] == default_recipe), None)
    if best is None:
        best = next((r for r in rows if r["fit"] == "recommended" and not r.get("example")), None)
    if best is None:
        best = next((r for r in rows if r["fit"] == "recommended"), None)
    if best and not apple:
        for row in rows:
            if (
                row["fit"] == "recommended"
                and row["file"] != best["file"]
                and row.get("engine") != "mlx"
            ):
                row["fit"] = "ok"
    return {
        "vram_mb": have,
        "apple": apple,
        "tier": (tier or {}).get("id"),
        "best": best["file"] if best else None,
        "recipes": rows,
        "models": models,
        "catalog_source": cat.get("source"),
        "catalog_live": bool(cat.get("fetched")),
    }


def pick_tier(tiers: list[Any], have: int, *, apple: bool) -> dict[str, Any] | None:
    typed = [t for t in tiers if isinstance(t, dict)]
    if apple:
        return next((t for t in typed if t.get("id") == "apple"), None)
    nvs = [t for t in typed if int(t.get("vram_mb") or 0) > 0]
    if not nvs:
        return None
    # Prefer the largest tier that the card actually has, so 8–12 GB
    # does not snap to the 16 GB recommendations.
    under = [t for t in nvs if int(t.get("vram_mb") or 0) <= have + 2048]
    if under:
        return max(under, key=lambda t: int(t.get("vram_mb") or 0))
    return min(nvs, key=lambda t: abs(int(t.get("vram_mb") or 0) - have))
