"""GPU vs RAM-offload placement for llama.cpp.

Pinning --n-gpu-layers 99 disables llama.cpp --fit, so anything that does
not fully fill the card is marked skip and never launched. Offload keeps
KV + as many layers as fit in VRAM and puts the rest in system RAM.
"""

from __future__ import annotations

from typing import Any


# Below this, even KV + embeddings will not stay on the GPU usefully.
_OFFLOAD_FLOOR_MB = 3500


def offload_floor_mb(
    min_vram_mb: int,
    *,
    context_length: int = 0,
    min_offload_vram_mb: int = 0,
) -> int:
    if min_offload_vram_mb > 0:
        return int(min_offload_vram_mb)
    ctx = int(context_length or 0)
    # KV-ish floor: ~0.04 MB/token + 1.8 GB embeddings/CUDA. 8k→2.1G, 64k→4.4G.
    from_ctx = 1800 + int(ctx * 0.04) if ctx else _OFFLOAD_FLOOR_MB
    from_full = int(min_vram_mb * 0.28) if min_vram_mb else 0
    return max(_OFFLOAD_FLOOR_MB, from_ctx, from_full)


def fit_label(
    min_vram_mb: int,
    have_mb: int,
    *,
    apple: bool = False,
    engine: str = "",
    context_length: int = 0,
    min_offload_vram_mb: int = 0,
) -> str:
    if engine == "mlx":
        return "recommended" if apple else "skip"
    if apple:
        return "ok"
    if min_vram_mb <= 0:
        return "unknown"
    if have_mb <= 0:
        return "cpu"
    if have_mb >= min_vram_mb:
        return "recommended"
    floor = offload_floor_mb(
        min_vram_mb,
        context_length=context_length,
        min_offload_vram_mb=min_offload_vram_mb,
    )
    if have_mb >= floor:
        return "offload"
    return "skip"


def apply_ngl_args(
    extra: list[str],
    fit: str,
    context_length: int | None = None,
) -> list[str]:
    """For offload: drop a pinned ngl so llama.cpp --fit can split layers."""
    if fit != "offload":
        return list(extra)
    out: list[str] = []
    skip_next = False
    for arg in extra:
        if skip_next:
            skip_next = False
            continue
        if arg in {"--n-gpu-layers", "-ngl", "--gpu-layers"}:
            skip_next = True
            continue
        out.append(arg)
    if "--fit" not in out and "-fit" not in out:
        out.extend(["--fit", "on"])
    ctx = int(context_length or 0)
    if ctx and "--fit-ctx" not in out and "-fitc" not in out:
        out.extend(["--fit-ctx", str(ctx)])
    return out


def ram_hint_gb(weight_gb: float | int | None) -> float | None:
    if not weight_gb:
        return None
    return round(float(weight_gb) + 4.0, 1)


def annotate_quant(
    quant: dict[str, Any],
    have_mb: int,
    *,
    apple: bool,
) -> dict[str, Any]:
    item = dict(quant)
    engine = str(item.get("engine") or "")
    fit = fit_label(
        int(item.get("min_vram_mb") or 0),
        have_mb,
        apple=apple,
        engine=engine,
        context_length=int(item.get("context_length") or 0),
        min_offload_vram_mb=int(item.get("min_offload_vram_mb") or 0),
    )
    item["fit"] = fit
    item["placement"] = "offload" if fit == "offload" else ("gpu" if fit in {"recommended", "ok"} else fit)
    hint = ram_hint_gb(item.get("weight_gb"))
    if fit == "offload" and hint:
        item["min_ram_gb"] = hint
        extra = f"GPU+RAM offload · wants ~{hint} GB system RAM"
        notes = str(item.get("notes") or "").strip()
        if "offload" not in notes.lower():
            item["notes"] = f"{extra}. {notes}".strip()
    return item
