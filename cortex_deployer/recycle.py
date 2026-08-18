"""Idle Metal-cap recycle — recover a wedged Rapid-MLX without touching busy work.

Rapid-MLX 0.12.x can pin Metal above ``gpu_memory_utilization`` after
completed/aborted streams (D-METAL-CAP). Prefix-cache bytes report 0 and
``/v1/cache/clear`` is a no-op, so every new generate 503s until the
process is replaced. The only safe recycle is the same-recipe engine
restart **while ``requests_running=0`` and ``requests_waiting=0``**.

This module is the detector. The connect process (always-on next to the
engine) polls ``/metrics`` and runs an operator-configured command
(typically ``launchctl kickstart -k`` of the engine launchd). It never
raises the memory cap and never recycles a live decode.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass


DEFAULT_POLL_S = 15.0
DEFAULT_COOLDOWN_S = 900.0
DEFAULT_MIN_HITS = 2

_RUNNING = "rapid_mlx_requests_running"
_WAITING = "rapid_mlx_requests_waiting"
_VIOLATIONS = "rapid_mlx_metal_cap_violations_total"
_METAL = "rapid_mlx_metal_active_memory_bytes"


@dataclass
class EngineSnapshot:
    running: float | None = None
    waiting: float | None = None
    metal_active: float | None = None
    cap_violations: float | None = None


@dataclass
class RecycleState:
    last_violations: float | None = None
    hits: int = 0
    last_recycle_mono: float = 0.0


def recycle_cmd_from_env(explicit: str = "") -> str:
    return (explicit or os.environ.get("CORTEX_DEPLOYER_RECYCLE_CMD") or "").strip()


def poll_seconds() -> float:
    raw = os.environ.get("CORTEX_DEPLOYER_RECYCLE_POLL_S", "").strip()
    try:
        return max(5.0, float(raw)) if raw else DEFAULT_POLL_S
    except ValueError:
        return DEFAULT_POLL_S


def cooldown_seconds() -> float:
    raw = os.environ.get("CORTEX_DEPLOYER_RECYCLE_COOLDOWN_S", "").strip()
    try:
        return max(60.0, float(raw)) if raw else DEFAULT_COOLDOWN_S
    except ValueError:
        return DEFAULT_COOLDOWN_S


def min_hits() -> int:
    raw = os.environ.get("CORTEX_DEPLOYER_RECYCLE_MIN_HITS", "").strip()
    try:
        return max(1, int(raw)) if raw else DEFAULT_MIN_HITS
    except ValueError:
        return DEFAULT_MIN_HITS


def _prom_unlabeled(text: str, name: str) -> float | None:
    """First unlabeled sample for ``name`` (ignore HELP/TYPE and labeled rows)."""
    prefix = name + " "
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(prefix):
            try:
                return float(line[len(prefix) :].split()[0])
            except (TypeError, ValueError, IndexError):
                return None
    return None


def parse_metrics(text: str) -> EngineSnapshot:
    return EngineSnapshot(
        running=_prom_unlabeled(text, _RUNNING),
        waiting=_prom_unlabeled(text, _WAITING),
        metal_active=_prom_unlabeled(text, _METAL),
        cap_violations=_prom_unlabeled(text, _VIOLATIONS),
    )


def should_recycle(
    snap: EngineSnapshot,
    state: RecycleState,
    *,
    now: float,
    min_hits_needed: int = DEFAULT_MIN_HITS,
    cooldown_s: float = DEFAULT_COOLDOWN_S,
) -> tuple[bool, str]:
    """Return (recycle?, reason). Mutates ``state``.

    Fire only when the engine is idle AND new D-METAL-CAP rejects are
    landing — that is the permanent-503 wedge. A high Metal gauge with
    successful completions is not a recycle.
    """
    if snap.running is None and snap.cap_violations is None:
        return False, "no-metrics"
    if (snap.running or 0.0) > 0.0 or (snap.waiting or 0.0) > 0.0:
        state.hits = 0
        if snap.cap_violations is not None:
            state.last_violations = snap.cap_violations
        return False, "busy"
    if state.last_recycle_mono and (now - state.last_recycle_mono) < cooldown_s:
        if snap.cap_violations is not None:
            state.last_violations = snap.cap_violations
        return False, "cooldown"
    if snap.cap_violations is None:
        return False, "no-violations"
    if state.last_violations is None:
        state.last_violations = snap.cap_violations
        return False, "prime"
    if snap.cap_violations > state.last_violations:
        state.hits += 1
        state.last_violations = snap.cap_violations
        if state.hits >= min_hits_needed:
            return True, "idle-metal-cap"
        return False, "accumulating"
    state.hits = 0
    state.last_violations = snap.cap_violations
    return False, "stable"


def run_recycle_cmd(cmd: str) -> int:
    if not cmd.strip():
        return 0
    proc = subprocess.run(cmd, shell=True, check=False)  # noqa: S602 — operator-configured
    return int(proc.returncode)
