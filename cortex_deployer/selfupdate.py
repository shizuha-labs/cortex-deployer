"""Upgrade the isolated install in-place. No OS pip, no extra user tokens."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from . import __version__
from .paths import home_dir

DEFAULT_TARBALL = (
    "https://github.com/shizuha-labs/cortex-deployer/archive/refs/heads/main.tar.gz"
)

# Standalone: the running server exits before this replaces package files.
_UPDATER = """\
import json
import os
import subprocess
import sys
import time
from pathlib import Path

cfg = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
log = Path(cfg["log"])
time.sleep(float(cfg.get("delay", 1.2)))
env = os.environ.copy()
env["UV_PYTHON_PREFERENCE"] = "only-managed"
try:
    proc = subprocess.run(
        [cfg["uv"], "pip", "install", "--python", cfg["py"], "--upgrade", cfg["tarball"]],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    log.write_text((proc.stdout or "") + "\\n" + (proc.stderr or ""), encoding="utf-8")
except Exception as exc:
    log.write_text(str(exc), encoding="utf-8")

kwargs = {"close_fds": True}
if os.name == "nt":
    kwargs["creationflags"] = (
        getattr(subprocess, "DETACHED_PROCESS", 0)
        | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    )
else:
    kwargs["start_new_session"] = True
subprocess.Popen(
    [cfg["py"], "-m", "cortex_deployer", "server", "--host", cfg["host"], "--port", str(cfg["port"])],
    **kwargs,
)
"""


def _venv_python() -> Path:
    root = home_dir() / "venv"
    if os.name == "nt":
        return root / "Scripts" / "python.exe"
    return root / "bin" / "python"


def _uv_bin() -> Path | None:
    name = "uv.exe" if os.name == "nt" else "uv"
    bundled = home_dir() / "bin" / name
    if bundled.is_file():
        return bundled
    found = shutil.which("uv")
    return Path(found) if found else None


def config_path() -> Path:
    return home_dir() / "config.json"


def load_config() -> dict[str, Any]:
    path = config_path()
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def save_config(patch: dict[str, Any]) -> dict[str, Any]:
    cfg = load_config()
    cfg.update(patch)
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    return cfg


def auto_update_enabled() -> bool:
    env = os.environ.get("CORTEX_DEPLOYER_AUTO_UPDATE", "").strip().lower()
    if env in {"1", "true", "yes"}:
        return True
    if env in {"0", "false", "no"}:
        return False
    return bool(load_config().get("auto_update"))


def set_auto_update(on: bool) -> None:
    save_config({"auto_update": bool(on)})


def latest_from_catalog(cat: dict[str, Any] | None) -> str:
    rel = (cat or {}).get("deployer_release") or {}
    return str(rel.get("version") or "").strip()


def tarball_from_catalog(cat: dict[str, Any] | None) -> str:
    rel = (cat or {}).get("deployer_release") or {}
    return str(rel.get("tarball") or "").strip() or DEFAULT_TARBALL


def _version_tuple(s: str) -> tuple[int, ...]:
    nums: list[int] = []
    for part in str(s).split("."):
        digits = ""
        for ch in part:
            if ch.isdigit():
                digits += ch
            else:
                break
        nums.append(int(digits or 0))
    return tuple(nums or (0,))


def update_available(latest: str) -> bool:
    """True only when catalog is a newer dotted version (never downgrade)."""
    if not latest:
        return False
    return _version_tuple(latest) > _version_tuple(__version__)


def _update_cmd(tarball: str) -> list[str]:
    uv = _uv_bin()
    py = _venv_python()
    if uv is not None and py.is_file():
        return [str(uv), "pip", "install", "--python", str(py), "--upgrade", tarball]
    return [sys.executable, "-m", "pip", "install", "--upgrade", tarball]


def run_update(tarball: str = "") -> dict[str, Any]:
    """In-process pip install. Isolated venv if present, else this interpreter."""
    src = tarball or DEFAULT_TARBALL
    cmd = _update_cmd(src)
    env = os.environ.copy()
    env["UV_PYTHON_PREFERENCE"] = "only-managed"
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "update failed").strip()
        raise RuntimeError(err[-800:])
    return {
        "ok": True,
        "previous": __version__,
        "restart_required": True,
        "detail": (proc.stdout or "").strip()[-400:],
    }


def apply_on_start(*, auto: bool, host: str, port: int) -> None:
    """Print or apply a catalog update before the server binds.

    Re-exec is one-shot (CORTEX_DEPLOYER_UPDATED) so a no-op pip cannot loop.
    Never raises — a catalog/network miss must not block listen.
    """
    if os.environ.get("CORTEX_DEPLOYER_UPDATED") == "1":
        return
    try:
        from . import catalog

        cat = catalog.fetch_catalog(force=True, timeout=5.0)
        latest = latest_from_catalog(cat)
        if not update_available(latest):
            return
        if not auto:
            print(
                f"update available  {__version__} → {latest}  ·  cortex-deployer update",
                flush=True,
            )
            return
        print(f"auto-update  {__version__} → {latest}", flush=True)
        run_update(tarball_from_catalog(cat))
        os.environ["CORTEX_DEPLOYER_UPDATED"] = "1"
        os.execv(
            sys.executable,
            [sys.executable, "-m", "cortex_deployer", "server", "--host", host, "--port", str(port)],
        )
    except Exception as exc:
        print(f"auto-update skipped: {exc}", file=sys.stderr, flush=True)


def write_updater_files(tarball: str, host: str, port: int) -> tuple[Path, Path]:
    tmp = home_dir() / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    script = tmp / "apply-update.py"
    cfg_path = tmp / "update.json"
    uv = _uv_bin()
    py = _venv_python()
    if uv is None:
        raise RuntimeError(
            "isolated uv not found; re-run https://cortex.shizuha.com/deployer/install.sh"
        )
    if not py.is_file():
        raise RuntimeError(f"venv python missing: {py}")
    payload = {
        "uv": str(uv),
        "py": str(py),
        "tarball": tarball or DEFAULT_TARBALL,
        "host": host,
        "port": int(port),
        "log": str(home_dir() / "update.log"),
        "delay": 1.2,
    }
    cfg_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    script.write_text(_UPDATER.lstrip() + "\n", encoding="utf-8")
    return script, cfg_path


def spawn_updater(tarball: str, host: str, port: int) -> dict[str, Any]:
    """Replace the package after this process exits, then start the server again."""
    script, cfg_path = write_updater_files(tarball, host, port)
    cmd = [str(_venv_python()), str(script), str(cfg_path)]
    kwargs: dict[str, Any] = {"close_fds": True}
    if os.name == "nt":
        kwargs["creationflags"] = (
            getattr(subprocess, "DETACHED_PROCESS", 0)
            | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        )
    else:
        kwargs["start_new_session"] = True
    subprocess.Popen(cmd, **kwargs)
    return {
        "ok": True,
        "previous": __version__,
        "restarting": True,
        "script": str(script),
    }


def spawn_restart(host: str, port: int) -> None:
    cmd = [
        str(_venv_python()),
        "-m",
        "cortex_deployer",
        "server",
        "--host",
        host,
        "--port",
        str(port),
    ]
    kwargs: dict[str, Any] = {"close_fds": True}
    if os.name == "nt":
        kwargs["creationflags"] = (
            getattr(subprocess, "DETACHED_PROCESS", 0)
            | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        )
    else:
        kwargs["start_new_session"] = True
    subprocess.Popen(cmd, **kwargs)
