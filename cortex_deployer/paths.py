"""Cross-platform data dirs. Windows %USERPROFILE%, macOS/Linux $HOME."""

from __future__ import annotations

import os
from pathlib import Path


def home_dir() -> Path:
    override = os.environ.get("CORTEX_DEPLOYER_HOME")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".cortex-deployer"


def state_path() -> Path:
    return home_dir() / "state.json"


def logs_dir() -> Path:
    return home_dir() / "logs"


def engines_dir() -> Path:
    return home_dir() / "engines" / "llamacpp"


def ensure_home() -> Path:
    root = home_dir()
    logs_dir().mkdir(parents=True, exist_ok=True)
    return root
