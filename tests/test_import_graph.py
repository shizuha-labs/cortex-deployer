"""The leaf package must not import Django, Cortex ORM, or Kubernetes."""

from __future__ import annotations

import ast
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "cortex_deployer"

BANNED = {
    "django",
    "rest_framework",
    "kubernetes",
    "inference",
    "cortex_project",
}


def _module_root(name: str) -> str:
    return name.split(".", 1)[0]


class ImportGraphTests(unittest.TestCase):
    def test_django_settings_unset(self):
        self.assertFalse(os.environ.get("DJANGO_SETTINGS_MODULE"))

    def test_source_has_no_banned_imports(self):
        hits: list[str] = []
        for path in PKG.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                names: list[str] = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                for name in names:
                    if _module_root(name) in BANNED:
                        hits.append(f"{path.relative_to(ROOT)}: {name}")
        self.assertEqual(hits, [])

    def test_import_does_not_load_banned_modules(self):
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        before = set(sys.modules)
        import cortex_deployer  # noqa: F401
        from cortex_deployer.client import relay_request  # noqa: F401
        from cortex_deployer.engines import render_process  # noqa: F401

        loaded = {
            _module_root(name)
            for name in sys.modules
            if name not in before
        }
        self.assertFalse(loaded & BANNED, loaded & BANNED)
