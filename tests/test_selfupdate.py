from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from cortex_deployer import __version__, selfupdate


class SelfUpdateTests(unittest.TestCase):
    def test_catalog_helpers(self):
        cat = {
            "deployer_release": {
                "version": "9.9.9",
                "tarball": "https://example.test/app.tar.gz",
            }
        }
        self.assertEqual(selfupdate.latest_from_catalog(cat), "9.9.9")
        self.assertEqual(
            selfupdate.tarball_from_catalog(cat), "https://example.test/app.tar.gz"
        )
        self.assertEqual(selfupdate.tarball_from_catalog({}), selfupdate.DEFAULT_TARBALL)
        self.assertTrue(selfupdate.update_available("9.9.9"))
        self.assertFalse(selfupdate.update_available(__version__))
        self.assertFalse(selfupdate.update_available(""))
        self.assertFalse(selfupdate.update_available("0.0.1"))
        self.assertFalse(selfupdate.update_available("0.3.9"))

    def test_write_updater_files(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        os.environ["CORTEX_DEPLOYER_HOME"] = tmp.name
        self.addCleanup(lambda: os.environ.pop("CORTEX_DEPLOYER_HOME", None))
        root = Path(tmp.name)
        (root / "bin").mkdir()
        (root / "venv" / "bin").mkdir(parents=True)
        uv = root / "bin" / "uv"
        py = root / "venv" / "bin" / "python"
        uv.write_text("#!/bin/sh\n", encoding="utf-8")
        py.write_text("#!/bin/sh\n", encoding="utf-8")
        uv.chmod(0o755)
        py.chmod(0o755)
        script, cfg = selfupdate.write_updater_files(
            selfupdate.DEFAULT_TARBALL, "127.0.0.1", 7481
        )
        data = json.loads(cfg.read_text(encoding="utf-8"))
        self.assertEqual(data["port"], 7481)
        self.assertEqual(data["host"], "127.0.0.1")
        self.assertEqual(data["uv"], str(uv))
        text = script.read_text(encoding="utf-8")
        self.assertIn("pip", text)
        self.assertIn("cortex_deployer", text)
        compile(text, str(script), "exec")

    def test_update_cmd_falls_back_to_pip(self):
        os.environ["CORTEX_DEPLOYER_HOME"] = "/tmp/cd-no-venv-here"
        self.addCleanup(lambda: os.environ.pop("CORTEX_DEPLOYER_HOME", None))
        cmd = selfupdate._update_cmd(selfupdate.DEFAULT_TARBALL)
        self.assertEqual(cmd[1:4], ["-m", "pip", "install"])

    def test_auto_update_persist(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        os.environ["CORTEX_DEPLOYER_HOME"] = tmp.name
        self.addCleanup(lambda: os.environ.pop("CORTEX_DEPLOYER_HOME", None))
        os.environ.pop("CORTEX_DEPLOYER_AUTO_UPDATE", None)
        self.assertFalse(selfupdate.auto_update_enabled())
        selfupdate.set_auto_update(True)
        self.assertTrue(selfupdate.auto_update_enabled())
        saved = json.loads((Path(tmp.name) / "config.json").read_text(encoding="utf-8"))
        self.assertTrue(saved["auto_update"])
        selfupdate.set_auto_update(False)
        self.assertFalse(selfupdate.auto_update_enabled())
        os.environ["CORTEX_DEPLOYER_AUTO_UPDATE"] = "1"
        self.addCleanup(lambda: os.environ.pop("CORTEX_DEPLOYER_AUTO_UPDATE", None))
        self.assertTrue(selfupdate.auto_update_enabled())
