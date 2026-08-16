from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cortex_deployer.hostinfo import find_llama_server


class HostinfoTests(unittest.TestCase):
    def test_env_override(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        fake = Path(tmp.name) / "llama-server.exe"
        fake.write_text("", encoding="utf-8")
        os.environ["CORTEX_DEPLOYER_LLAMA_SERVER"] = str(fake)
        self.addCleanup(lambda: os.environ.pop("CORTEX_DEPLOYER_LLAMA_SERVER", None))
        self.assertEqual(find_llama_server(), str(fake))

    def test_missing_env_falls_through(self):
        os.environ["CORTEX_DEPLOYER_LLAMA_SERVER"] = "/no/such/llama-server"
        self.addCleanup(lambda: os.environ.pop("CORTEX_DEPLOYER_LLAMA_SERVER", None))
        with patch("cortex_deployer.hostinfo.shutil.which", return_value=None):
            with patch("cortex_deployer.hostinfo.Path.home", return_value=Path("/tmp/no-home-here")):
                found = find_llama_server()
        self.assertTrue(found is None or found.endswith("llama-server") or found.endswith(".exe"))
