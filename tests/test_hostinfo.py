from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cortex_deployer.hostinfo import advertise_urls, default_bind_host, find_llama_server


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

    def test_linux_default_is_all_interfaces(self):
        if os.name == "nt":
            self.assertEqual(default_bind_host(), "127.0.0.1")
        else:
            self.assertEqual(default_bind_host(), "0.0.0.0")

    def test_advertise_urls_all_interfaces(self):
        urls = advertise_urls("0.0.0.0", 7480)
        self.assertTrue(urls[0].startswith("http://127.0.0.1:7480"))
        pinned = advertise_urls("127.0.0.1", 7480)
        self.assertEqual(pinned, ["http://127.0.0.1:7480/"])
