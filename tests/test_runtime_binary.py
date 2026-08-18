from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cortex_deployer.runtime import resolve_binary


class ResolveBinaryTests(unittest.TestCase):
    def test_generic_name_uses_env_path(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        real = Path(tmp.name) / "llama-server"
        real.write_text("", encoding="utf-8")
        os.environ["CORTEX_DEPLOYER_LLAMA_SERVER"] = str(real)
        self.addCleanup(lambda: os.environ.pop("CORTEX_DEPLOYER_LLAMA_SERVER", None))
        self.assertEqual(resolve_binary("llamacpp", "llama-server"), str(real))

    def test_existing_override_wins(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        real = Path(tmp.name) / "custom"
        real.write_text("", encoding="utf-8")
        self.assertEqual(resolve_binary("llamacpp", str(real)), str(real))

    def test_generic_name_uses_finder(self):
        with patch("cortex_deployer.hostinfo.find_llama_server", return_value="/opt/llama-server"):
            self.assertEqual(resolve_binary("llamacpp", "llama-server"), "/opt/llama-server")

    def test_mlx_lm_server_not_replaced_by_rapid_mlx_env(self):
        os.environ["CORTEX_DEPLOYER_MLX"] = "/opt/rapid-mlx"
        self.addCleanup(lambda: os.environ.pop("CORTEX_DEPLOYER_MLX", None))
        self.assertEqual(resolve_binary("mlx", "mlx_lm.server"), "mlx_lm.server")
