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


class GpuDetectionTests(unittest.TestCase):
    """CTX-754: detect_gpus() must parse nvidia-smi and Apple sysctl output
    without a live GPU (mocked)."""

    def test_nvidia_smi_parses_vram(self):
        from cortex_deployer import hostinfo

        smi_out = (
            "0, NVIDIA GeForce RTX 5080, 16376, 580.0\n"
            "1, NVIDIA GeForce RTX 3090, 24576, 580.0\n"
        )
        with (
            patch.object(hostinfo.shutil, "which", return_value="/usr/bin/nvidia-smi"),
            patch.object(hostinfo, "_run", return_value=smi_out),
            patch.object(hostinfo.platform, "system", return_value="Linux"),
        ):
            gpus = hostinfo.detect_gpus()
        self.assertEqual(len(gpus), 2)
        self.assertEqual(gpus[0]["vendor"], "nvidia")
        self.assertEqual(gpus[0]["memory_mb"], 16376)
        self.assertEqual(gpus[0]["name"], "NVIDIA GeForce RTX 5080")
        self.assertEqual(gpus[1]["memory_mb"], 24576)

    def test_no_nvidia_smi_returns_empty_on_linux(self):
        from cortex_deployer import hostinfo

        with (
            patch.object(hostinfo.shutil, "which", return_value=None),
            patch.object(hostinfo.platform, "system", return_value="Linux"),
        ):
            gpus = hostinfo.detect_gpus()
        self.assertEqual(gpus, [])

    def test_apple_silicon_reports_unified_memory(self):
        from cortex_deployer import hostinfo

        # 16 GiB unified memory = 17179869184 bytes.
        with (
            patch.object(hostinfo.shutil, "which", return_value=None),
            patch.object(hostinfo.platform, "system", return_value="Darwin"),
            patch.object(
                hostinfo, "_run",
                side_effect=lambda cmd: (
                    "Apple M4 Pro" if "brand_string" in cmd else "17179869184"
                ),
            ),
        ):
            gpus = hostinfo.detect_gpus()
        self.assertEqual(len(gpus), 1)
        self.assertEqual(gpus[0]["vendor"], "apple")
        self.assertEqual(gpus[0]["memory_mb"], 16384)
        self.assertEqual(gpus[0]["driver"], "metal")
