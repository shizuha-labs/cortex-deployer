from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cortex_deployer.enginebin import (
    find_extracted_server,
    pick_release_assets,
)


class PickReleaseAssetsTests(unittest.TestCase):
    NAMES = [
        "llama-b10453-bin-win-cuda-12.4-x64.zip",
        "llama-b10453-bin-win-cuda-13.3-x64.zip",
        "llama-b10453-bin-win-cpu-x64.zip",
        "cudart-llama-bin-win-cuda-12.4-x64.zip",
        "cudart-llama-bin-win-cuda-13.3-x64.zip",
        "llama-b10453-bin-macos-arm64.tar.gz",
        "llama-b10453-bin-ubuntu-x64.tar.gz",
        "llama-b10453-bin-ubuntu-vulkan-x64.tar.gz",
    ]

    def test_windows_x64_prefers_cuda_13(self):
        llama, cudart = pick_release_assets(
            self.NAMES, system="Windows", machine="AMD64"
        )
        self.assertEqual(llama, "llama-b10453-bin-win-cuda-13.3-x64.zip")
        self.assertEqual(cudart, "cudart-llama-bin-win-cuda-13.3-x64.zip")

    def test_macos_arm(self):
        llama, cudart = pick_release_assets(
            self.NAMES, system="Darwin", machine="arm64"
        )
        self.assertEqual(llama, "llama-b10453-bin-macos-arm64.tar.gz")
        self.assertIsNone(cudart)

    def test_linux_skips_vulkan(self):
        llama, cudart = pick_release_assets(
            self.NAMES, system="Linux", machine="x86_64"
        )
        self.assertEqual(llama, "llama-b10453-bin-ubuntu-x64.tar.gz")
        self.assertIsNone(cudart)


class ExtractedServerTests(unittest.TestCase):
    def test_finds_exe_in_nested_zip_layout(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        dest = root / "out"
        dest.mkdir()
        nested = dest / "bin"
        nested.mkdir()
        (nested / "llama-server.exe").write_bytes(b"mz")
        self.assertEqual(find_extracted_server(dest).name, "llama-server.exe")
