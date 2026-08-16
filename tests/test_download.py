from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from cortex_deployer import download


class DownloadTests(unittest.TestCase):
    def test_list_local(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        os.environ["CORTEX_DEPLOYER_MODELS"] = tmp.name
        self.addCleanup(lambda: os.environ.pop("CORTEX_DEPLOYER_MODELS", None))
        Path(tmp.name, "toy.gguf").write_bytes(b"gguf")
        rows = download.list_local_models()
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]["path"].endswith("toy.gguf"))

    def test_start_download_rejects_bad_repo(self):
        with self.assertRaises(ValueError):
            download.start_download("nopath")

    def test_start_download_requires_glob_or_filename(self):
        with self.assertRaises(ValueError):
            download.start_download("unsloth/Qwen3.8-27B-GGUF")

    def test_select_weight_files(self):
        names = [
            "README.md",
            "config.json",
            ".gitattributes",
            "mmproj-F16.gguf",
            "Qwen3.8-27B-UD-Q3_K_XL.gguf",
            "Qwen3.8-27B-UD-Q4_K_XL.gguf",
            "BF16/Qwen3.8-27B-BF16-00001-of-00002.gguf",
        ]
        picked = download.select_weight_files(names, "*UD-Q3_K_XL.gguf")
        self.assertEqual(picked, ["Qwen3.8-27B-UD-Q3_K_XL.gguf"])
        all_gguf = download.select_weight_files(names, "")
        self.assertIn("Qwen3.8-27B-UD-Q3_K_XL.gguf", all_gguf)
        self.assertNotIn("mmproj-F16.gguf", all_gguf)
        self.assertNotIn("README.md", all_gguf)
