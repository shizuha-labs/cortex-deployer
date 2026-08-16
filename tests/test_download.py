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

    def test_guess_filenames_skips_api(self):
        names = download.guess_filenames(
            "unsloth/Qwen3.8-27B-GGUF", "*UD-Q2_K_XL.gguf"
        )
        self.assertIn("Qwen3.8-27B-UD-Q2_K_XL.gguf", names)
        resolved = download.resolve_names(
            "unsloth/Qwen3.8-27B-GGUF",
            filename="",
            glob="*UD-Q2_K_XL.gguf",
        )
        self.assertEqual(resolved[0], "Qwen3.8-27B-UD-Q2_K_XL.gguf")
        exact = download.resolve_names(
            "unsloth/Qwen3.8-27B-GGUF",
            filename="Qwen3.8-27B-UD-Q2_K_XL.gguf",
        )
        self.assertEqual(exact, ["Qwen3.8-27B-UD-Q2_K_XL.gguf"])

    def test_friendly_rate_limit(self):
        from urllib.error import HTTPError
        from io import BytesIO

        err = HTTPError(
            "https://huggingface.co/api/models/x",
            403,
            "rate limit exceeded",
            hdrs=None,
            fp=BytesIO(b""),
        )
        msg = download._friendly_hf_error(err)
        self.assertIn("Download blocked", msg)
        self.assertIn("403", msg)
        self.assertNotIn("settings/tokens", msg)

    def test_resolve_names_does_not_call_api_when_guessed(self):
        called = {"n": 0}

        def boom(*_a, **_k):
            called["n"] += 1
            raise AssertionError("list_hf_files should not run")

        orig = download.list_hf_files
        download.list_hf_files = boom  # type: ignore[method-assign]
        self.addCleanup(lambda: setattr(download, "list_hf_files", orig))
        names = download.resolve_names(
            "unsloth/Qwen3.8-27B-GGUF", glob="*UD-Q3_K_XL.gguf"
        )
        self.assertEqual(called["n"], 0)
        self.assertIn("Qwen3.8-27B-UD-Q3_K_XL.gguf", names)

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

    def test_candidate_urls_include_mirror(self):
        urls = download._candidate_urls(
            "unsloth/Qwen3.8-27B-GGUF", "Qwen3.8-27B-UD-Q3_K_XL.gguf"
        )
        self.assertTrue(any("huggingface.co" in u for u in urls))
        self.assertTrue(any("hf-mirror.com" in u for u in urls))
