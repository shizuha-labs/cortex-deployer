from __future__ import annotations

import unittest
from unittest.mock import patch

from cortex_deployer.recommend import fit_label, recommend


class RecommendTests(unittest.TestCase):
    def test_fit_labels(self):
        self.assertEqual(fit_label(14000, 16000, apple=False), "recommended")
        self.assertEqual(fit_label(22000, 16000, apple=False), "tight")
        self.assertEqual(fit_label(22000, 8000, apple=False), "skip")
        self.assertEqual(fit_label(14000, 0, apple=False), "cpu")

    def test_16gb_prefers_q3(self):
        snap = {
            "gpus": [{"vendor": "nvidia", "memory_mb": 16376, "name": "RTX 5080"}],
            "os": "Windows",
        }
        with patch("cortex_deployer.recommend.hostinfo.snapshot", return_value=snap):
            out = recommend()
        self.assertEqual(out["vram_mb"], 16376)
        self.assertEqual(out["best"], "qwen38-27b-q3-llamacpp.yaml")
        fits = {r["file"]: r["fit"] for r in out["recipes"]}
        self.assertEqual(fits["qwen38-27b-q3-llamacpp.yaml"], "recommended")
        self.assertIn(fits["qwen38-27b-llamacpp.yaml"], {"tight", "skip"})
        self.assertNotEqual(out["best"], "sglang-openai.yaml")
        self.assertNotEqual(out["best"], "vllm-openai.yaml")

    def test_24gb_prefers_q4(self):
        snap = {
            "gpus": [{"vendor": "nvidia", "memory_mb": 24576, "name": "RTX 3090"}],
            "os": "Linux",
        }
        with patch("cortex_deployer.recommend.hostinfo.snapshot", return_value=snap):
            out = recommend()
        self.assertEqual(out["best"], "qwen38-27b-llamacpp.yaml")

    def test_apple_prefers_mlx_macos(self):
        snap = {
            "gpus": [{"vendor": "apple", "memory_mb": 0, "name": "M4 (Metal)"}],
            "os": "Darwin",
        }
        with patch("cortex_deployer.recommend.hostinfo.snapshot", return_value=snap):
            out = recommend()
        self.assertTrue(out["apple"])
        self.assertEqual(out["best"], "mlx-macos.yaml")
