from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from cortex_deployer.recommend import fit_label, recommend


class RecommendTests(unittest.TestCase):
    def setUp(self):
        os.environ["CORTEX_DEPLOYER_CATALOG_URL"] = "bundled"
        self.addCleanup(lambda: os.environ.pop("CORTEX_DEPLOYER_CATALOG_URL", None))

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
        qwen = next(m for m in out["models"] if m["id"] == "qwen3.8-27b")
        self.assertEqual(qwen["recommended_recipe"], "qwen38-27b-q3-llamacpp.yaml")
        fits = {r["file"]: r["fit"] for r in out["recipes"]}
        self.assertEqual(fits["qwen38-27b-q3-llamacpp.yaml"], "recommended")
        self.assertEqual(fits["qwen38-27b-q2-llamacpp.yaml"], "ok")
        self.assertIn(fits["qwen38-27b-llamacpp.yaml"], {"tight", "skip"})
        qfits = {q["id"]: q["fit"] for q in qwen["quants"]}
        self.assertEqual(qwen["recommended_quant"], "qwen38-27b-q3")
        self.assertEqual(qfits["qwen38-27b-q3"], "recommended")
        self.assertEqual(qfits["qwen38-27b-q2"], "ok")
        self.assertEqual(qfits["qwen38-27b-q4"], "tight")
        rec_quants = [q["id"] for q in qwen["quants"] if q["fit"] == "recommended"]
        self.assertEqual(rec_quants, ["qwen38-27b-q3"])
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
        qwen = next(m for m in out["models"] if m["id"] == "qwen3.8-27b")
        self.assertEqual(qwen["recommended_quant"], "qwen38-27b-q4")
        qfits = {q["id"]: q["fit"] for q in qwen["quants"]}
        self.assertEqual(qfits["qwen38-27b-q4"], "recommended")
        self.assertEqual(qfits["qwen38-27b-q3"], "ok")
        self.assertEqual(qfits["qwen38-27b-q2"], "ok")

    def test_apple_prefers_mlx_macos(self):
        snap = {
            "gpus": [{"vendor": "apple", "memory_mb": 0, "name": "M4 (Metal)"}],
            "os": "Darwin",
        }
        with patch("cortex_deployer.recommend.hostinfo.snapshot", return_value=snap):
            out = recommend()
        self.assertTrue(out["apple"])
        self.assertEqual(out["best"], "mlx-macos.yaml")
