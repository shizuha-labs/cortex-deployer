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
        self.assertEqual(fit_label(22000, 16000, apple=False), "offload")
        self.assertEqual(fit_label(22000, 8000, apple=False), "offload")
        self.assertEqual(fit_label(22000, 2000, apple=False), "skip")
        self.assertEqual(fit_label(14000, 0, apple=False), "cpu")

    def test_16gb_prefers_9b_longctx(self):
        snap = {
            "gpus": [{"vendor": "nvidia", "memory_mb": 16376, "name": "RTX 5080"}],
            "os": "Windows",
        }
        with patch("cortex_deployer.recommend.hostinfo.snapshot", return_value=snap):
            out = recommend()
        self.assertEqual(out["vram_mb"], 16376)
        self.assertEqual(out["tier"], "16gb")
        self.assertEqual(out["best"], "qwen35-9b-q6-llamacpp.yaml")
        qwen = next(m for m in out["models"] if m["id"] == "qwen3.8-27b")
        self.assertEqual(qwen["recommended_recipe"], "qwen38-27b-q2-llamacpp.yaml")
        qfits = {q["id"]: q["fit"] for q in qwen["quants"]}
        self.assertEqual(qwen["recommended_quant"], "qwen38-27b-q2")
        self.assertEqual(qfits["qwen38-27b-q2"], "recommended")
        self.assertEqual(qfits["qwen38-27b-q3"], "offload")
        self.assertEqual(qfits["qwen38-27b-q4"], "offload")
        nine = next(m for m in out["models"] if m["id"] == "qwen3.5-9b")
        self.assertEqual(nine["recommended_quant"], "qwen35-9b-q6")
        fourteen = next(m for m in out["models"] if m["id"] == "qwen3-14b")
        self.assertEqual(fourteen["recommended_quant"], "qwen3-14b-q4")
        fits = {r["file"]: r["fit"] for r in out["recipes"]}
        self.assertEqual(fits["qwen35-9b-q6-llamacpp.yaml"], "recommended")
        self.assertEqual(fits["qwen38-27b-q2-llamacpp.yaml"], "ok")
        self.assertEqual(fits["qwen38-27b-q3-llamacpp.yaml"], "offload")
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
        h3 = next(m for m in out["models"] if m["id"] == "minimax-h3")
        self.assertEqual(h3["recommended_quant"], "minimax-h3-fl2va")
        h3fits = {q["id"]: q["fit"] for q in h3["quants"]}
        self.assertEqual(h3fits["minimax-h3-fl2va"], "recommended")

    def test_apple_prefers_mlx_macos(self):
        snap = {
            "gpus": [{"vendor": "apple", "memory_mb": 0, "name": "M4 (Metal)"}],
            "os": "Darwin",
        }
        with patch("cortex_deployer.recommend.hostinfo.snapshot", return_value=snap):
            out = recommend()
        self.assertTrue(out["apple"])
        self.assertEqual(out["best"], "qwen38-27b-mlx.yaml")

    def test_8gb_full_gpu_8b_and_offload_rest(self):
        snap = {
            "gpus": [{"vendor": "nvidia", "memory_mb": 8192, "name": "RTX 4060"}],
            "os": "Linux",
        }
        with patch("cortex_deployer.recommend.hostinfo.snapshot", return_value=snap):
            out = recommend()
        self.assertEqual(out["tier"], "8gb")
        self.assertEqual(out["best"], "qwen3-8b-q5-llamacpp.yaml")
        eight = next(m for m in out["models"] if m["id"] == "qwen3-8b")
        self.assertEqual(eight["recommended_quant"], "qwen3-8b-q5")
        nine = next(m for m in out["models"] if m["id"] == "qwen3.5-9b")
        self.assertEqual(nine["recommended_quant"], "qwen35-9b-q4")
        qfits = {q["id"]: q["fit"] for q in nine["quants"]}
        self.assertEqual(qfits["qwen35-9b-q4"], "offload")
        self.assertEqual(qfits["qwen35-9b-q5"], "offload")
        qwen = next(m for m in out["models"] if m["id"] == "qwen3.8-27b")
        q27 = {q["id"]: q["fit"] for q in qwen["quants"]}
        self.assertEqual(q27["qwen38-27b-q2"], "offload")
        self.assertNotEqual(q27["qwen38-27b-q2"], "skip")
        fits = {r["file"]: r["fit"] for r in out["recipes"]}
        self.assertEqual(fits["qwen3-8b-q5-llamacpp.yaml"], "recommended")
        self.assertEqual(fits["qwen38-27b-q2-llamacpp.yaml"], "offload")
