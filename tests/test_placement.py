from __future__ import annotations

import unittest

from cortex_deployer.placement import apply_ngl_args, fit_label, offload_floor_mb


class PlacementTests(unittest.TestCase):
    def test_offload_floor(self):
        self.assertGreaterEqual(offload_floor_mb(22000, context_length=16384), 3500)
        self.assertEqual(offload_floor_mb(22000, min_offload_vram_mb=6000), 6000)

    def test_8gb_marks_27b_offload_not_skip(self):
        self.assertEqual(
            fit_label(15500, 8192, apple=False, engine="llamacpp", min_offload_vram_mb=5500),
            "offload",
        )
        self.assertEqual(
            fit_label(7800, 8192, apple=False, engine="llamacpp"),
            "recommended",
        )

    def test_apply_ngl_drops_pin_for_offload(self):
        extra = ["--n-gpu-layers", "99", "-np", "1"]
        out = apply_ngl_args(extra, "offload", 32768)
        self.assertNotIn("99", out)
        self.assertNotIn("--n-gpu-layers", out)
        self.assertIn("--fit", out)
        self.assertIn("32768", out)
        kept = apply_ngl_args(extra, "recommended", 32768)
        self.assertEqual(kept, extra)
