from __future__ import annotations

import unittest

from cortex_deployer.recipes import list_examples, load_recipe
from cortex_deployer.spec import recipe_from_dict


class SpecTests(unittest.TestCase):
    def test_unknown_engine_fails(self):
        with self.assertRaises(ValueError):
            recipe_from_dict(
                {
                    "schema_version": "deployer.recipe.v1",
                    "name": "x",
                    "engine": "ollama",
                    "model": {"id": "x"},
                }
            )

    def test_examples_load(self):
        paths = list_examples()
        self.assertGreaterEqual(len(paths), 4)
        names = {path.name for path in paths}
        self.assertIn("qwen38-27b-q3-llamacpp.yaml", names)
        self.assertIn("mlx-macos.yaml", names)
        for path in paths:
            recipe = load_recipe(path)
            self.assertTrue(recipe.model.id)
            self.assertTrue(recipe.upstream_url().startswith("http://"))
            blob = path.read_text(encoding="utf-8").lower()
            self.assertNotIn("v4", blob)
            self.assertNotRegex(blob, r"100\.64\.0\.")

    def test_q3_recipe_points_at_public_gguf(self):
        path = next(p for p in list_examples() if p.name == "qwen38-27b-q3-llamacpp.yaml")
        recipe = load_recipe(path)
        self.assertEqual(recipe.model.repo, "unsloth/Qwen3.8-27B-GGUF")
        self.assertEqual(recipe.min_vram_mb, 22000)
        self.assertIn("UD-Q3_K_XL", recipe.download_glob)
