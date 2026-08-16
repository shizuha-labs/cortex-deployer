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
        for path in paths:
            recipe = load_recipe(path)
            self.assertTrue(recipe.model.id)
            self.assertTrue(recipe.upstream_url().startswith("http://"))
