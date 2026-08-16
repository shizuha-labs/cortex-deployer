from __future__ import annotations

import os
import unittest

from cortex_deployer.catalog import bundled_catalog, fetch_catalog


class CatalogTests(unittest.TestCase):
    def test_bundled_schema(self):
        cat = bundled_catalog()
        self.assertEqual(cat["schema"], "cortex.deployer.catalog.v1")
        ids = {m["id"] for m in cat["models"]}
        self.assertIn("qwen3.8-27b", ids)

    def test_env_bundled_skips_network(self):
        os.environ["CORTEX_DEPLOYER_CATALOG_URL"] = "bundled"
        self.addCleanup(lambda: os.environ.pop("CORTEX_DEPLOYER_CATALOG_URL", None))
        cat = fetch_catalog()
        self.assertFalse(cat["fetched"])
        self.assertEqual(cat["source"], "bundled")
