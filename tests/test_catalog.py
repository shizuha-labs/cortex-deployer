from __future__ import annotations

import os
import unittest
from unittest.mock import patch
from urllib.error import URLError

from cortex_deployer.catalog import bundled_catalog, fetch_catalog


class CatalogTests(unittest.TestCase):
    def test_bundled_schema(self):
        cat = bundled_catalog()
        self.assertEqual(cat["schema"], "cortex.deployer.catalog.v1")
        ids = {m["id"] for m in cat["models"]}
        self.assertIn("qwen3.8-27b", ids)
        self.assertIn("qwen3.5-9b", ids)
        self.assertIn("qwen3-14b", ids)
        self.assertIn("qwen3-8b", ids)
        self.assertIn("minimax-h3", ids)
        rel = cat.get("deployer_release") or {}
        self.assertTrue(rel.get("version"))
        self.assertIn("tarball", rel)

    def test_env_bundled_skips_network(self):
        os.environ["CORTEX_DEPLOYER_CATALOG_URL"] = "bundled"
        self.addCleanup(lambda: os.environ.pop("CORTEX_DEPLOYER_CATALOG_URL", None))
        cat = fetch_catalog()
        self.assertFalse(cat["fetched"])
        self.assertEqual(cat["source"], "bundled")

    def test_force_adds_cache_buster(self):
        seen: dict[str, str] = {}

        def fake_urlopen(req, timeout=0):  # noqa: ARG001
            seen["url"] = req.full_url
            raise URLError("skip")

        with patch("cortex_deployer.catalog.urlopen", fake_urlopen):
            fetch_catalog("https://example.test/catalog.json", force=True)
        self.assertIn("t=", seen["url"])
        self.assertTrue(seen["url"].startswith("https://example.test/catalog.json?"))
