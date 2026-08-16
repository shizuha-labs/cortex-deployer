from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from cortex_deployer.httpapi import serve_in_thread


class ServerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        os.environ["CORTEX_DEPLOYER_HOME"] = self.tmp.name
        os.environ["CORTEX_DEPLOYER_CATALOG_URL"] = "bundled"
        self.httpd = serve_in_thread("127.0.0.1", 0)
        self.port = self.httpd.server_address[1]
        self.base = f"http://127.0.0.1:{self.port}"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.tmp.cleanup()
        os.environ.pop("CORTEX_DEPLOYER_HOME", None)
        os.environ.pop("CORTEX_DEPLOYER_CATALOG_URL", None)

    def _json(self, method: str, path: str, payload=None):
        data = None if payload is None else json.dumps(payload).encode()
        req = Request(self.base + path, data=data, method=method)
        if data is not None:
            req.add_header("content-type", "application/json")
        try:
            with urlopen(req, timeout=5) as resp:
                return resp.status, json.loads(resp.read().decode())
        except HTTPError as exc:
            return exc.code, json.loads(exc.read().decode())

    def test_health_and_ui(self):
        status, body = self._json("GET", "/api/health")
        self.assertEqual(status, 200)
        self.assertTrue(body["ok"])
        with urlopen(self.base + "/", timeout=5) as resp:
            html = resp.read().decode()
        self.assertIn("Cortex Deployer", html)
        self.assertIn("Choose a Qwen build", html)
        self.assertIn("Update Deployer", html)
        self.assertIn("Refresh catalog", html)
        self.assertNotIn("id=\"hf-token\"", html)
        self.assertNotIn("id=\"hf-box\"", html)

    def test_host_and_recipes(self):
        _, host = self._json("GET", "/api/host")
        self.assertIn(host["os"], {"Linux", "Darwin", "Windows"})
        _, rec = self._json("GET", "/api/recipes")
        files = {r["file"] for r in rec["recipes"]}
        self.assertIn("qwen38-27b-llamacpp.yaml", files)
        self.assertIn("llamacpp-cuda.yaml", files)

    def test_adopt_list_delete(self):
        _, created = self._json(
            "POST",
            "/api/backends",
            {
                "kind": "adopt",
                "model_id": "demo",
                "base_url": "http://127.0.0.1:9/v1",
            },
        )
        self.assertEqual(created["served_name"], "demo")
        self.assertEqual(created["kind"], "adopt")
        _, listed = self._json("GET", "/api/backends")
        self.assertEqual(len(listed["backends"]), 1)
        bid = created["id"]
        _, deleted = self._json("DELETE", f"/api/backends/{bid}")
        self.assertTrue(deleted["ok"])
        _, listed = self._json("GET", "/api/backends")
        self.assertEqual(listed["backends"], [])

    def test_v1_models_empty(self):
        _, body = self._json("GET", "/v1/models")
        self.assertEqual(body["data"], [])

    def test_setup_unknown_recipe(self):
        status, err = self._json("POST", "/api/setup", {"recipe": "nope.yaml"})
        self.assertEqual(status, 400)
        self.assertIn("unknown recipe", err["error"])

    def test_recommend_and_connect_requires_token(self):
        _, rec = self._json("GET", "/api/recommend")
        self.assertTrue(rec["recipes"])
        files = {r["file"] for r in rec["recipes"]}
        self.assertIn("qwen38-27b-q3-llamacpp.yaml", files)
        self.assertIn("mlx-macos.yaml", files)
        _, created = self._json(
            "POST",
            "/api/backends",
            {"kind": "adopt", "model_id": "c", "base_url": "http://127.0.0.1:9/v1"},
        )
        status, err = self._json(
            "POST",
            f"/api/backends/{created['id']}/connect",
            {"gateway": "wss://example.invalid/ws", "token": ""},
        )
        self.assertEqual(status, 400)
        self.assertIn("token", err["error"])

    def test_llamacpp_recipe_requires_weights(self):
        status, err = self._json(
            "POST",
            "/api/backends",
            {
                "kind": "recipe",
                "recipe": "qwen38-27b-q3-llamacpp.yaml",
                "autostart": False,
            },
        )
        self.assertEqual(status, 400)
        self.assertIn("GGUF", err["error"])

    def test_recipe_by_filename_no_start(self):
        status, body = self._json(
            "POST",
            "/api/backends",
            {
                "kind": "recipe",
                "recipe": "qwen38-27b-q3-llamacpp.yaml",
                "model_path": "/tmp/Qwen3.8-27B-UD-Q3_K_XL.gguf",
                "autostart": False,
                "port": 18081,
            },
        )
        self.assertEqual(status, 201)
        self.assertEqual(body["engine"], "llamacpp")
        self.assertIn("Qwen3.8-27B-UD-Q3_K_XL.gguf", " ".join(body.get("argv") or []))

    def test_download_requires_glob(self):
        status, err = self._json(
            "POST",
            "/api/downloads",
            {"repo": "unsloth/Qwen3.8-27B-GGUF"},
        )
        self.assertEqual(status, 400)

    def test_version_and_catalog_refresh(self):
        from cortex_deployer import __version__

        status, ver = self._json("GET", "/api/version")
        self.assertEqual(status, 200)
        self.assertEqual(ver["version"], __version__)
        self.assertIn("update_available", ver)
        status, cat = self._json("POST", "/api/catalog/refresh", {})
        self.assertEqual(status, 200)
        self.assertTrue(cat["ok"])

    def test_update_spawns_then_exits(self):
        from unittest.mock import patch

        fake = {"ok": True, "restarting": True, "previous": "0.3.5"}
        with (
            patch("cortex_deployer.selfupdate.spawn_updater", return_value=fake) as spawn,
            patch("cortex_deployer.httpapi.os._exit"),
            patch("cortex_deployer.httpapi.time.sleep"),
        ):
            status, body = self._json("POST", "/api/update", {})
        self.assertEqual(status, 200)
        self.assertTrue(body.get("ok") or body.get("restarting"))
        spawn.assert_called_once()

    def test_cors_options(self):
        req = Request(self.base + "/v1/models", method="OPTIONS")
        with urlopen(req, timeout=5) as resp:
            self.assertEqual(resp.status, 204)
            self.assertEqual(resp.headers.get("access-control-allow-origin"), "*")
