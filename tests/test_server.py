from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from urllib.request import Request, urlopen

from cortex_deployer.httpapi import serve_in_thread


class ServerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        os.environ["CORTEX_DEPLOYER_HOME"] = self.tmp.name
        self.httpd = serve_in_thread("127.0.0.1", 0)
        self.port = self.httpd.server_address[1]
        self.base = f"http://127.0.0.1:{self.port}"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.tmp.cleanup()
        os.environ.pop("CORTEX_DEPLOYER_HOME", None)

    def _json(self, method: str, path: str, payload=None):
        data = None if payload is None else json.dumps(payload).encode()
        req = Request(self.base + path, data=data, method=method)
        if data is not None:
            req.add_header("content-type", "application/json")
        with urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read().decode())

    def test_health_and_ui(self):
        status, body = self._json("GET", "/api/health")
        self.assertEqual(status, 200)
        self.assertTrue(body["ok"])
        with urlopen(self.base + "/", timeout=5) as resp:
            html = resp.read().decode()
        self.assertIn("Cortex Deployer", html)
        self.assertIn("Deploy model", html)

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
