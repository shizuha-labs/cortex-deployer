"""Tests for pairing tokens (CTX-733): short-lived, revocable, constant-time
validation, plus the /api/pairing-token endpoints."""

from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from cortex_deployer import pairing, store
from cortex_deployer.httpapi import serve_in_thread


class PairingUnitTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        os.environ["CORTEX_DEPLOYER_HOME"] = self.tmp.name
        os.environ["CORTEX_DEPLOYER_CATALOG_URL"] = "bundled"

    def tearDown(self):
        os.environ.pop("CORTEX_DEPLOYER_HOME", None)
        os.environ.pop("CORTEX_DEPLOYER_CATALOG_URL", None)
        self.tmp.cleanup()

    def test_generate_returns_bounded_token(self):
        rec = pairing.generate()
        self.assertIn("token", rec)
        self.assertTrue(len(rec["token"]) >= 32)
        self.assertEqual(rec["ttl_seconds"], pairing.TOKEN_TTL_SECONDS)
        self.assertGreater(rec["expires_at"], rec["created_at"])

    def test_current_returns_active_token(self):
        rec = pairing.generate()
        cur = pairing.current()
        self.assertIsNotNone(cur)
        self.assertEqual(cur["token"], rec["token"])

    def test_validate_constant_time(self):
        rec = pairing.generate()
        self.assertTrue(pairing.validate(rec["token"]))
        self.assertFalse(pairing.validate("wrong-token"))
        self.assertFalse(pairing.validate(""))

    def test_revoke_invalidates(self):
        rec = pairing.generate()
        self.assertTrue(pairing.validate(rec["token"]))
        pairing.revoke()
        self.assertFalse(pairing.validate(rec["token"]))
        self.assertIsNone(pairing.current())

    def test_regenerate_revokes_previous(self):
        first = pairing.generate()
        second = pairing.generate()
        self.assertNotEqual(first["token"], second["token"])
        self.assertFalse(pairing.validate(first["token"]))
        self.assertTrue(pairing.validate(second["token"]))

    def test_expired_token_invalidates(self):
        rec = pairing.generate()
        state = store.load_state()
        state["pairing"]["expires_at"] = int(time.time()) - 1
        store.save_state(state)
        self.assertFalse(pairing.validate(rec["token"]))
        self.assertIsNone(pairing.current())


class PairingApiTests(unittest.TestCase):
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
        os.environ.pop("CORTEX_DEPLOYER_HOME", None)
        os.environ.pop("CORTEX_DEPLOYER_CATALOG_URL", None)
        self.tmp.cleanup()

    def _json(self, method, path, payload=None):
        data = None if payload is None else json.dumps(payload).encode()
        req = Request(self.base + path, data=data, method=method)
        if data is not None:
            req.add_header("content-type", "application/json")
        try:
            with urlopen(req, timeout=5) as resp:
                return resp.status, json.loads(resp.read().decode())
        except HTTPError as exc:
            return exc.code, json.loads(exc.read().decode())

    def test_get_pairing_token_mints_on_first_use(self):
        status, data = self._json("GET", "/api/pairing-token")
        self.assertEqual(status, 200)
        self.assertIn("token", data)
        self.assertGreaterEqual(len(data["token"]), 32)

    def test_get_pairing_token_is_stable(self):
        _, first = self._json("GET", "/api/pairing-token")
        _, second = self._json("GET", "/api/pairing-token")
        self.assertEqual(first["token"], second["token"])

    def test_revoke_then_new_token(self):
        _, first = self._json("GET", "/api/pairing-token")
        status, _ = self._json("POST", "/api/pairing-token/revoke")
        self.assertEqual(status, 200)
        _, second = self._json("GET", "/api/pairing-token")
        self.assertNotEqual(first["token"], second["token"])


if __name__ == "__main__":
    unittest.main()
