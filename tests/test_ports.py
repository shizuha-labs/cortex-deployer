from __future__ import annotations

import socket
import unittest

from cortex_deployer.ports import (
    argv_port,
    bind_error_in_log,
    pick_free_port,
    set_argv_port,
)
from cortex_deployer.runtime import pick_port


class PortTests(unittest.TestCase):
    def test_skips_busy_preferred(self):
        blocker = socket.socket()
        self.addCleanup(blocker.close)
        blocker.bind(("127.0.0.1", 0))
        busy = blocker.getsockname()[1]
        chosen = pick_port(busy)
        self.assertNotEqual(chosen, busy)
        self.assertGreaterEqual(chosen, 1024)

    def test_8080_busy_does_not_stick(self):
        blocker = socket.socket()
        self.addCleanup(blocker.close)
        try:
            blocker.bind(("127.0.0.1", 8080))
        except OSError:
            self.skipTest("8080 already unusable")
        chosen = pick_free_port(8080, host="127.0.0.1")
        self.assertNotEqual(chosen, 8080)

    def test_argv_port_roundtrip(self):
        argv = ["llama-server", "-m", "m.gguf", "--host", "127.0.0.1", "--port", "8080"]
        self.assertEqual(argv_port(argv), 8080)
        out = set_argv_port(argv, 18080)
        self.assertEqual(argv_port(out), 18080)
        self.assertEqual(out[0], "llama-server")

    def test_bind_error_detects_llama_log(self):
        log = (
            "E srv         start: couldn't bind HTTP server socket, "
            "hostname: 127.0.0.1, port: 8080"
        )
        self.assertTrue(bind_error_in_log(log))
        self.assertFalse(bind_error_in_log("model loaded"))
