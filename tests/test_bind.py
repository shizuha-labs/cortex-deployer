from __future__ import annotations

import errno
import socket
import unittest
from unittest import mock

from cortex_deployer.httpapi import _bind_unavailable, candidate_ports, serve


class BindTests(unittest.TestCase):
    def test_candidates_skip_privileged_and_end_with_os_assign(self):
        ports = candidate_ports(80)
        self.assertTrue(all(p == 0 or p >= 1024 for p in ports))
        self.assertEqual(ports[-1], 0)
        self.assertIn(18765, ports)

    def test_preferred_unprivileged_is_first(self):
        ports = candidate_ports(7480)
        self.assertEqual(ports[0], 7480)
        self.assertGreater(ports[1], 7480)

    def test_winerror_10013_is_unavailable(self):
        exc = OSError("forbidden")
        exc.winerror = 10013
        exc.errno = errno.EACCES
        self.assertTrue(_bind_unavailable(exc))
        self.assertFalse(_bind_unavailable(ValueError("nope")))

    def test_falls_back_when_preferred_is_busy(self):
        blocker = socket.socket()
        self.addCleanup(blocker.close)
        blocker.bind(("127.0.0.1", 0))
        busy = blocker.getsockname()[1]
        httpd = serve("127.0.0.1", busy)
        self.addCleanup(httpd.server_close)
        self.assertGreaterEqual(httpd.server_address[1], 1024)

    def test_skips_privileged_request(self):
        httpd = serve("127.0.0.1", 80)
        self.addCleanup(httpd.server_close)
        self.assertGreaterEqual(httpd.server_address[1], 1024)

    def test_walks_after_access_denied(self):
        calls: list[int] = []

        class Boom(OSError):
            def __init__(self, port: int):
                super().__init__("denied")
                self.winerror = 10013
                self.errno = errno.EACCES
                self.port = port

        orig = serve.__globals__["ThreadingHTTPServer"]

        def fake_server(addr, handler):
            host, port = addr
            calls.append(port)
            if port in {7480, 7481, 0}:
                if port == 0:
                    return orig((host, 0), handler)
                raise Boom(port)
            return orig((host, port), handler)

        with mock.patch("cortex_deployer.httpapi.ThreadingHTTPServer", side_effect=fake_server):
            httpd = serve("127.0.0.1", 7480)
            self.addCleanup(httpd.server_close)
        self.assertIn(7480, calls)
        self.assertNotEqual(httpd.server_address[1], 7480)
        self.assertGreaterEqual(httpd.server_address[1], 1024)
