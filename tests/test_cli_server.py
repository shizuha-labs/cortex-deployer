from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout

from cortex_deployer.cli import build_parser, main


class CliServerTests(unittest.TestCase):
    def test_parser_has_server_aliases(self):
        parser = build_parser()
        args = parser.parse_args(["server", "--port", "9"])
        self.assertEqual(args.command, "server")
        self.assertEqual(args.port, 9)
        args = parser.parse_args(["up"])
        self.assertIn(args.command, {"up", "server"})

    def test_help_mentions_server(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            with self.assertRaises(SystemExit) as caught:
                main([])
        self.assertEqual(caught.exception.code, 0)
        self.assertIn("server", buf.getvalue())
