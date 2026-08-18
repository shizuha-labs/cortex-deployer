from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread

from cortex_deployer import attach
from cortex_deployer.cli import main


class _FakeModels(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        return

    def do_GET(self):
        if self.path.rstrip("/") in {"/v1/models", "/models"}:
            body = json.dumps({"data": [{"id": "qwen2.5-7b"}, {"id": "other"}]}).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_error(404)


class AttachUnitTests(unittest.TestCase):
    def test_normalize(self):
        self.assertEqual(attach.normalize_openai_url("127.0.0.1:1234"), "http://127.0.0.1:1234/v1")
        self.assertEqual(attach.normalize_openai_url("http://127.0.0.1:1234/v1/"), "http://127.0.0.1:1234/v1")
        self.assertEqual(
            attach.normalize_openai_url("http://127.0.0.1:11434/v1/models"),
            "http://127.0.0.1:11434/v1",
        )

    def test_infer_engine(self):
        self.assertEqual(attach.infer_engine("http://127.0.0.1:1234/v1"), "lmstudio")
        self.assertEqual(attach.infer_engine("http://127.0.0.1:11434/v1"), "ollama")
        self.assertEqual(attach.infer_engine("http://127.0.0.1:8000/v1"), "vllm")

    def test_attach_force_without_listener(self):
        tmp = tempfile.TemporaryDirectory()
        os.environ["CORTEX_DEPLOYER_HOME"] = tmp.name
        try:
            row = attach.attach(
                "http://127.0.0.1:9/v1",
                model="forced",
                require_probe=False,
            )
            self.assertEqual(row["served_name"], "forced")
            self.assertEqual(row["kind"], "adopt")
            self.assertEqual(row["engine"], "external")
        finally:
            os.environ.pop("CORTEX_DEPLOYER_HOME", None)
            tmp.cleanup()

    def test_attach_probes_models(self):
        httpd = HTTPServer(("127.0.0.1", 0), _FakeModels)
        Thread(target=httpd.serve_forever, daemon=True).start()
        port = httpd.server_address[1]
        tmp = tempfile.TemporaryDirectory()
        os.environ["CORTEX_DEPLOYER_HOME"] = tmp.name
        try:
            row = attach.attach(f"http://127.0.0.1:{port}")
            self.assertEqual(row["served_name"], "qwen2.5-7b")
            self.assertTrue(row.get("healthy"))
        finally:
            httpd.shutdown()
            os.environ.pop("CORTEX_DEPLOYER_HOME", None)
            tmp.cleanup()

    def test_cli_scan_and_attach_parser(self):
        from cortex_deployer.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["attach", "--scan"])
        self.assertTrue(args.scan)
        args = parser.parse_args(["attach", "http://127.0.0.1:1234/v1", "--model", "x"])
        self.assertEqual(args.url, "http://127.0.0.1:1234/v1")
        self.assertEqual(args.model, "x")

    def test_cli_force_attach(self):
        tmp = tempfile.TemporaryDirectory()
        os.environ["CORTEX_DEPLOYER_HOME"] = tmp.name
        try:
            buf = io.StringIO()
            with redirect_stdout(buf):
                with self.assertRaises(SystemExit) as caught:
                    main(["attach", "http://127.0.0.1:9/v1", "--model", "cli-forced", "--force"])
            self.assertEqual(caught.exception.code, 0)
            self.assertIn("cli-forced", buf.getvalue())
        finally:
            os.environ.pop("CORTEX_DEPLOYER_HOME", None)
            tmp.cleanup()
