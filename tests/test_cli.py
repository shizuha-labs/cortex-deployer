from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout

from cortex_deployer.cli import main
from cortex_deployer.recipes import list_examples


class CliTests(unittest.TestCase):
    def test_engines(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            with self.assertRaises(SystemExit) as caught:
                main(["engines"])
        self.assertEqual(caught.exception.code, 0)
        self.assertIn("llamacpp", buf.getvalue())
        self.assertIn("vllm", buf.getvalue())

    def test_render_example(self):
        path = next(p for p in list_examples() if p.name.startswith("llamacpp"))
        buf = io.StringIO()
        with redirect_stdout(buf):
            with self.assertRaises(SystemExit) as caught:
                main(["render", str(path)])
        self.assertEqual(caught.exception.code, 0)
        self.assertIn("llama-server", buf.getvalue())

    def test_recommend_cli(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            with self.assertRaises(SystemExit) as caught:
                main(["recommend"])
        self.assertEqual(caught.exception.code, 0)
        self.assertIn("qwen38-27b-q3-llamacpp.yaml", buf.getvalue())

    def test_render_q3(self):
        path = next(p for p in list_examples() if p.name == "qwen38-27b-q3-llamacpp.yaml")
        buf = io.StringIO()
        with redirect_stdout(buf):
            with self.assertRaises(SystemExit) as caught:
                main(["render", str(path)])
        self.assertEqual(caught.exception.code, 0)
        self.assertIn("llama-server", buf.getvalue())
        self.assertIn("--n-gpu-layers", buf.getvalue())
