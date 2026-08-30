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
        self.assertIn("comfyui", buf.getvalue())

    def test_render_example(self):
        path = next(p for p in list_examples() if p.name.startswith("llamacpp"))
        buf = io.StringIO()
        with redirect_stdout(buf):
            with self.assertRaises(SystemExit) as caught:
                main(["render", str(path)])
        self.assertEqual(caught.exception.code, 0)
        self.assertIn("llama-server", buf.getvalue())

    def test_setup_in_parser(self):
        from cortex_deployer.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["setup", "--recipe", "qwen38-27b-q3-llamacpp.yaml"])
        self.assertEqual(args.command, "setup")
        self.assertEqual(args.recipe, "qwen38-27b-q3-llamacpp.yaml")

    def test_update_in_parser(self):
        from cortex_deployer.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["update", "--check"])
        self.assertEqual(args.command, "update")
        self.assertTrue(args.check)
        args = parser.parse_args(["upgrade", "--restart"])
        self.assertEqual(args.command, "upgrade")
        self.assertTrue(args.restart)
        args = parser.parse_args(["server", "--auto-update"])
        self.assertTrue(args.auto_update)
        args = parser.parse_args(["auto-update", "--off"])
        self.assertEqual(args.command, "auto-update")
        self.assertTrue(args.off)
        args = parser.parse_args(
            [
                "connect",
                "--gateway",
                "wss://example/ws",
                "--model",
                "Qwen3.8-27B-MLX",
                "--recycle-cmd",
                "launchctl kickstart -k gui/501/com.shizuha.mlx-qwen38-27b-8bit",
                "--auto-update",
            ]
        )
        self.assertEqual(args.command, "connect")
        self.assertTrue(args.auto_update)
        self.assertIn("mlx-qwen38-27b-8bit", args.recycle_cmd)
        args = parser.parse_args(
            [
                "connect",
                "--gateway",
                "wss://example/ws",
                "--model",
                "Qwen3.8-27B-MLX",
                "--rewrite-model",
                "default_model",
            ]
        )
        self.assertEqual(args.rewrite_model, "default_model")

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

    def test_run_dry_run_mlx_recipe(self):
        path = next(p for p in list_examples() if p.name == "qwen38-27b-mlx.yaml")
        buf = io.StringIO()
        with redirect_stdout(buf):
            with self.assertRaises(SystemExit) as caught:
                main(["run", str(path), "--dry-run"])
        self.assertEqual(caught.exception.code, 0)
        out = buf.getvalue()
        self.assertIn("rapid-mlx", out)
        self.assertIn("serve", out)
        self.assertIn("/models/Qwen3.8-27B-8bit", out)
        self.assertIn("--hybrid-cache-entries", out)
        self.assertIn("--pin-system-prompt", out)
        self.assertIn("Qwen3.8-27B-MTP-8bit", out)
