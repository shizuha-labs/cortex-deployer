from __future__ import annotations

import unittest

from cortex_deployer.engines import render_process
from cortex_deployer.spec import recipe_from_dict


def _recipe(**overrides):
    data = {
        "schema_version": "deployer.recipe.v1",
        "name": "t",
        "engine": "llamacpp",
        "executor": "process",
        "model": {
            "id": "m",
            "served_name": "m",
            "source": {"kind": "local_path", "path": "/models/m.gguf"},
        },
        "launch": {"host": "127.0.0.1", "port": 8080, "context_length": 4096},
    }
    data.update(overrides)
    return recipe_from_dict(data)


class EngineTests(unittest.TestCase):
    def test_llamacpp_argv(self):
        launch = render_process(_recipe())
        self.assertEqual(launch.argv[0], "llama-server")
        self.assertIn("/models/m.gguf", launch.argv)
        self.assertIn("-c", launch.argv)

    def test_vllm_argv(self):
        recipe = _recipe(
            engine="vllm",
            model={
                "id": "m",
                "served_name": "m",
                "source": {"kind": "huggingface", "repo": "org/m"},
            },
        )
        launch = render_process(recipe)
        self.assertIn("vllm.entrypoints.openai.api_server", launch.argv)
        self.assertIn("org/m", launch.argv)

    def test_sglang_rejects_multinode(self):
        recipe = _recipe(
            engine="sglang",
            model={
                "id": "m",
                "served_name": "m",
                "source": {"kind": "huggingface", "repo": "org/m"},
            },
            launch={
                "host": "127.0.0.1",
                "port": 30000,
                "extra_args": ["--nnodes", "2"],
            },
        )
        with self.assertRaises(ValueError):
            render_process(recipe)

    def test_mlx_argv(self):
        recipe = _recipe(
            engine="mlx",
            model={
                "id": "m",
                "served_name": "m",
                "source": {"kind": "local_path", "path": "/models/m"},
            },
            launch={"host": "127.0.0.1", "port": 8014},
        )
        launch = render_process(recipe)
        self.assertEqual(launch.argv[:3], ("rapid-mlx", "serve", "/models/m"))

    def test_mlx_extra_args_keep_mtp_flags(self):
        recipe = _recipe(
            engine="mlx",
            model={
                "id": "qwen3.8-27b",
                "served_name": "qwen3.8-27b",
                "source": {"kind": "local_path", "path": "/models/Qwen3.8-27B-8bit"},
            },
            launch={
                "host": "127.0.0.1",
                "port": 8014,
                "extra_args": [
                    "--hybrid-cache-entries",
                    "8",
                    "--force-spec-decode",
                    "--pin-system-prompt",
                ],
            },
        )
        launch = render_process(recipe)
        self.assertIn("--hybrid-cache-entries", launch.argv)
        self.assertIn("8", launch.argv)
        self.assertIn("--pin-system-prompt", launch.argv)

    def test_mlx_lm_server_argv_does_not_inject_rapid_serve(self):
        recipe = _recipe(
            engine="mlx",
            model={
                "id": "qwen3.8-27b",
                "served_name": "qwen3.8-27b",
                "source": {"kind": "local_path", "path": "/models/Qwen3.8-27B-8bit"},
            },
            launch={
                "host": "127.0.0.1",
                "port": 8015,
                "extra_args": [
                    "mlx_lm.server",
                    "--max-tokens",
                    "32768",
                    "--decode-concurrency",
                    "1",
                ],
            },
        )
        launch = render_process(recipe)
        self.assertEqual(launch.argv[0], "mlx_lm.server")
        self.assertNotIn("serve", launch.argv)
        self.assertNotIn("--served-model-name", launch.argv)
        self.assertEqual(
            launch.argv[:6],
            (
                "mlx_lm.server",
                "--model",
                "/models/Qwen3.8-27B-8bit",
                "--host",
                "127.0.0.1",
                "--port",
            ),
        )
        self.assertIn("8015", launch.argv)
        self.assertIn("--decode-concurrency", launch.argv)

    def test_comfyui_argv(self):
        recipe = _recipe(
            engine="comfyui",
            model={
                "id": "MiniMax-H3",
                "served_name": "MiniMax-H3",
                "source": {"kind": "local_path", "path": "~/opt/ComfyUI"},
            },
            launch={"host": "127.0.0.1", "port": 8000},
        )
        launch = render_process(recipe)
        self.assertEqual(launch.engine, "comfyui")
        self.assertIn("-m", launch.argv)
        self.assertIn("cortex_deployer.engines.comfyui", launch.argv)
        self.assertIn("--comfy-root", launch.argv)
        self.assertIn("--served-name", launch.argv)
        self.assertIn("MiniMax-H3", launch.argv)
        self.assertIn("8000", launch.argv)
