from __future__ import annotations

import unittest

from cortex_deployer.protocol import (
    b64d,
    cancel_frame,
    hello_frame,
    models_with_max_model_len,
    request_wants_stream,
)


class HelloTests(unittest.TestCase):
    def test_model_is_first_alias(self):
        frame = hello_frame("My-Model", ["my-model", "My-Model"])
        self.assertEqual(frame["model"], "My-Model")
        self.assertEqual(frame["aliases"], ["My-Model", "my-model"])

    def test_rejects_empty(self):
        with self.assertRaises(ValueError):
            hello_frame("   ", [])

    def test_optional_inventory(self):
        frame = hello_frame(
            "m",
            max_model_len=8192,
            engine="llamacpp",
            quant="Q4_K_M",
            max_concurrent=2,
        )
        self.assertEqual(frame["max_model_len"], 8192)
        self.assertEqual(frame["engine"], "llamacpp")
        self.assertEqual(frame["quant"], "Q4_K_M")
        self.assertEqual(frame["max_concurrent"], 2)


class StreamDetectTests(unittest.TestCase):
    def test_stream_true(self):
        self.assertTrue(request_wants_stream(b'{"stream":true}'))

    def test_stream_false_or_missing(self):
        self.assertFalse(request_wants_stream(b'{"stream":false}'))
        self.assertFalse(request_wants_stream(b"{}"))
        self.assertFalse(request_wants_stream(None))


class ModelsLenTests(unittest.TestCase):
    def test_copies_context_window(self):
        payload, changed = models_with_max_model_len(
            {"data": [{"id": "m", "context_window": 262144}]}
        )
        self.assertTrue(changed)
        self.assertEqual(payload["data"][0]["max_model_len"], 262144)

    def test_b64_roundtrip(self):
        self.assertEqual(b64d(None), b"")

    def test_cancel_frame(self):
        self.assertEqual(cancel_frame("rid-9"), {"kind": "cancel", "id": "rid-9"})
