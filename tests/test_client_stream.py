from __future__ import annotations

import asyncio
import json
import unittest

from cortex_deployer import client as dc
from cortex_deployer.protocol import b64d, b64e


class FakeStream:
    def __init__(self, chunks, status=200, headers=None):
        self._chunks = chunks
        self.status_code = status
        self.headers = headers or {"content-type": "text/event-stream"}

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def aiter_bytes(self):
        for chunk in self._chunks:
            yield chunk

    async def aread(self):
        return b"".join(self._chunks)


class FakeClient:
    def __init__(self, stream):
        self._stream = stream

    def stream(self, *args, **kwargs):
        return self._stream


class Metrics404SynthTests(unittest.TestCase):
    def test_metrics_404_becomes_200(self):
        sent = []

        async def send(obj):
            sent.append(obj)

        fake = FakeStream([b"not found"], status=404, headers={"content-type": "text/plain"})
        asyncio.run(
            dc.relay_request(
                FakeClient(fake),
                "http://127.0.0.1:8016/v1",
                {"id": "m1", "method": "GET", "path": "/metrics", "headers": {}},
                send,
            )
        )
        self.assertEqual(sent[0]["status"], 200)
        self.assertIn(b"no /metrics", dc.b64d(sent[0]["body_b64"]))


class AdvertiseModelsTests(unittest.TestCase):
    def test_injects_served_name_first(self):
        raw = json.dumps({"object": "list", "data": [{"id": "/models/x"}]}).encode()
        out = json.loads(
            dc._inject_advertised_models(raw, ["qwen3.8-27b", "Qwen3.8-27B-MLX"], 131072)
        )
        self.assertEqual(out["data"][0]["id"], "qwen3.8-27b")
        self.assertEqual(out["data"][0]["context_window"], 131072)
        self.assertEqual(out["data"][1]["id"], "Qwen3.8-27B-MLX")


class RewriteModelTests(unittest.TestCase):
    def test_rewrites_openai_model_field(self):
        body = json.dumps({"model": "Qwen3.8-27B-MLX", "stream": True}).encode()
        out = dc._rewrite_json_model(body, "default_model")
        self.assertEqual(json.loads(out)["model"], "default_model")
        self.assertTrue(json.loads(out)["stream"])

    def test_leaves_non_json_alone(self):
        self.assertEqual(dc._rewrite_json_model(b"not-json", "default_model"), b"not-json")


class JoinUpstreamTests(unittest.TestCase):
    def test_chat_stays_under_v1(self):
        self.assertEqual(
            dc.join_upstream("http://127.0.0.1:8014/v1", "/chat/completions"),
            "http://127.0.0.1:8014/v1/chat/completions",
        )

    def test_metrics_and_slots_use_origin_root(self):
        base = "http://127.0.0.1:8014/v1"
        self.assertEqual(dc.join_upstream(base, "/metrics"), "http://127.0.0.1:8014/metrics")
        self.assertEqual(dc.join_upstream(base, "/slots"), "http://127.0.0.1:8014/slots")
        self.assertEqual(dc.join_upstream(base, "/health"), "http://127.0.0.1:8014/health")

    def test_already_root_upstream_is_unchanged(self):
        self.assertEqual(
            dc.join_upstream("http://127.0.0.1:8014", "/metrics"),
            "http://127.0.0.1:8014/metrics",
        )


class RelayStreamTests(unittest.TestCase):
    def test_stream_emits_start_chunks_end(self):
        sent = []

        async def send(obj):
            sent.append(obj)

        body = json.dumps({"stream": True, "model": "example"}).encode()
        fake = FakeStream([b"data: a\n\n", b"data: b\n\n"])
        asyncio.run(
            dc.relay_request(
                FakeClient(fake),
                "http://127.0.0.1:8014/v1",
                {
                    "id": "rid-1",
                    "method": "POST",
                    "path": "/chat/completions",
                    "headers": {},
                    "body_b64": b64e(body),
                },
                send,
            )
        )
        self.assertEqual([m["kind"] for m in sent], ["start", "chunk", "chunk", "end"])
        self.assertEqual(sent[0]["status"], 200)
        self.assertEqual(b64d(sent[1]["body_b64"]), b"data: a\n\n")
        self.assertEqual(b64d(sent[2]["body_b64"]), b"data: b\n\n")

    def test_non_stream_stays_buffered(self):
        sent = []

        async def send(obj):
            sent.append(obj)

        fake = FakeStream(
            [b'{"id":"x"}'],
            headers={"content-type": "application/json"},
        )
        asyncio.run(
            dc.relay_request(
                FakeClient(fake),
                "http://127.0.0.1:8014/v1",
                {
                    "id": "rid-2",
                    "method": "GET",
                    "path": "/models",
                    "headers": {},
                },
                send,
            )
        )
        self.assertEqual([m["kind"] for m in sent], ["response"])
        self.assertEqual(sent[0]["status"], 200)
        self.assertEqual(b64d(sent[0]["body_b64"]), b'{"id":"x"}')

    def test_cancel_closes_upstream_stream(self):
        closed = {"n": 0}

        class SlowStream(FakeStream):
            async def aiter_bytes(self):
                yield b"data: a\n\n"
                await asyncio.sleep(30)
                yield b"data: b\n\n"

            async def __aexit__(self, *exc):
                closed["n"] += 1
                return False

        sent = []

        async def send(obj):
            sent.append(obj)

        body = json.dumps({"stream": True, "model": "example"}).encode()
        fake = SlowStream([b"data: a\n\n"])

        async def run():
            task = asyncio.create_task(
                dc.relay_request(
                    FakeClient(fake),
                    "http://127.0.0.1:8014/v1",
                    {
                        "id": "rid-cancel",
                        "method": "POST",
                        "path": "/chat/completions",
                        "headers": {},
                        "body_b64": b64e(body),
                    },
                    send,
                )
            )
            for _ in range(50):
                if sent:
                    break
                await asyncio.sleep(0.01)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task

        asyncio.run(run())
        self.assertEqual(closed["n"], 1)
        self.assertEqual(sent[0]["kind"], "start")
