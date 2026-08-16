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
