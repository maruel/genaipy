# Copyright 2026 Marc-Antoine Ruel. All rights reserved.
# Use of this source code is governed under the Apache License, Version 2.0
# that can be found in the LICENSE file.

# Offline Kyutai STT worker word assembly, marker timing, protocol errors and session slot tests.

import asyncio
import importlib.util
import pathlib
import sys
import types
import unittest
from unittest import mock

BINARY = "binary"


class FakeWebSocket:
    """Plays a client: received messages are dicts, as the msgpack stub keeps them."""

    def __init__(self):
        self.incoming = asyncio.Queue()
        self.sent = []
        self.closed = False

    async def prepare(self, _request):
        return

    def __aiter__(self):
        return self

    async def __anext__(self):
        msg = await self.incoming.get()
        if msg is None:
            raise StopAsyncIteration
        return types.SimpleNamespace(type=BINARY, data=msg)

    async def send_bytes(self, data):
        self.sent.append(data)

    async def close(self):
        self.closed = True


def load_worker():
    aiohttp = types.ModuleType("aiohttp")
    aiohttp.WSMsgType = types.SimpleNamespace(BINARY=BINARY)
    aiohttp.web = types.SimpleNamespace(Request=None, WebSocketResponse=FakeWebSocket)
    hub = types.ModuleType("huggingface_hub")
    hub.snapshot_download = None
    msgpack = types.ModuleType("msgpack")
    msgpack.packb = lambda message, use_single_float: message
    msgpack.unpackb = lambda data: data
    path = pathlib.Path(__file__).resolve().parents[1] / "providers" / "kyutai" / "kyutaisrv" / "worker.py"
    spec = importlib.util.spec_from_file_location("genaipy_test_kyutai_worker", path)
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(
        sys.modules,
        {
            "aiohttp": aiohttp,
            "huggingface_hub": hub,
            "msgpack": msgpack,
            "sentencepiece": types.ModuleType("sentencepiece"),
            spec.name: module,
        },
    ):
        spec.loader.exec_module(module)
    return module


worker = load_worker()
FRAME = [0.0] * worker.FRAME_SIZE


class FakeRecognizer:
    """Returns one scripted text token per frame, then padding."""

    delay_steps = 2

    def __init__(self, tokens):
        self.tokens = list(tokens)
        self.frames = 0
        self.resets = 0

    def reset(self):
        self.resets += 1

    def step(self, frame):
        assert len(frame) == worker.FRAME_SIZE
        self.frames += 1
        token = self.tokens.pop(0) if self.tokens else worker.PADDING
        if isinstance(token, Exception):
            raise token
        return token

    def decode(self, tokens):
        return "".join(chr(ord("a") + t - 10) for t in tokens)


class SessionTest(unittest.TestCase):
    def test_words(self):
        # Tokens before the delay are dropped; a padding token ends a word and
        # an end-of-padding token sets the next word's start.
        r = FakeRecognizer([10, 3, 10, 11, 3, 0, 12, 13, 3])
        out = worker.Session(r).audio(FRAME * 9)
        self.assertEqual(
            out,
            [
                {"type": "Word", "text": "ab", "start_time": 0.0},
                {"type": "Word", "text": "cd", "start_time": 4 / worker.FRAME_RATE},
            ],
        )

    def test_partial_frames(self):
        r = FakeRecognizer([])
        s = worker.Session(r)
        s.audio([0.0] * (worker.FRAME_SIZE - 1))
        self.assertEqual(r.frames, 0)
        s.audio([0.0] * (worker.FRAME_SIZE + 1))
        self.assertEqual(r.frames, 2)

    def test_marker(self):
        # The marker returns after the delay; like moshi-server, a word still
        # waiting for its padding token comes after it.
        r = FakeRecognizer([3, 3, 10, 11, 12, 3])
        s = worker.Session(r)
        self.assertEqual(s.audio(FRAME * 3), [])
        s.marker(7)
        self.assertEqual(s.audio(FRAME), [])
        self.assertEqual(s.audio(FRAME), [{"type": "Marker", "id": 7}])
        self.assertEqual(s.audio(FRAME), [{"type": "Word", "text": "abc", "start_time": 0.0}])


class ServerTest(unittest.IsolatedAsyncioTestCase):
    async def run_session(self, server, messages):
        ws = FakeWebSocket()
        for m in messages:
            ws.incoming.put_nowait(m)
        ws.incoming.put_nowait(None)
        with mock.patch.object(worker.web, "WebSocketResponse", lambda: ws):
            await server.asr_streaming(None)
        return ws

    async def test_errors(self):
        for name, recognizer, message, want in (
            ("type", FakeRecognizer([]), {"type": "OggOpus", "data": b""}, "unsupported message type 'OggOpus'"),
            ("model", FakeRecognizer([RuntimeError("oom")]), {"type": "Audio", "pcm": FRAME}, "oom"),
        ):
            with self.subTest(name):
                ws = await self.run_session(worker.Server(recognizer), [message])
                self.assertEqual(ws.sent, [{"type": "Error", "message": want}])
                self.assertTrue(ws.closed)

    async def test_slot(self):
        r = FakeRecognizer([])
        server = worker.Server(r)
        first, second = FakeWebSocket(), FakeWebSocket()
        sockets = [first, second]
        second.incoming.put_nowait({"type": "Audio", "pcm": FRAME})
        second.incoming.put_nowait(None)
        with mock.patch.object(worker.web, "WebSocketResponse", lambda: sockets.pop(0)):
            a = asyncio.create_task(server.asr_streaming(None))
            # The first session holds the slot while it waits for messages.
            while r.resets == 0:
                await asyncio.sleep(0)
            b = asyncio.create_task(server.asr_streaming(None))
            await asyncio.sleep(0)
            self.assertTrue(server.slot.locked())
            self.assertFalse(b.done())
            self.assertEqual((r.resets, second.incoming.qsize()), (1, 2))
            first.incoming.put_nowait(None)
            await asyncio.wait_for(asyncio.gather(a, b), 5)
        self.assertEqual((r.resets, r.frames, second.incoming.qsize()), (2, 1, 0))


if __name__ == "__main__":
    unittest.main()
