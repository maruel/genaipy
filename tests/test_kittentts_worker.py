# Copyright 2026 Marc-Antoine Ruel. All rights reserved.
# Use of this source code is governed under the Apache License, Version 2.0
# that can be found in the LICENSE file.

# Offline KittenTTS worker request validation and streaming failure tests.

import contextlib
import http.client
import importlib.util
import io
import json
import pathlib
import sys
import threading
import types
import unittest
from unittest import mock


class Audio:
    def __init__(self, pcm):
        self.pcm = pcm

    def __mul__(self, _scale):
        return self

    def astype(self, _dtype, *, copy):
        return self

    def tobytes(self):
        return self.pcm


class FakeModel:
    def __init__(self, mode="normal"):
        self.mode = mode
        self.requests = []
        self.available_voices = ["Luna"]
        self.sample_rate = 24000

    def generate_stream(self, text, voice, normalize):
        self.requests.append({"text": text, "voice": voice, "normalize": normalize})
        if self.mode == "first failure":
            raise RuntimeError("inference failed")
        if self.mode == "empty":
            return
        yield Audio(b"\x01\x00\x02\x00")
        if self.mode == "stream failure":
            raise RuntimeError("inference failed midstream")
        yield Audio(b"")
        yield Audio(b"\x03\x00")


def load_worker():
    numpy = types.ModuleType("numpy")
    numpy.ndarray = Audio
    numpy.clip = lambda audio, _min, _max: audio
    kitten = types.ModuleType("kittenml")
    kitten.KittenTTS = FakeModel
    kitten.KittenTTS2 = FakeModel
    path = pathlib.Path(__file__).resolve().parents[1] / "kittentts" / "worker.py"
    spec = importlib.util.spec_from_file_location("genaipy_test_kitten_worker", path)
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, {"numpy": numpy, "kittenml": kitten, spec.name: module}):
        spec.loader.exec_module(module)
    return module


worker = load_worker()


class WorkerTest(unittest.TestCase):
    def serve(self, model):
        class Handler(worker.Handler):
            model_lock = threading.Lock()

            def log_message(self, _fmt, *_args):
                return

        Handler.model = model
        server = worker.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        def cleanup():
            server.shutdown()
            thread.join()
            server.server_close()

        self.addCleanup(cleanup)
        connection = http.client.HTTPConnection(*server.server_address, timeout=3)
        self.addCleanup(connection.close)
        return connection

    def post(self, connection, raw):
        connection.request("POST", "/synthesize", json.dumps(raw))
        return connection.getresponse()

    def test_streaming(self):
        model = FakeModel()
        connection = self.serve(model)
        response = self.post(connection, {"text": "hello", "voice": "Luna"})
        self.assertEqual(response.status, 200)
        self.assertEqual(response.getheader("Content-Type"), "audio/pcm")
        self.assertTrue(response.chunked)
        self.assertEqual(response.read(), b"\x01\x00\x02\x00\x03\x00")
        self.assertEqual(model.requests, [{"text": "hello", "voice": "Luna", "normalize": True}])

    def test_empty_synthesis(self):
        response = self.post(self.serve(FakeModel("empty")), {"text": "emoji", "voice": "Luna"})
        self.assertEqual(response.status, 200)
        self.assertEqual(response.read(), b"")

    def test_first_inference_failure(self):
        response = self.post(self.serve(FakeModel("first failure")), {"text": "hello", "voice": "Luna"})
        self.assertEqual(response.status, 500)
        self.assertEqual(json.loads(response.read())["error"], "inference failed")

    def test_midstream_inference_failure(self):
        with contextlib.redirect_stderr(io.StringIO()):
            response = self.post(self.serve(FakeModel("stream failure")), {"text": "hello", "voice": "Luna"})
            self.assertEqual(response.status, 200)
            with self.assertRaises(http.client.IncompleteRead) as caught:
                response.read()
        self.assertEqual(caught.exception.partial, b"\x01\x00\x02\x00")

    def test_bad_request(self):
        model = FakeModel()
        response = self.post(self.serve(model), {"text": 3, "voice": "Luna"})
        self.assertEqual(response.status, 400)
        self.assertIn("text", json.loads(response.read())["error"])
        self.assertEqual(model.requests, [])

    def test_request_validation(self):
        valid = {"text": "hello", "voice": "Luna"}
        for raw in ([], {**valid, "voice": ""}, {**valid, "voice": None}, {**valid, "text": None}):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                worker.Request.from_json(raw)

    def test_configured_model(self):
        server = mock.Mock()
        server.server_address = ("127.0.0.1", 8765)
        model = FakeModel()
        model.sample_rate = 48000
        output = io.StringIO()
        with (
            mock.patch.object(sys, "argv", ["worker.py", "--model", "owner/model", "--cache-dir", "/tmp/owner/cache"]),
            mock.patch.object(worker, "KittenTTS", return_value=model) as factory,
            mock.patch.object(worker, "ThreadingHTTPServer", return_value=server),
            mock.patch.object(worker.threading, "Thread"),
            contextlib.redirect_stdout(output),
        ):
            self.assertEqual(worker.main(), 0)
        factory.assert_called_once_with("owner/model", cache_dir="/tmp/owner/cache", device="cpu")
        ready = json.loads(output.getvalue())
        self.assertEqual(ready["sample_rate"], 48000)
        self.assertEqual(ready["voices"], ["Luna"])
        server.server_close.assert_called_once_with()
