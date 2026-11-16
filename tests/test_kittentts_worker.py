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
import tempfile
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

    def generate_stream(self, text, voice, **options):
        self.requests.append({"text": text, "voice": voice, **options})
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
    numpy.float32 = "float32"
    numpy.zeros = lambda n, dtype: Audio(b"\x00\x00" * n)
    kitten = types.ModuleType("kittenml")
    kitten.KittenTTS = FakeModel
    get_model = types.ModuleType("kittenml.get_model")
    get_model._kitten2_files = lambda config: sorted(config["files"])
    tiny = types.ModuleType("kittenml.kittentts2_tiny.model")
    # 2 samples of silence at the fake 24 kHz rate.
    tiny.GAP_S = 2 / 24000
    repo = types.ModuleType("kittenml.repo")
    repo.KITTEN2_TINY_TYPE = "KITTEN2_TINY"
    hub = types.ModuleType("huggingface_hub")
    hub.hf_hub_download = None
    hub.snapshot_download = None
    path = pathlib.Path(__file__).resolve().parents[1] / "providers" / "kittentts" / "kittenttssrv" / "worker.py"
    spec = importlib.util.spec_from_file_location("genaipy_test_kitten_worker", path)
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(
        sys.modules,
        {
            "numpy": numpy,
            "huggingface_hub": hub,
            "kittenml": kitten,
            "kittenml.get_model": get_model,
            "kittenml.kittentts2_tiny.model": tiny,
            "kittenml.repo": repo,
            spec.name: module,
        },
    ):
        spec.loader.exec_module(module)
    return module


worker = load_worker()


class WorkerTest(unittest.TestCase):
    def serve(self, model, synthesizer=None):
        class Handler(worker.Handler):
            model_lock = threading.Lock()

            def log_message(self, _fmt, *_args):
                return

        Handler.synthesize = staticmethod((synthesizer or worker.kitten2_synthesizer)(model))
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

    def test_language_normalization(self):
        # Upstream normalization is English-tuned; other languages are spoken verbatim.
        for language, normalize in (("en", True), ("fr", False)):
            with self.subTest(language=language):
                model = FakeModel()
                response = self.post(self.serve(model), {"text": "1 €", "voice": "French", "language": language})
                self.assertEqual(response.status, 200)
                response.read()
                self.assertEqual(model.requests, [{"text": "1 €", "voice": "French", "normalize": normalize}])

    def test_tiny_language(self):
        # Nano, pico and mini take the language; an empty one means English.
        for language, want in (("", "en"), ("de", "de")):
            with self.subTest(language=language):
                model = FakeModel()
                connection = self.serve(model, worker.tiny_synthesizer)
                response = self.post(connection, {"text": "1 €", "voice": "Anna", "language": language})
                self.assertEqual(response.status, 200)
                response.read()
                self.assertEqual(model.requests, [{"text": "1 €", "voice": "Anna", "language": want}])

    def test_tiny_gap(self):
        # Chunks are separated by silence, as upstream generate() does.
        response = self.post(self.serve(FakeModel(), worker.tiny_synthesizer), {"text": "a. b.", "voice": "Anna"})
        self.assertEqual(response.status, 200)
        self.assertEqual(response.read(), b"\x01\x00\x02\x00" + b"\x00\x00" * 4 + b"\x03\x00")

    def test_tiny_first_inference_failure(self):
        model = FakeModel("first failure")
        response = self.post(self.serve(model, worker.tiny_synthesizer), {"text": "hello", "voice": "nobody"})
        self.assertEqual(response.status, 500)
        self.assertEqual(json.loads(response.read())["error"], "inference failed")

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
        for raw in (
            [],
            {**valid, "voice": ""},
            {**valid, "voice": None},
            {**valid, "text": None},
            {**valid, "language": 1},
        ):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                worker.Request.from_json(raw)

    def test_configured_model(self):
        config = pathlib.Path(self.enterContext(tempfile.TemporaryDirectory())) / "config.json"
        config.write_text(json.dumps({"files": ["lm/*", "config.json"]}))
        server = mock.Mock()
        server.server_address = ("127.0.0.1", 8765)
        model = FakeModel()
        model.sample_rate = 48000
        output = io.StringIO()
        with (
            mock.patch.object(
                sys,
                "argv",
                ["worker.py", "--model", "owner/model", "--revision", "0123abcd", "--cache-dir", "/tmp/owner/cache"],
            ),
            mock.patch.object(worker, "hf_hub_download", return_value=str(config)) as get_config,
            mock.patch.object(worker, "snapshot_download", return_value="/tmp/owner/snapshot") as snapshot,
            mock.patch.object(worker, "KittenTTS", return_value=model) as factory,
            mock.patch.object(worker, "ThreadingHTTPServer", return_value=server),
            mock.patch.object(worker.threading, "Thread"),
            contextlib.redirect_stdout(output),
        ):
            self.assertEqual(worker.main(), 0)
        # Both downloads are pinned; KittenTTS itself would fetch main.
        get_config.assert_called_once_with(
            "owner/model", "config.json", revision="0123abcd", cache_dir="/tmp/owner/cache"
        )
        snapshot.assert_called_once_with(
            "owner/model", revision="0123abcd", cache_dir="/tmp/owner/cache", allow_patterns=["config.json", "lm/*"]
        )
        factory.assert_called_once_with("/tmp/owner/snapshot", cache_dir="/tmp/owner/cache", device="cpu")
        ready = json.loads(output.getvalue())
        self.assertEqual(ready["sample_rate"], 48000)
        self.assertEqual(ready["voices"], ["Luna"])
        server.server_close.assert_called_once_with()

    def test_configured_tiny_model(self):
        config = pathlib.Path(self.enterContext(tempfile.TemporaryDirectory())) / "config.json"
        config.write_text(
            json.dumps(
                {
                    "type": "KITTEN2_TINY",
                    "languages": {
                        "de": {"model_file": "m.onnx", "voices": "voices_de.npz"},
                        "en": {"model_file": "m.onnx", "voices": "voices_en.npz"},
                    },
                }
            )
        )
        server = mock.Mock()
        server.server_address = ("127.0.0.1", 8765)
        model = FakeModel()
        model.languages = ["en", "de"]
        model.available_voices = {"en": ["Eleanor", "Otis"], "de": ["Anna", "Eleanor"]}
        model.phonemize = mock.Mock()
        output = io.StringIO()
        with (
            mock.patch.object(
                sys,
                "argv",
                ["worker.py", "--model", "owner/tiny", "--revision", "0123abcd", "--cache-dir", "/tmp/owner/cache"],
            ),
            mock.patch.object(worker, "hf_hub_download", return_value=str(config)),
            mock.patch.object(worker, "snapshot_download", return_value="/tmp/owner/snapshot") as snapshot,
            mock.patch.object(worker, "KittenTTS", return_value=model),
            mock.patch.object(worker, "ThreadingHTTPServer", return_value=server),
            mock.patch.object(worker.threading, "Thread"),
            contextlib.redirect_stdout(output),
        ):
            self.assertEqual(worker.main(), 0)
        # Only the ONNX runtime's files; the PyTorch weights are skipped.
        snapshot.assert_called_once_with(
            "owner/tiny",
            revision="0123abcd",
            cache_dir="/tmp/owner/cache",
            allow_patterns=["config.json", "g2p/*", "g2p/**", "m.onnx", "vocab.json", "voices_de.npz", "voices_en.npz"],
        )
        model.phonemize.assert_called_once_with("Hello.", "en")
        self.assertEqual(json.loads(output.getvalue())["voices"], ["Anna", "Eleanor", "Otis"])
