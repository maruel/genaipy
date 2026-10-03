#!/usr/bin/env python3
# Copyright 2026 Marc-Antoine Ruel. All rights reserved.
# Use of this source code is governed under the Apache License, Version 2.0
# that can be found in the LICENSE file.

"""Offline HTTP protocol regressions; model libraries are replaced at import."""

import http.server
import importlib
import io
import json
import os
import sys
import threading
import types
import unittest
import urllib.error
import urllib.request
from unittest import mock


def load_runtime(name):
    torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False),
        backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False)),
        float16="float16",
        float32="float32",
    )
    dependencies = {lib: types.ModuleType(lib) for lib in ("diffusers", "huggingface_hub", "segmoe", "transformers")}
    dependencies["torch"] = torch
    with mock.patch.dict(sys.modules, dependencies):
        return importlib.import_module(name)


class ServerTests(unittest.TestCase):
    def start(self, module):
        server = http.server.HTTPServer(("127.0.0.1", 0), module.Handler)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        self.addCleanup(server.server_close)
        self.addCleanup(worker.join, 2)
        self.addCleanup(server.shutdown)
        return f"http://127.0.0.1:{server.server_port}"

    def request(self, url, body=None, headers=None):
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json", **(headers or {})})
        # Ignore proxies inherited from the test runner.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            return opener.open(req, timeout=2)
        except urllib.error.HTTPError as err:
            return err

    def test_llm_sync_stream_health_and_shutdown(self):
        module = load_runtime("llm")
        pipeline = mock.Mock(return_value=[{"generated_text": "Hello"}])
        with mock.patch.object(module.Handler, "_pipe", pipeline):
            root = self.start(module)
            with self.request(root + "/health") as response:
                self.assertEqual(json.load(response), {"status": "ok", "pid": os.getpid()})
            self.assertEqual(pipeline.call_count, 0)
            for stream in (False, True):
                body = json.dumps({"messages": [{"role": "user", "content": "Hi"}], "stream": stream}).encode()
                with self.request(root + "/v1/chat/completions", body) as response:
                    self.assertEqual(response.status, 200)
                    if stream:
                        self.assertEqual(response.headers["Content-Type"], "text/event-stream")
                        frames = response.read().decode().split("\n\n")
                        self.assertEqual(
                            json.loads(frames[0].removeprefix("data: "))["choices"][0]["delta"]["content"], "Hello"
                        )
                        self.assertEqual(frames[1], "data: [DONE]")
                    else:
                        self.assertEqual(json.load(response)["choices"][0]["message"]["content"], "Hello")
            with self.request(root + "/api/quit", b"{}") as response:
                self.assertEqual(json.load(response), {"quitting": True})

    def test_llm_rejects_invalid_requests_and_recovers(self):
        module = load_runtime("llm")
        pipeline = mock.Mock(return_value=[{"generated_text": "Hello"}])
        with mock.patch.object(module.Handler, "_pipe", pipeline):
            root = self.start(module)
            for body in (
                b"{",
                b"[]",
                b"{}",
                b"\xff",
                b'{"content":"\xff"}',
                b'{"messages":[{"role":"user","content":"Hi"}],"max_tokens":0}',
            ):
                with self.request(root + "/v1/chat/completions", body) as response:
                    self.assertEqual(response.status, 400)
            for length in ("-1", "invalid", "1048577"):
                with self.request(root + "/v1/chat/completions", b"x", {"Content-Length": length}) as response:
                    self.assertEqual(response.status, 400)
            pipeline.assert_not_called()
            pipeline.side_effect = RuntimeError("model failure")
            body = b'{"messages":[{"role":"user","content":"Hi"}]}'
            with self.request(root + "/v1/chat/completions", body) as response:
                self.assertEqual(response.status, 500)
            with self.request(root + "/health") as response:
                self.assertEqual(response.status, 200)

    def test_image_health_validation_and_model_failure(self):
        module = load_runtime("image_gen")
        root = self.start(module)
        with self.request(root + "/health") as response:
            self.assertEqual(json.load(response)["pid"], os.getpid())
        for body in (b"{", b"[]", b"{}", b"\xff", b'{"content":"\xff"}', b'{"message":"image","steps":-1,"seed":1}'):
            with self.request(root + "/api/generate", body) as response:
                self.assertEqual(response.status, 400)
        for length in ("-1", "invalid", "1048577"):
            with self.request(root + "/api/generate", b"x", {"Content-Length": length}) as response:
                self.assertEqual(response.status, 400)
        with mock.patch.object(module.Handler, "gen_image", side_effect=RuntimeError("model failure")):
            with self.request(root + "/api/generate", b'{"message":"image","steps":1,"seed":1}') as response:
                self.assertEqual(response.status, 500)
        with self.request(root + "/health") as response:
            self.assertEqual(response.status, 200)
        image = mock.Mock()
        image.save.side_effect = lambda output, **kwargs: (
            output.write(b"png") if isinstance(output, io.BytesIO) else None
        )
        with mock.patch.object(module.Handler, "gen_image", return_value=image):
            with self.request(root + "/api/generate", b'{"message":"image","steps":1,"seed":1}') as response:
                self.assertEqual(json.load(response)["image"], "cG5n")


if __name__ == "__main__":
    sys.exit(0 if unittest.main(exit=False).result.wasSuccessful() else 1)
