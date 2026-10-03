# Copyright 2026 Marc-Antoine Ruel. All rights reserved.
# Use of this source code is governed under the Apache License, Version 2.0
# that can be found in the LICENSE file.

# Offline tests for Whistle PCM conversion, limits and native-engine serialization.
import array
import concurrent.futures
import http.client
import json
import struct
import threading
import time
import unittest

from whistle import worker as whistle_worker


class Model:
    def __init__(self):
        self.calls = []
        self.active = 0
        self.overlap = False

    def transcribe(self, audio):
        self.active += 1
        self.overlap |= self.active > 1
        self.calls.append(audio)
        if audio[0] == -1:
            self.active -= 1
            raise RuntimeError("native decoding failed")
        time.sleep(0.01)
        self.active -= 1
        return {"text": "hello"}


class WorkerTest(unittest.TestCase):
    def setUp(self):
        self.model = Model()
        self.server = whistle_worker.Server(self.model)
        self.thread = threading.Thread(target=self.server.serve_forever)
        self.thread.start()
        self.addCleanup(self.close_server)

    def close_server(self):
        self.server.shutdown()
        self.thread.join()
        self.server.server_close()

    def request(self, pcm, path="/transcribe"):
        conn = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=5)
        try:
            conn.request("POST", path, body=pcm)
            resp = conn.getresponse()
            return resp.status, resp.read()
        finally:
            conn.close()

    def test_pcm_conversion(self):
        status, data = self.request(struct.pack("<hhh", -16384, 0, 32767))
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(data), {"text": "hello"})
        audio = self.model.calls[0]
        self.assertIsInstance(audio, array.array)
        self.assertEqual(audio.typecode, "f")
        self.assertEqual(list(audio), [-0.5, 0, 32767 / 32768])

    def test_invalid_pcm_does_not_reach_model(self):
        for pcm in (b"", b"x"):
            with self.subTest(size=len(pcm)):
                self.assertEqual(self.request(pcm)[0], 400)
        self.assertEqual(self.model.calls, [])
        self.assertEqual(self.request(b"\0\0", "/other")[0], 404)

    def test_oversized_pcm_is_rejected_before_reading_body(self):
        conn = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=5)
        self.addCleanup(conn.close)
        conn.putrequest("POST", "/transcribe")
        conn.putheader("Content-Length", "960002")
        # Uploading a rejected body races the server's connection close on macOS.
        conn.endheaders()
        resp = conn.getresponse()
        self.assertEqual(resp.status, 400)
        self.assertIn("at most 30 seconds", json.loads(resp.read())["error"])
        self.assertEqual(self.model.calls, [])
        self.assertEqual(self.request(b"\0\0")[0], 200)

    def test_native_failure_leaves_worker_usable(self):
        status, data = self.request(struct.pack("<h", -32768))
        self.assertEqual(status, 400)
        self.assertIn("native decoding failed", json.loads(data)["error"])
        self.assertEqual(self.request(b"\0\0")[0], 200)

    def test_native_calls_are_serialized(self):
        with concurrent.futures.ThreadPoolExecutor(4) as pool:
            results = list(pool.map(self.request, [b"\0\0"] * 8))
        self.assertTrue(all(status == 200 for status, _ in results))
        self.assertEqual(len(self.model.calls), 8)
        self.assertFalse(self.model.overlap)


if __name__ == "__main__":
    unittest.main()
