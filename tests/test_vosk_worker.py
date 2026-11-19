# Copyright 2026 Marc-Antoine Ruel. All rights reserved.
# Use of this source code is governed under the Apache License, Version 2.0
# that can be found in the LICENSE file.

# Offline tests for the Vosk worker's WebSocket framing and protocol errors.
import io
import json
import os
import struct
import unittest

from providers.vosk.vosksrv import worker as vosk_worker


def frame(opcode, data, fin=True, masked=True):
    n = len(data)
    first = (0x80 if fin else 0) | opcode
    if n < 126:
        head = struct.pack("!BB", first, (0x80 if masked else 0) | n)
    else:
        head = struct.pack("!BBH", first, (0x80 if masked else 0) | 126, n)
    if not masked:
        return head + data
    mask = os.urandom(4)
    return head + mask + bytes(b ^ mask[i % 4] for i, b in enumerate(data))


def frames(out):
    """Decodes the server's unmasked frames."""
    r = io.BytesIO(out)
    got = []
    while r.tell() < len(out):
        b0, b1 = r.read(2)
        n = b1 & 0x7F
        if n == 126:
            (n,) = struct.unpack("!H", r.read(2))
        elif n == 127:
            (n,) = struct.unpack("!Q", r.read(8))
        got.append((b0 & 0x0F, r.read(n)))
    return got


class Recognizer:
    def __init__(self, rate):
        self.rate = rate
        self.samples = 0

    def AcceptWaveform(self, data):  # noqa: N802
        self.samples += len(data) // 2
        return False

    def PartialResult(self):  # noqa: N802
        return json.dumps({"partial": str(self.samples)})

    def FinalResult(self):  # noqa: N802
        return json.dumps({"text": str(self.samples)})


CONFIG = frame(vosk_worker.OP_TEXT, b'{"config": {"sample_rate": 16000}}')


class FramingTest(unittest.TestCase):
    def test_fragmented_message_with_ping(self):
        r = io.BytesIO(
            frame(vosk_worker.OP_BINARY, b"\x01\x02", fin=False)
            + frame(vosk_worker.OP_PING, b"hi")
            + frame(vosk_worker.OP_CONTINUATION, b"\x03" * 300)
        )
        w = io.BytesIO()
        opcode, data = vosk_worker.read_message(r, w)
        self.assertEqual(opcode, vosk_worker.OP_BINARY)
        self.assertEqual(data, b"\x01\x02" + b"\x03" * 300)
        self.assertEqual(frames(w.getvalue()), [(vosk_worker.OP_PONG, b"hi")])

    def test_close_is_echoed(self):
        w = io.BytesIO()
        with self.assertRaises(vosk_worker.Closed):
            vosk_worker.read_message(io.BytesIO(frame(vosk_worker.OP_CLOSE, struct.pack("!H", 1000))), w)
        self.assertEqual(frames(w.getvalue()), [(vosk_worker.OP_CLOSE, struct.pack("!H", 1000))])

    def test_protocol_errors(self):
        cases = {
            "unmasked": frame(vosk_worker.OP_BINARY, b"\x00\x00", masked=False),
            "continuation first": frame(vosk_worker.OP_CONTINUATION, b"\x00\x00"),
            "reserved opcode": frame(0x3, b""),
            "fragmented control": frame(vosk_worker.OP_PING, b"", fin=False),
            "oversized": struct.pack("!BBQ", 0x82, 0xFF, vosk_worker.MAX_MESSAGE_BYTES + 1),
        }
        for name, data in cases.items():
            with self.subTest(name), self.assertRaises(vosk_worker.ProtocolError):
                vosk_worker.read_message(io.BytesIO(data), io.BytesIO())


class RecognizeTest(unittest.TestCase):
    def recognize(self, data):
        w = io.BytesIO()
        vosk_worker.recognize(io.BytesIO(data), w, Recognizer)
        return frames(w.getvalue())

    def test_utterance(self):
        got = self.recognize(
            CONFIG + frame(vosk_worker.OP_BINARY, b"\x00" * 4) + frame(vosk_worker.OP_TEXT, b'{"eof" : 1}')
        )
        self.assertEqual(
            got,
            [
                (vosk_worker.OP_TEXT, b'{"partial": "2"}'),
                (vosk_worker.OP_TEXT, b'{"text": "2"}'),
                (vosk_worker.OP_CLOSE, struct.pack("!H", 1000)),
            ],
        )

    def test_protocol_errors(self):
        audio = frame(vosk_worker.OP_BINARY, b"\x00\x00")
        cases = {
            "audio before config": audio,
            "eof before config": frame(vosk_worker.OP_TEXT, b'{"eof" : 1}'),
            "config after audio": CONFIG + audio + CONFIG,
            "unknown config": frame(vosk_worker.OP_TEXT, b'{"config": {"sample_rate": 16000, "words": true}}'),
            "malformed config": frame(vosk_worker.OP_TEXT, b'{"config": {"sample_rate": "fast"}}'),
            "partial sample": CONFIG + frame(vosk_worker.OP_BINARY, b"\x00"),
        }
        for name, data in cases.items():
            with self.subTest(name), self.assertRaises(vosk_worker.ProtocolError):
                self.recognize(data)


if __name__ == "__main__":
    unittest.main()
