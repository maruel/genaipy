#!/usr/bin/env python3
# Copyright 2026 Marc-Antoine Ruel. All rights reserved.
# Use of this source code is governed under the Apache License, Version 2.0
# that can be found in the LICENSE file.

# Managed Vosk worker: the vosk-server WebSocket protocol on a loopback port.
#
# Each connection is one utterance with its own recognizer over the shared
# model, which Vosk allows to decode concurrently. A JSON text message
# {"config": {"sample_rate": N}} precedes binary S16LE PCM messages; each audio
# message and the final {"eof" : 1} receive one recognizer result. The server
# then closes the connection. The stdlib WebSocket server implements only what
# this protocol needs: unextended RFC 6455 framing without subprotocols.
import argparse
import base64
import hashlib
import http.server
import json
import os
import struct
import sys
import threading
from collections.abc import Callable
from typing import BinaryIO, Protocol

EOF_MESSAGE = '{"eof" : 1}'
MAX_MESSAGE_BYTES = 1 << 20
WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

OP_CONTINUATION = 0x0
OP_TEXT = 0x1
OP_BINARY = 0x2
OP_CLOSE = 0x8
OP_PING = 0x9
OP_PONG = 0xA


class Recognizer(Protocol):
    def AcceptWaveform(self, data: bytes) -> bool: ...  # noqa: N802 - Vosk's API.
    def Result(self) -> str: ...  # noqa: N802
    def PartialResult(self) -> str: ...  # noqa: N802
    def FinalResult(self) -> str: ...  # noqa: N802


class ProtocolError(ValueError):
    pass


class Closed(Exception):
    pass


def read_exact(r: BinaryIO, n: int) -> bytes:
    data = r.read(n)
    if len(data) != n:
        raise Closed()
    return data


def read_frame(r: BinaryIO) -> tuple[bool, int, bytes]:
    b0, b1 = read_exact(r, 2)
    if b0 & 0x70:
        raise ProtocolError("reserved bits are set")
    if not b1 & 0x80:
        raise ProtocolError("client frames must be masked")
    n = b1 & 0x7F
    if n == 126:
        (n,) = struct.unpack("!H", read_exact(r, 2))
    elif n == 127:
        (n,) = struct.unpack("!Q", read_exact(r, 8))
    if n > MAX_MESSAGE_BYTES:
        raise ProtocolError("message too large")
    mask = read_exact(r, 4)
    payload = read_exact(r, n)
    key = (mask * (n // 4 + 1))[:n]
    data = (int.from_bytes(payload, "little") ^ int.from_bytes(key, "little")).to_bytes(n, "little")
    return bool(b0 & 0x80), b0 & 0x0F, data


def write_frame(w: BinaryIO, opcode: int, data: bytes) -> None:
    n = len(data)
    if n < 126:
        head = struct.pack("!BB", 0x80 | opcode, n)
    elif n < 1 << 16:
        head = struct.pack("!BBH", 0x80 | opcode, 126, n)
    else:
        head = struct.pack("!BBQ", 0x80 | opcode, 127, n)
    w.write(head + data)
    w.flush()


def read_message(r: BinaryIO, w: BinaryIO) -> tuple[int, bytes]:
    """Returns the next text or binary message, answering pings meanwhile."""
    opcode, parts, size = None, [], 0
    while True:
        fin, op, data = read_frame(r)
        if op >= OP_CLOSE:
            if not fin or len(data) > 125:
                raise ProtocolError("invalid control frame")
            if op == OP_CLOSE:
                write_frame(w, OP_CLOSE, data[:2])
                raise Closed()
            if op == OP_PING:
                write_frame(w, OP_PONG, data)
            elif op != OP_PONG:
                raise ProtocolError(f"unknown opcode {op}")
            continue
        if (op == OP_CONTINUATION) != (opcode is not None):
            raise ProtocolError("unexpected continuation")
        if op not in (OP_CONTINUATION, OP_TEXT, OP_BINARY):
            raise ProtocolError(f"unknown opcode {op}")
        opcode = opcode if opcode is not None else op
        size += len(data)
        if size > MAX_MESSAGE_BYTES:
            raise ProtocolError("message too large")
        parts.append(data)
        if fin:
            return opcode, b"".join(parts)


def close(w: BinaryIO, code: int, reason: str) -> None:
    write_frame(w, OP_CLOSE, struct.pack("!H", code) + reason.encode()[:123])


def recognize(r: BinaryIO, w: BinaryIO, new_recognizer: Callable[[int], Recognizer]) -> None:
    """Serves one utterance; raises ProtocolError for a client error."""
    rec = None
    while True:
        opcode, data = read_message(r, w)
        if opcode == OP_TEXT:
            text = data.decode()
            if text == EOF_MESSAGE:
                if rec is None:
                    raise ProtocolError("eof before config")
                write_frame(w, OP_TEXT, rec.FinalResult().encode())
                close(w, 1000, "")
                return
            if rec is not None:
                raise ProtocolError("config after audio")
            try:
                config = json.loads(text)["config"]
                if set(config) != {"sample_rate"}:
                    raise ProtocolError("config supports only sample_rate")
                rec = new_recognizer(int(config["sample_rate"]))
            except (KeyError, TypeError, ValueError) as e:
                raise ProtocolError(f"invalid config: {e}") from e
            continue
        if rec is None:
            raise ProtocolError("audio before config")
        if len(data) % 2:
            raise ProtocolError("audio must be whole S16LE samples")
        result = rec.Result() if rec.AcceptWaveform(data) else rec.PartialResult()
        write_frame(w, OP_TEXT, result.encode())


class Server(http.server.ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, new_recognizer: Callable[[int], Recognizer]):
        super().__init__(("127.0.0.1", 0), Handler)
        self.new_recognizer = new_recognizer


class Handler(http.server.BaseHTTPRequestHandler):
    server: Server

    def do_GET(self) -> None:
        key = self.headers.get("Sec-WebSocket-Key", "")
        upgrade = self.headers.get("Upgrade", "").lower() == "websocket"
        if self.path != "/" or not upgrade or not key or self.headers.get("Sec-WebSocket-Version") != "13":
            self.send_error(400, "expected a WebSocket upgrade of /")
            return
        accept = base64.b64encode(hashlib.sha1((key + WS_GUID).encode()).digest()).decode()
        self.send_response(101)
        self.send_header("Upgrade", "websocket")
        self.send_header("Connection", "Upgrade")
        self.send_header("Sec-WebSocket-Accept", accept)
        self.end_headers()
        self.wfile.flush()
        self.close_connection = True
        try:
            recognize(self.rfile, self.wfile, self.server.new_recognizer)
        except Closed:
            pass
        except (BrokenPipeError, ConnectionResetError):
            # Cancellation only disconnects this utterance; the model stays loaded.
            pass
        except (ProtocolError, UnicodeDecodeError) as e:
            self.log_message("%s", e)
            self.try_close(1002, str(e))
        except Exception as e:  # noqa: BLE001 - Report a native failure to the client.
            self.log_message("recognition failed: %s", e)
            self.try_close(1011, str(e))

    def try_close(self, code: int, reason: str) -> None:
        try:
            close(self.wfile, code, reason)
        except OSError:
            pass

    def log_message(self, fmt: str, *args: object) -> None:
        print(fmt % args, file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    # EOF stops the Python child even if its uv launcher or Go parent dies.
    def lifeline() -> None:
        sys.stdin.buffer.read()
        os._exit(0)

    threading.Thread(target=lifeline, daemon=True).start()
    # Import only at worker startup so offline tests require no native package.
    import vosk

    model = vosk.Model(args.model)
    server = Server(lambda rate: vosk.KaldiRecognizer(model, rate))

    print(json.dumps({"url": f"http://127.0.0.1:{server.server_port}"}), flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
