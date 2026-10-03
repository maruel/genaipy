#!/usr/bin/env python3
# Copyright 2026 Marc-Antoine Ruel. All rights reserved.
# Use of this source code is governed under the Apache License, Version 2.0
# that can be found in the LICENSE file.

# KittenTTS HTTP worker serving chunked 24 kHz mono S16LE speech.

import argparse
import json
import os
import sys
import threading
import traceback
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import numpy as np
from kittenml import KittenTTS, KittenTTS2


@dataclass(frozen=True)
class Request:
    text: str
    voice: str

    @classmethod
    def from_json(cls, raw: Any) -> "Request":
        if not isinstance(raw, dict):
            raise ValueError("request must be an object")
        text = raw.get("text")
        voice = raw.get("voice")
        if not isinstance(text, str):
            raise ValueError("text must be a string")
        if not isinstance(voice, str) or not voice:
            raise ValueError("voice is required")
        return cls(text=text, voice=voice)


class Handler(BaseHTTPRequestHandler):
    model: KittenTTS2
    # Serialize the upstream model's generation state and reference cache.
    model_lock = threading.Lock()
    protocol_version = "HTTP/1.1"

    def do_POST(self) -> None:
        if self.path != "/synthesize":
            self.send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        try:
            request = self.read_request()
        except (ValueError, TypeError, OverflowError) as exc:
            self.send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return

        with self.model_lock:
            try:
                stream = self.model.generate_stream(text=request.text, voice=request.voice, normalize=True)
                first_audio = next(stream, None)
            except Exception as exc:  # noqa: BLE001
                self.send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})
                return

            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "audio/pcm")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            try:
                if first_audio is not None:
                    self.write_pcm(first_audio)
                for audio in stream:
                    self.write_pcm(audio)
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                # The caller stopped iteration or cancelled the HTTP request.
                self.close_connection = True
            except Exception as exc:  # noqa: BLE001
                # Omitting the terminal chunk makes HTTP clients report a
                # truncated transfer instead of accepting partial synthesis.
                self.close_connection = True
                print(f"synthesize stream failed: {exc}\n{traceback.format_exc()}", file=sys.stderr)

    def read_request(self) -> Request:
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            raise ValueError("missing Content-Length")
        length = int(raw_length)
        if length < 0 or length > (1 << 20):
            raise ValueError(f"invalid Content-Length {length}")
        payload = self.rfile.read(length)
        if len(payload) != length:
            raise ValueError("truncated request")
        return Request.from_json(json.loads(payload))

    def send_json(self, status: HTTPStatus, message: dict[str, Any]) -> None:
        payload = json.dumps(message, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        # Invalid request bodies may be unread; never interpret them as another request.
        self.send_header("Connection", "close")
        self.close_connection = True
        self.end_headers()
        self.wfile.write(payload)

    def write_pcm(self, audio: np.ndarray) -> None:
        pcm = np.clip(audio, -1.0, 1.0)
        payload = (pcm * 32767.0).astype("<i2", copy=False).tobytes()
        if payload:
            self.wfile.write(f"{len(payload):x}\r\n".encode("ascii"))
            self.wfile.write(payload)
            self.wfile.write(b"\r\n")
            self.wfile.flush()

    def log_message(self, fmt: str, *args: Any) -> None:
        print(fmt % args, file=sys.stderr)


def write_control(message: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(message, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def exit_on_stdin_eof() -> None:
    # EOF ends the worker even if only the uv launcher is killed.
    sys.stdin.buffer.read()
    os._exit(0)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the genaipy KittenTTS HTTP worker.")
    parser.add_argument("--cache-dir", required=True, help="Directory for Hugging Face model cache.")
    parser.add_argument("--model", required=True, help="KittenTTS Hugging Face model ID.")
    args = parser.parse_args()

    Handler.model = KittenTTS(args.model, cache_dir=args.cache_dir, device="cpu")
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    host, port = server.server_address
    threading.Thread(target=exit_on_stdin_eof, daemon=True).start()
    write_control(
        {
            "kind": "ready",
            "url": f"http://{host}:{port}",
            "voices": Handler.model.available_voices,
            "sample_rate": Handler.model.sample_rate,
        }
    )
    try:
        server.serve_forever()
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
