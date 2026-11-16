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
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import numpy as np
from huggingface_hub import hf_hub_download, snapshot_download
from kittenml import KittenTTS
from kittenml.get_model import _kitten2_files
from kittenml.kittentts2_tiny.model import GAP_S
from kittenml.repo import KITTEN2_TINY_TYPE


@dataclass(frozen=True)
class Request:
    text: str
    voice: str
    # ISO 639-1 code of the spoken language; empty means English.
    language: str

    @classmethod
    def from_json(cls, raw: Any) -> "Request":
        if not isinstance(raw, dict):
            raise ValueError("request must be an object")
        text = raw.get("text")
        voice = raw.get("voice")
        language = raw.get("language", "")
        if not isinstance(text, str):
            raise ValueError("text must be a string")
        if not isinstance(voice, str) or not voice:
            raise ValueError("voice is required")
        if not isinstance(language, str):
            raise ValueError("language must be a string")
        return cls(text=text, voice=voice, language=language)


# Synthesizer streams the float audio chunks of a request.
Synthesizer = Callable[[Request], Iterator[np.ndarray]]


def kitten2_synthesizer(model: Any) -> Synthesizer:
    """Speaks through KittenTTS 2, whose voice carries the language."""

    def synthesize(request: Request) -> Iterator[np.ndarray]:
        # The upstream normalizer is English-tuned; its documentation says to
        # pass normalize=False for non-English text.
        normalize = request.language in ("", "en")
        return model.generate_stream(text=request.text, voice=request.voice, normalize=normalize)

    return synthesize


def tiny_synthesizer(model: Any) -> Synthesizer:
    """Speaks through KittenTTS 2 nano, pico or mini in the request language.

    Their generate_stream yields one chunk per sentence group; generate, which
    this mirrors, separates the chunks with GAP_S of silence.
    """
    gap = np.zeros(int(GAP_S * model.sample_rate), dtype=np.float32)

    def synthesize(request: Request) -> Iterator[np.ndarray]:
        stream = model.generate_stream(text=request.text, voice=request.voice, language=request.language or "en")
        for i, audio in enumerate(stream):
            if i:
                yield gap
            yield audio

    return synthesize


class Handler(BaseHTTPRequestHandler):
    synthesize: Synthesizer
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
                stream = iter(type(self).synthesize(request))
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


def download(model: str, revision: str, cache_dir: str) -> tuple[str, dict[str, Any]]:
    # KittenTTS takes no revision and downloads a repository ID from main, but
    # loads a local directory. Fetch the files it would select at revision.
    with open(hf_hub_download(model, "config.json", revision=revision, cache_dir=cache_dir)) as f:
        config = json.load(f)
    if config.get("type") == KITTEN2_TINY_TYPE:
        patterns = tiny_files(config)
    else:
        # The private selection helper is stable under the pinned kittenml.
        patterns = _kitten2_files(config)
    return snapshot_download(model, revision=revision, cache_dir=cache_dir, allow_patterns=patterns), config


def tiny_files(config: dict[str, Any]) -> list[str]:
    """The files a nano, pico or mini repository's ONNX runtime reads.

    The repositories also hold float32 PyTorch weights, which the CPU ONNX
    runtime does not load (kittentts2_tiny/model.py in the pinned kittenml).
    """
    g2p = config.get("g2p_dir", "g2p")
    wanted = {"config.json", config.get("vocab", "vocab.json"), f"{g2p}/*", f"{g2p}/**"}
    for entry in config["languages"].values():
        wanted.update((entry["model_file"], entry["voices"]))
    return sorted(wanted)


def load(local_dir: str, config: dict[str, Any], cache_dir: str) -> tuple[Synthesizer, list[str], int]:
    """Loads the model and returns its synthesizer, voice names and sample rate."""
    model = KittenTTS(local_dir, cache_dir=cache_dir, device="cpu")
    if config.get("type") != KITTEN2_TINY_TYPE:
        return kitten2_synthesizer(model), model.available_voices, model.sample_rate
    # The English front end loads its spaCy pipeline and G2P model on first
    # use, about 10 s on CPU; pay it before reporting readiness.
    if "en" in model.languages:
        model.phonemize("Hello.", "en")
    voices = sorted({name for names in model.available_voices.values() for name in names})
    return tiny_synthesizer(model), voices, model.sample_rate


def exit_on_stdin_eof() -> None:
    # EOF ends the worker even if only the uv launcher is killed.
    sys.stdin.buffer.read()
    os._exit(0)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the genaipy KittenTTS HTTP worker.")
    parser.add_argument("--cache-dir", required=True, help="Directory for Hugging Face model cache.")
    parser.add_argument("--model", required=True, help="KittenTTS Hugging Face model ID.")
    parser.add_argument("--revision", required=True, help="Hugging Face commit of the model.")
    args = parser.parse_args()

    local_dir, config = download(args.model, args.revision, args.cache_dir)
    synthesize, voices, sample_rate = load(local_dir, config, args.cache_dir)
    Handler.synthesize = staticmethod(synthesize)
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    host, port = server.server_address
    threading.Thread(target=exit_on_stdin_eof, daemon=True).start()
    write_control(
        {
            "kind": "ready",
            "url": f"http://{host}:{port}",
            "voices": voices,
            "sample_rate": sample_rate,
        }
    )
    try:
        server.serve_forever()
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
