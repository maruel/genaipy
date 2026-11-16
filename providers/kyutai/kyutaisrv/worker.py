#!/usr/bin/env python3
# Copyright 2026 Marc-Antoine Ruel. All rights reserved.
# Use of this source code is governed under the Apache License, Version 2.0
# that can be found in the LICENSE file.

# Kyutai STT worker serving the moshi-server /api/asr-streaming subset the kyutai client uses.

import argparse
import asyncio
import json
import math
import os
import socket
import sys
import threading
from collections import deque
from pathlib import Path
from typing import Any, Protocol

import msgpack
import sentencepiece
from aiohttp import WSMsgType, web
from huggingface_hub import snapshot_download

SAMPLE_RATE = 24000
# Mimi encodes one 80 ms frame per model step.
FRAME_SIZE = 1920
FRAME_RATE = SAMPLE_RATE / FRAME_SIZE
# Text tokens that end a word, as moshi-core's asr.rs decodes them.
END_OF_PADDING = 0
PADDING = 3
# moshi_mlx preallocates its token history; bound an MLX session to 10 minutes.
MLX_MAX_STEPS = int(600 * FRAME_RATE)


class Recognizer(Protocol):
    # Steps between a frame and the text token that transcribes it.
    delay_steps: int

    def reset(self) -> None: ...

    def step(self, frame: list[float]) -> int:
        """Consumes one frame and returns the text token of the step."""
        ...

    def decode(self, tokens: list[int]) -> str: ...


class Session:
    """Transcribes one stream like moshi-server's ASR module.

    Words follow rust/moshi-core/src/asr.rs: text tokens accumulate until a
    padding token ends the word. A Marker returns once the delay has passed
    over the audio sent before it, as in batched_asr.rs, so the client flushes
    by sending that much silence, which this processes as fast as the model
    runs.
    """

    def __init__(self, recognizer: Recognizer) -> None:
        recognizer.reset()
        self.recognizer = recognizer
        self.pcm: list[float] = []
        self.steps = 0
        # (step at which to return, marker id), in arrival order.
        self.markers: deque[tuple[int, int]] = deque()
        self.word: list[int] = []
        self.start_time = 0.0

    def audio(self, pcm: list[float]) -> list[dict[str, Any]]:
        self.pcm += pcm
        out: list[dict[str, Any]] = []
        n = len(self.pcm) // FRAME_SIZE
        for i in range(n):
            self.step(self.pcm[i * FRAME_SIZE : (i + 1) * FRAME_SIZE], out)
        self.pcm = self.pcm[n * FRAME_SIZE :]
        return out

    def marker(self, marker_id: int) -> None:
        # Audio is stepped on arrival; only a partial frame remains buffered.
        self.markers.append((self.steps + self.recognizer.delay_steps, marker_id))

    def step(self, frame: list[float], out: list[dict[str, Any]]) -> None:
        token = self.recognizer.step(frame)
        self.steps += 1
        delay = self.recognizer.delay_steps
        if self.steps >= delay:
            if token in (END_OF_PADDING, PADDING):
                if self.word:
                    text = self.recognizer.decode(self.word)
                    out.append({"type": "Word", "text": text, "start_time": self.start_time})
                    self.word = []
            else:
                self.word.append(token)
            if token == END_OF_PADDING:
                self.start_time = (self.steps - delay) / FRAME_RATE
        while self.markers and self.markers[0][0] <= self.steps:
            out.append({"type": "Marker", "id": self.markers.popleft()[1]})


def delay_steps(config: dict[str, Any]) -> int:
    return math.ceil(config["stt_config"]["audio_delay_seconds"] * FRAME_RATE)


class TorchRecognizer:
    """Runs the PyTorch checkpoint with moshi, on CUDA when available."""

    def __init__(self, local_dir: Path, config: dict[str, Any]) -> None:
        # Imported here: moshi is installed only on the PyTorch platforms.
        import torch
        from moshi.models import LMGen, loaders

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        info = loaders.CheckpointInfo.from_hf_repo(
            "",
            moshi_weights=local_dir / (config.get("moshi_name") or "model.safetensors"),
            mimi_weights=local_dir / config["mimi_name"],
            tokenizer=local_dir / config["tokenizer_name"],
            config_path=local_dir / "config.json",
        )
        self.delay_steps = delay_steps(config)
        self.mimi = info.get_mimi(device=self.device)
        self.lm_gen = LMGen(info.get_moshi(device=self.device, dtype=torch.bfloat16), temp=0, temp_text=0)
        self.tokenizer = info.get_text_tokenizer()
        self.mimi.streaming_forever(1)
        self.lm_gen.streaming_forever(1)

    def reset(self) -> None:
        self.mimi.reset_streaming()
        self.lm_gen.reset_streaming()

    def step(self, frame: list[float]) -> int:
        with self.torch.no_grad():
            codes = self.mimi.encode(self.torch.tensor(frame, device=self.device)[None, None])
            tokens = self.lm_gen.step(codes)
        return int(tokens[0, 0, 0])

    def decode(self, tokens: list[int]) -> str:
        return self.tokenizer.decode(tokens)


class MLXRecognizer:
    """Runs the MLX checkpoint with moshi_mlx on Apple silicon."""

    def __init__(self, local_dir: Path, config: dict[str, Any]) -> None:
        # Imported here: moshi_mlx is installed only on macOS.
        import mlx.core as mx
        from moshi_mlx import models, utils

        self.mx = mx
        self.models = models
        self.utils = utils
        self.delay_steps = delay_steps(config)
        self.model = models.Lm(models.LmConfig.from_config_dict(config))
        self.model.set_dtype(mx.bfloat16)
        self.model.load_weights(str(local_dir / (config.get("moshi_name") or "model.safetensors")), strict=True)
        self.mimi = models.mimi.Mimi(models.mimi_202407(config["n_q"] - config["dep_q"]))
        self.mimi.load_pytorch_weights(str(local_dir / config["mimi_name"]), strict=True)
        self.tokenizer = sentencepiece.SentencePieceProcessor(str(local_dir / config["tokenizer_name"]))
        self.model.warmup()

    def reset(self) -> None:
        self.mimi.reset_all()
        for c in self.model.transformer_cache:
            c.reset()
        self.gen = self.models.LmGen(
            model=self.model,
            max_steps=MLX_MAX_STEPS,
            text_sampler=self.utils.Sampler(top_k=25, temp=0),
            audio_sampler=self.utils.Sampler(top_k=250, temp=0.8),
            check=False,
        )

    def step(self, frame: list[float]) -> int:
        codes = self.mimi.encode_step(self.mx.array(frame)[None, None]).transpose(0, 2, 1)
        return int(self.gen.step(codes[0])[0].item())

    def decode(self, tokens: list[int]) -> str:
        return self.tokenizer.decode(tokens)


class Server:
    """Serves /api/asr-streaming; one session decodes at a time.

    The model's streaming state has batch size 1, so later sessions wait for
    the slot instead of failing with moshi-server's "no free channels".
    Messages are msgpack maps tagged by "type": Audio carries 24 kHz float
    PCM and Marker an id; Word, Marker and Error answer. Other moshi-server
    messages are not sent. The kyutai-api-key header is not checked: the
    worker listens on loopback only.
    """

    def __init__(self, recognizer: Recognizer) -> None:
        self.recognizer = recognizer
        self.slot = asyncio.Lock()

    async def asr_streaming(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        async with self.slot:
            try:
                await self.transcribe(ws)
            except Exception as exc:  # noqa: BLE001
                if not ws.closed:
                    await send(ws, {"type": "Error", "message": str(exc)})
                    await ws.close()
        return ws

    async def transcribe(self, ws: web.WebSocketResponse) -> None:
        session = await asyncio.to_thread(Session, self.recognizer)
        async for msg in ws:
            if msg.type != WSMsgType.BINARY:
                # moshi-server ignores text frames.
                continue
            m = msgpack.unpackb(msg.data)
            kind = m.get("type") if isinstance(m, dict) else None
            if kind == "Audio":
                out = await asyncio.to_thread(session.audio, m["pcm"])
            elif kind == "Marker":
                session.marker(int(m["id"]))
                out = []
            else:
                raise ValueError(f"unsupported message type {kind!r}")
            for o in out:
                await send(ws, o)


async def send(ws: web.WebSocketResponse, message: dict[str, Any]) -> None:
    await ws.send_bytes(msgpack.packb(message, use_single_float=True))


def write_control(message: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(message, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def exit_on_stdin_eof() -> None:
    # EOF ends the worker even if only the uv launcher is killed.
    sys.stdin.buffer.read()
    os._exit(0)


async def serve(recognizer: Recognizer) -> None:
    app = web.Application()
    app.router.add_get("/api/asr-streaming", Server(recognizer).asr_streaming)
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    host, port = sock.getsockname()
    await web.SockSite(runner, sock).start()
    write_control({"kind": "ready", "url": f"http://{host}:{port}", "sample_rate": SAMPLE_RATE})
    await asyncio.Event().wait()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the genaipy Kyutai STT worker.")
    parser.add_argument("--cache-dir", required=True, help="Directory for Hugging Face model cache.")
    parser.add_argument("--model", required=True, help="Kyutai STT Hugging Face repository.")
    parser.add_argument("--revision", required=True, help="Hugging Face commit of the repository.")
    parser.add_argument("--backend", required=True, choices=["mlx", "torch"])
    args = parser.parse_args()

    local_dir = Path(snapshot_download(args.model, revision=args.revision, cache_dir=args.cache_dir))
    config = json.loads((local_dir / "config.json").read_text())
    recognizer: Recognizer
    if args.backend == "mlx":
        recognizer = MLXRecognizer(local_dir, config)
    else:
        recognizer = TorchRecognizer(local_dir, config)
    # Warm up before reporting readiness, as moshi-server does.
    Session(recognizer).audio([0.0] * 2 * FRAME_SIZE)
    threading.Thread(target=exit_on_stdin_eof, daemon=True).start()
    asyncio.run(serve(recognizer))
    return 0


if __name__ == "__main__":
    sys.exit(main())
