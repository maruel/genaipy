#!/usr/bin/env python3
# Copyright 2024 Marc-Antoine Ruel. All rights reserved.
# Use of this source code is governed under the Apache License, Version 2.0
# that can be found in the LICENSE file.

"""Runs a LLM."""

import argparse
import http.server
import json
import logging
import os
import signal
import sys
import threading
import time

import huggingface_hub
import torch
import transformers

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE in ("cuda", "mps") else torch.float32


def get_generator(seed):
    """Returns a deterministic random number generator."""
    if DEVICE in ("cuda", "mps"):
        return torch.Generator(DEVICE).manual_seed(seed)
    return torch.Generator().manual_seed(seed)


def load_model(model: str):
    return transformers.pipeline(
        "text-generation",
        model=model,
        dtype=DTYPE,
        device_map="auto",
    )


def load_phi_3_mini():
    """Returns Phi-3 medium."""
    # https://huggingface.co/microsoft/Phi-3-mini-4k-instruct
    model_id = "microsoft/Phi-3-mini-4k-instruct"
    return transformers.pipeline(
        "text-generation",
        model=transformers.AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=DEVICE,
            torch_dtype="auto",
            attn_implementation="eager",
            trust_remote_code=True,
        ),
        tokenizer=transformers.AutoTokenizer.from_pretrained(model_id),
    )


def load_phi_3_medium():
    """Returns Phi-3 medium."""
    # https://huggingface.co/microsoft/Phi-3-medium-128k-instruct
    model_id = "microsoft/Phi-3-medium-128k-instruct"
    # attn_implementation="eager" ?
    return transformers.pipeline(
        "text-generation",
        model=transformers.AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=DEVICE,
            torch_dtype="auto",
            attn_implementation="eager",
            trust_remote_code=True,
        ),
        tokenizer=transformers.AutoTokenizer.from_pretrained(model_id),
    )


def load_mistral_nemo():
    """Loads Mistral Nemo.

    It's a 12B model with 128k context window. It requires a ton of RAM. It cannot
    run in float16 with a macOS system wit 36GiB of RAM. This model is designed to
    work well in fp8 but I don't think (?) pytorch metal mps implementation
    supports fp8. As of 2024-07-19, adding the tekken tokenizer to llama.cpp is
    being worked on.

    https://mistral.ai/news/mistral-nemo/ specifically mentions that using FP8
    inference is fine but they don't tell which form of FP8 (!!)

    See https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407
    """
    # As of 2024-07-19, this doesn't work. Filed
    # https://github.com/pytorch/pytorch/issues/131196
    return transformers.pipeline(
        "text-generation",
        model="mistralai/Mistral-Nemo-Instruct-2407",
        torch_dtype=torch.float8_e4m3fn,
    )


class Handler(http.server.BaseHTTPRequestHandler):
    _pipe = None

    def do_GET(self):
        try:
            if self.path == "/health":
                self.on_health()
            else:
                self.reply_json(
                    {
                        "error": {
                            "type": "unknown path",
                            "message": self.path,
                        }
                    },
                    status=400,
                )
        except Exception as e:
            self.reply_json(
                {
                    "error": {
                        "type": "exception",
                        "message": str(e),
                    }
                },
                status=500,
            )
            print(str(e), file=sys.stderr)

    def do_POST(self):
        try:
            logging.info("Got request %s", self.path)
            if self.path == "/v1/chat/completions":
                self.on_completions()
            elif self.path == "/api/quit":
                self.on_quit()
            else:
                self.reply_json(
                    {
                        "error": {
                            "type": "unknown path",
                            "message": self.path,
                        }
                    },
                    status=400,
                )
        except Exception as e:
            self.reply_json(
                {
                    "error": {
                        "type": "exception",
                        "message": str(e),
                    }
                },
                status=500,
            )
            print(str(e), file=sys.stderr)

    def reply_json(self, data, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("ascii"))

    def on_health(self):
        self.reply_json({"status": "ok", "pid": os.getpid()})

    def on_quit(self):
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self.reply_json({"error": {"message": "invalid Content-Length"}}, status=400)
            return
        if not 0 <= content_length <= 1048576:
            self.reply_json({"error": {"message": "request body must be at most 1 MiB"}}, status=400)
            return
        # Closing a socket with unread request data can reset the response on macOS.
        self.rfile.read(content_length)
        self.reply_json({"quitting": True})
        threading.Thread(target=self.server.shutdown, daemon=True).start()

    def on_completions(self):
        start = time.time()
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self.reply_json({"error": {"message": "invalid Content-Length"}}, status=400)
            return
        if not 0 < content_length <= 1048576:
            self.reply_json({"error": {"message": "request body must be between 1 byte and 1 MiB"}}, status=400)
            return
        post_data = self.rfile.read(content_length)
        try:
            data = json.loads(post_data.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            self.reply_json(
                {
                    "error": {
                        "type": "decode_error",
                        "message": str(e),
                    }
                },
                status=400,
            )
            return
        # TODO: Structured format and verifications.
        if not isinstance(data, dict):
            self.reply_json({"error": {"message": "request must be an object"}}, status=400)
            return
        stream = data.get("stream", False)
        prompt = data.get("messages")
        if (
            not isinstance(prompt, list)
            or not prompt
            or any(
                not isinstance(msg, dict)
                or msg.get("role") not in ("system", "user", "assistant")
                or not isinstance(msg.get("content"), str)
                for msg in prompt
            )
        ):
            self.reply_json(
                {"error": {"message": "messages must contain text chat messages"}},
                status=400,
            )
            return
        max_tokens = data.get("max_tokens", 500)
        if not isinstance(stream, bool) or type(max_tokens) is not int or max_tokens <= 0:
            self.reply_json({"error": {"message": "invalid stream or max_tokens"}}, status=400)
            return
        output = self._pipe(
            prompt,
            max_new_tokens=max_tokens,
            return_full_text=False,
            do_sample=False,
        )[0]["generated_text"]
        if stream:
            resp = {
                "choices": [
                    {
                        "delta": {
                            "content": output,
                        },
                        "finish_reason": "stop",
                    },
                ],
            }
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(b"data: " + json.dumps(resp).encode("ascii") + b"\n\n")
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        else:
            resp = {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": output,
                        },
                    },
                ],
            }
            self.reply_json(resp)
        logging.info("Generated text in %.1fs", time.time() - start)


def main():
    parser = argparse.ArgumentParser(description=sys.modules[__name__].__doc__, allow_abbrev=False)
    parser.add_argument("--token", help="Create a read token at htps://huggingface.co/settings/tokens")
    parser.add_argument(
        "--host",
        default="localhost",
        help="Host to listen to. Use 0.0.0.0 to listen on all IPs",
    )
    parser.add_argument("--port", default=8031, type=int)
    parser.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct", help="Hugging Face text model ID")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    if args.token:
        huggingface_hub.login(token=args.token)
    if DEVICE == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    Handler._pipe = load_model(args.model)
    # Handler._pipe = load_phi_3_mini()
    # Handler._pipe = load_mistral_nemo()
    logging.info("Model loaded using %s", DEVICE)
    httpd = http.server.HTTPServer((args.host, args.port), Handler)
    logging.info(f"Started server on port {args.host}:{args.port}")

    def handle(signum, frame):
        # It tried various way to quick cleanly but it's just a pain.
        # TODO: Figure out how to do it the right way.
        logging.info("Got signal")
        # httpd.server_close()
        # logging.info("Did server_close")
        sys.exit(0)

    signal.signal(signal.SIGINT, handle)
    signal.signal(signal.SIGTERM, handle)

    try:
        httpd.serve_forever()
    finally:
        httpd.server_close()
    logging.info("quitting (this never runs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
