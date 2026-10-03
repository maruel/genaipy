#!/usr/bin/env python3
# Copyright 2026 Marc-Antoine Ruel. All rights reserved.
# Use of this source code is governed under the Apache License, Version 2.0
# that can be found in the LICENSE file.

# Managed Whistle worker: serialized native ASR behind a loopback HTTP endpoint.
import argparse
import array
import contextlib
import http.server
import json
import os
import sys
import threading

MAX_PCM_BYTES = 16000 * 2 * 30


class Server(http.server.ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, model):
        super().__init__(("127.0.0.1", 0), Handler)
        self.model = model
        # Needle's process-global native engine is not thread safe.
        self.model_lock = threading.Lock()


class Handler(http.server.BaseHTTPRequestHandler):
    server: Server

    def do_POST(self) -> None:
        if self.path != "/transcribe":
            self.send_error(404)
            return
        try:
            n = int(self.headers.get("Content-Length", "0"))
            if n <= 0 or n > MAX_PCM_BYTES or n % 2:
                raise ValueError("expected aligned 16 kHz mono S16LE PCM, at most 30 seconds")
            pcm = self.rfile.read(n)
            if len(pcm) != n:
                raise ValueError("truncated PCM")
            audio = array.array("h", pcm)
            if sys.byteorder != "little":
                audio.byteswap()
            samples = array.array("f", (v / 32768.0 for v in audio))
            with self.server.model_lock, contextlib.redirect_stdout(sys.stderr):
                text = self.server.model.transcribe(samples)["text"]
            self.reply(200, {"text": text})
        except (ValueError, RuntimeError) as e:
            self.reply(400, {"error": str(e)})

    def reply(self, status: int, result: dict[str, str]) -> None:
        data = json.dumps(result).encode()
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            # Cancellation only disconnects this request; the model stays loaded.
            return

    def log_message(self, fmt: str, *args: object) -> None:
        print(fmt % args, file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="")
    args = parser.parse_args()
    os.environ["NEEDLE_TELEMETRY"] = "0"
    os.environ["DO_NOT_TRACK"] = "1"

    # EOF stops the Python child even if its uv launcher or Go parent dies.
    def lifeline() -> None:
        sys.stdin.buffer.read()
        os._exit(0)

    threading.Thread(target=lifeline, daemon=True).start()
    # Import only at worker startup so offline tests require no native package.
    with contextlib.redirect_stdout(sys.stderr):
        import needle

        model = needle.Whistle(weights=args.model or None)
    server = Server(model)

    print(json.dumps({"url": f"http://127.0.0.1:{server.server_port}"}), flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
