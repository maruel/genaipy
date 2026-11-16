# genaipy

Local backend python implementations for [github.com/maruel/genai](https://github.com/maruel/genai).


## Manual testing the python servers

`NewServer` embeds the runtime scripts, prepares a virtualenv in an absolute
cache directory, and starts `llm.py` or `image_gen.py`. It may install packages
and download models. Pass a context for cancellation and call `Close` when done.

These scripts can be run stand alone, e.g. to run the Stable Diffusion
image generator on a separate machine, or to customize the image generation.


### Image Generation

#### macOS or linux

```
./setup.sh
source venv/bin/activate
./image_gen.py --host 0.0.0.0 --port 8032
```

#### Windows

```
setup.bat
venv\Scripts\activate
python image_gen.py --host 0.0.0.0 --port 8032
```


### LLM

#### macOS or linux

```
./setup.sh
source venv/bin/activate
./llm.py --host 0.0.0.0 --port 8031
```

#### Windows

```
setup.bat
venv\Scripts\activate
python llm.py --host 0.0.0.0 --port 8031
```


## Speech

Recognizers in `providers/` are `genai.Provider`s whose `ConnectRealtime`, with
`genai.RealtimeOptionTranscription`, transcribes one utterance of signed 16-bit
little-endian mono PCM. `genai.RealtimeOptionAudio.InputLanguage` carries the
spoken language as a BCP 47 tag; a language the model does not support fails
the connection with `base.ErrNotSupported`. Send audio, then `Commit`; the
session yields the final transcript and `TurnDone`.

Each engine pairs a client package with a server package that runs the
engine locally; pass the server's `URL` to the client's `New`. The client
reaches a remote server the same way.

`providers/nemospeech` transcribes through an NVIDIA
[NeMo-Speech.cpp](https://github.com/NVIDIA/NeMo-Speech.cpp) server running
Parakeet or Nemotron ASR. Nemotron and Parakeet CTC stream over the server's
realtime WebSocket, yielding interim and segment transcripts as the server
decodes. Parakeet TDT cannot stream: its session buffers the audio and
transcribes it at `Commit`. `nemospeechsrv.Download` fetches a pinned
native release, verified against its published SHA-256, and `nemospeechsrv.New`
runs it on a loopback port; pass its `URL` to `nemospeech.New`. The server
downloads the model into NeMo-Speech.cpp's own model cache on first use. No
Python or `uv` is required.

`providers/kyutai` streams 24 kHz audio to Kyutai STT 1B (English and French)
or 2.6B (English) over moshi-server's `/api/asr-streaming` WebSocket and yields
each word as a draft. `kyutaisrv.New` runs the models through `uv` with
`moshi`, or `moshi-mlx` on Apple silicon. The client also reaches a stock
`moshi-server`, sending the API key given to `kyutai.New`.

`providers/whistle` transcribes 16 kHz audio through a Cactus Whistle worker
at `Commit`, without drafts. Long clips use consecutive 30-second windows, with
possible loss of accuracy on words crossing window boundaries.
`whistlesrv.New(ctx, model)` runs the worker with the pinned `uv`, installing
`cactus-needle==3.1.0`, which fetches a **prebuilt native engine** and the
default Whistle weights on first use; `model` is an optional local `.cact`
weights path. Telemetry is disabled.

`providers/vosk` streams 16 kHz or faster audio to a [Vosk](https://alphacephei.com/vosk/)
server over the [vosk-server](https://github.com/alphacep/vosk-server)
WebSocket protocol, yielding interim drafts and a phrase transcript at each
detected pause. Each `vosk.Models` entry recognizes one language;
`vosk.English` is the 1.8 GB US English server model, using about 7 GB of
memory.
`vosksrv.New(ctx, model)` downloads the model archive, verified against a
pinned SHA-256, into the user's `genaipy/vosk` cache directory and runs the
worker with the pinned `uv`, Python 3.12 and `vosk` package. The client also
reaches a remote vosk-server, such as the `alphacep/kaldi-en` container.

`providers/kittentts` streams speech from a KittenTTS worker: `Synthesize`
yields owned mono S16LE PCM chunks at `kittentts.SampleRate`, 24 kHz, in the
voice given to `kittentts.New`, Jasper by default. `kittenttssrv.New(ctx,
kittenttssrv.Config{})` runs the worker with the pinned `uv` and Python 3.12,
pinning the `kittenml` package and using its streaming API on CPU. The default
model is `KittenML/kitten-tts-2` (1.7B); `kittentts.Models` also lists the
`kitten-tts-2-mini` (82M), `-nano` (17M per language) and `-pico` (7M per
language) ONNX models. Each model loads at a pinned Hugging Face revision,
cached under the user's `genaipy/kittentts` cache directory. The worker reports inference failures as HTTP errors or
truncated streams.

The Python workers download a pinned `uv` release, verified against its
published SHA-256, into the user's `genaipy/uv` cache directory on first use; a
`uv` on `PATH` is not required.

Each server's constructor context owns the worker's lifetime; cancelling a
request leaves the worker available. The worker exits when its stdin lifeline
closes. Startup installs the pinned package and downloads uncached models; no
model installation occurs in default tests.

## Development

Run `make fix`, `make verify`, `make build`, `go vet ./...`, `make test`, and
`make test-race`. `make tools` installs pinned Go formatting/lint tools and Ruff
(requires uv); ShellCheck must be installed separately. `make git-hooks` enables
the repository's pre-commit static gate.

Default tests run offline, using a fake interpreter process and stubbed model
libraries. `make smoke` explicitly opts into installing Python dependencies and
downloading the small ungated SmolLM2 model. Set
`GENAIPY_MODEL_CACHE` to an absolute directory to reuse its virtualenv.
A working model requires sufficient memory and appropriate accelerator packages.

`llm.py --model MODEL_ID` selects a Hugging Face text model. The default Llama
model requires Hugging Face authentication; an ungated model such as
`HuggingFaceTB/SmolLM2-135M-Instruct` is suitable for a small CPU smoke test.

Offline tests cover the worker without installing Needle. To validate a real
speech clip, explicitly opt in with
`GENAIPY_WHISTLE_SMOKE_PCM=/path/to/16k-mono-s16le.pcm go test -run TestWhistleSmoke -v ./providers/whistle/whistlesrv`.
