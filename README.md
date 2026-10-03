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
