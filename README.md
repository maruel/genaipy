# genaipy

Local backend python implementations for [github.com/maruel/genai](https://github.com/maruel/genai).


## Manual testing the python servers

These scripts are embedded in the main executable and are run when model
`"python"` is requested.

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

