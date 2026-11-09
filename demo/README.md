# MuseKG interactive demo

This is the maintained demo for the KDAIC-Lab fork. It combines the research
interface (chat, graph exploration, object details and optional images) with the
packaged MuseKG implementation. There is one entry point: `demo.server:app`.
The paper's original repository remains https://github.com/JinhaoLee/MuseKG.

## Run with fictional data

From the repository root, using Python 3.10 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[demo]'
musekg-build \
  --objects examples/fictional_museum/objects.json \
  --entities examples/fictional_museum/entities.json \
  --ocr examples/fictional_museum/ocr.json \
  --output output/fictional_knowledge_graph.gexf
uvicorn demo.server:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000 and select a suggested object. Suggestions come from
the loaded graph. Ask `What is the material of the Demonstration Brass Telescope?`
or `Who produced the Demonstration Brass Telescope?`, then click a graph node to
inspect its attributes. Without a model, answers come directly from graph fields.
This is a fictional smoke example, not a reproduction of the paper's benchmark.
The graph UI uses the included Cytoscape.js 3.23.0 bundle and needs no CDN.

## Optional local LLM

```bash
python -m pip install -e '.[demo,model]'
export MUSEKG_MODEL=/path/to/your/local/model
uvicorn demo.server:app --host 127.0.0.1 --port 8000
```

A compatible Hugging Face identifier such as `google/gemma-2-9b-it` may also be
used. This can download model weights and requires access to that model. Supply
any required Hugging Face credentials through your local environment or credential
store; never put credentials in tracked files. No token is embedded in this repo.
GPU use is automatic when CUDA is available; large models may be impractical on CPU.
Remote model code is disabled.

Entity resolution tries exact, partial and fuzzy names, plus research-demo question
patterns. If these fail, the optional model extracts an entity name and the system
checks it against the graph. Answers use one-hop graph context. Model failures or
empty completions fall back to actual graph attributes; responses identify whether
the answer came from the model or the graph. This is not a general multi-hop planner.

## Your own graph and optional images

Set `MUSEKG_GRAPH_PATH` to an existing local GEXF file. Both `relationship` and
legacy `relation` edge attributes are supported, including parallel edges.

Images are optional and remain outside the repository:

```bash
export MUSEKG_GRAPH_PATH=/path/to/collection.gexf
export MUSEKG_IMAGE_DIR=/path/to/collection-images
export MUSEKG_IMAGE_MAPPING=/path/to/image-mapping.json
```

The mapping is a JSON object from graph node ID (or its `source_id`) to an image
filename. An image ID without an extension is interpreted as a `.jpg` filename:

```json
{"object:SYN-001": "fictional-telescope.png", "SYN-002": "fictional-case"}
```

Only mapped JPG, JPEG, PNG and WebP files inside the configured directory are
served. The mapping file itself is not exposed. Missing images do not stop chat.
No museum photographs, collection records, private graphs or model weights are
included. A demo connected to a private collection returns its records and images:
keep it on localhost or behind your institution's access controls.

## Configuration and troubleshooting

Optional variables are listed in `.env.example`. The app does not automatically
load `.env`; export variables in your shell before starting the server. Relative
paths are resolved from the current working directory, so run from the repo root.
`MUSEKG_ALLOWED_ORIGINS` is a comma-separated list of browser origins;
`MUSEKG_STATIC_PATH` can select another static UI directory.

`GET /health` reports graph readiness, model status and image status. A missing
or empty graph leaves the UI available but `/api/chat` returns HTTP 503. Check
that you ran `musekg-build` or selected a valid graph. Model loading failures leave
graph answers available and report `model_error: true` and `status: degraded`.
Error responses do not contain local paths or provider exception messages.

## Development and provenance

```bash
python -m pip install -e '.[dev,demo]'
python -m pytest
ruff check .
ruff format --check .
```

The interface and optional image workflow were adapted from the MuseKG research
demo. Query patterns and LLM extraction were integrated into
`src/musekg/demo_system.py`; the old separate `models.py` and machine-specific
startup script are not needed. Missing-graph handling, object-title preference
and empty-model-output fallback incorporate the later MuseKG demo fixes.
Core graph construction and evaluation are unchanged by this demo integration.
The source retains the repository's Apache-2.0 licence. The bundled Cytoscape.js
is MIT-licensed; its copyright and licence text are preserved in
`static/vendor/cytoscape.min.js`.
