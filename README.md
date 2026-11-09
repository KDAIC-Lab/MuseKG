# MuseKG

MuseKG is an interactive knowledge-graph system for constructing and querying museum
collections. It combines structured museum records, optional OCR and extracted entities,
deterministic graph evaluation, research prompt templates, and a local web demo.

This repository accompanies **“MuseKG: An Interactive Knowledge Graph Over Museum
Collections”**, published at SIGIR 2026 ([DOI: 10.1145/3805712.3808366](https://doi.org/10.1145/3805712.3808366)).

## Privacy and release scope

The museum collection used in the paper is private and is **not included**. The repository
contains a wholly fictional three-object collection so that every public component can be
run end to end. Users can supply their own records using the documented input contract.

## Included

- Streaming graph construction from museum JSON records
- Optional OCR and extracted-entity enrichment
- Exact-match evaluation and an optional local LLM judge for structured QA pairs
- Prompt templates used by the research pipeline
- A FastAPI and NetworkX demo with an optional local Hugging Face model
- A fictional dataset and tests

## Installation

MuseKG requires Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Install the lightweight web-demo dependencies with:

```bash
python -m pip install -e ".[demo]"
```

Only if you want local Hugging Face generation or LLM-based evaluation, install the larger
model dependencies:

```bash
python -m pip install -e ".[model]"
```

## Build the fictional graph

```bash
musekg-build \
  --objects examples/fictional_museum/objects.json \
  --entities examples/fictional_museum/entities.json \
  --ocr examples/fictional_museum/ocr.json \
  --output output/fictional_knowledge_graph.gexf
```

The inputs stay on the local machine. To use a different collection, follow
[the data-format guide](docs/data-format.md) and point these arguments to local files.

## Evaluate

```bash
musekg-evaluate \
  --graph output/fictional_knowledge_graph.gexf \
  --qa examples/fictional_museum/qa_pairs.json \
  --output output/fictional_evaluation.json
```

The fictional benchmark should report 3/3 correct answers. This validates the public
pipeline; it does not reproduce the paper's private-data results.

Exact matching is the default and requires no model. To evaluate semantic equivalence with the
released LLM-judge prompt, install the `model` extra and select a compatible local Hugging Face
model or public model identifier:

```bash
musekg-evaluate \
  --graph output/fictional_knowledge_graph.gexf \
  --qa examples/fictional_museum/qa_pairs.json \
  --judge llm \
  --judge-model MODEL_ID_OR_LOCAL_PATH \
  --output output/fictional_evaluation_llm.json
```

The default judge prompt is the packaged copy of `prompts/eval_prompt.txt`. Use
`--judge-prompt` to test a replacement prompt. Model-generated decisions must contain a JSON
`correctness` value of `correct` or `incorrect`; malformed decisions stop the evaluation instead
of being silently counted. A public model identifier may cause Transformers to download model
files. Use a local path for fully offline evaluation. Remote model code is disabled.

## Run the local demo

The KDAIC-Lab fork maintains the integrated research demo in [`demo/`](demo/README.md):
chat, graph exploration, object details, optional local images and optional LLM answers.
See the [demo guide](demo/README.md) for configuration and troubleshooting.

```bash
MUSEKG_GRAPH_PATH=output/fictional_knowledge_graph.gexf \
  uvicorn demo.server:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000>. By default the demo uses deterministic answers and needs no
GPU. To enable an optional local model, install the `model` extra and set `MUSEKG_MODEL`
to a compatible Hugging Face model identifier. MuseKG does not enable remote model code.
The graph visualisation uses the included MIT-licensed Cytoscape.js 3.23.0 bundle,
so the interface does not need a CDN connection.

Do not expose a private graph through a public demo: responses, context, and visualised
neighbourhoods can reveal the underlying records.

## Development

```bash
pytest
ruff check .
ruff format --check .
```

## Citation

Please cite the paper when using MuseKG in academic work:

```bibtex
@inproceedings{li2026musekg,
  title={MuseKG: An Interactive Knowledge Graph Over Museum Collections},
  author={Li, Jinhao and Qi, Jianzhong and Han, Soyeon Caren and Holden, Eun-Jung},
  booktitle={Proceedings of the 49th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  pages={5176--5181},
  year={2026}
}
```

GitHub can also generate a citation from [`CITATION.cff`](CITATION.cff).

## License

The MuseKG source code and fictional example collection are licensed under the
[Apache License 2.0](LICENSE). The accompanying paper is published separately under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The private museum collection
used in the paper is not distributed or licensed by this repository.
