"""MuseKG demo with optional local images and model inference."""

from __future__ import annotations

import json
import logging
import os
import re
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import quote

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from musekg.demo_system import MuseKGSystem, relation_labels

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
LOGGER = logging.getLogger("musekg.demo.server")


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)

    @field_validator("query")
    @classmethod
    def strip_query(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Enter a question")
        return value.strip()


class ChatResponse(BaseModel):
    answer: str
    context: str
    entity: str
    graph_data: dict
    answer_mode: str


def create_app(
    graph_path: str | Path | None = None,
    model_name: str | None = None,
    image_dir: str | Path | None = None,
    image_mapping_path: str | Path | None = None,
) -> FastAPI:
    graph_path = graph_path or os.getenv(
        "MUSEKG_GRAPH_PATH", str(REPOSITORY_ROOT / "output" / "fictional_knowledge_graph.gexf")
    )
    model_name = model_name or os.getenv("MUSEKG_MODEL") or None
    image_dir = image_dir or os.getenv("MUSEKG_IMAGE_DIR")
    image_mapping_path = image_mapping_path or os.getenv("MUSEKG_IMAGE_MAPPING")
    image_root = Path(image_dir).resolve() if image_dir else None
    static_root = Path(os.getenv("MUSEKG_STATIC_PATH", REPOSITORY_ROOT / "demo" / "static"))
    system = MuseKGSystem(str(graph_path), model_name)
    image_mapping: dict[str, str] = {}
    image_files: dict[str, Path] = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.startup_error = None
        app.state.image_error = False
        image_mapping.clear()
        image_files.clear()
        try:
            system.load_resources()
        except Exception as error:  # noqa: BLE001 - keep health and UI available
            app.state.startup_error = "Graph unavailable. Build it or set MUSEKG_GRAPH_PATH."
            LOGGER.error("Graph loading failed (%s)", type(error).__name__)
        if image_root and image_mapping_path:
            try:
                mapping = json.loads(Path(image_mapping_path).read_text())
                if not isinstance(mapping, dict):
                    raise TypeError("Expected an object-to-image mapping")
                for node, value in mapping.items():
                    if not isinstance(value, (str, int)):
                        continue
                    filename = str(value)
                    if not Path(filename).suffix:
                        filename += ".jpg"
                    if not re.fullmatch(
                        r"[A-Za-z0-9_-]+\.(?:jpg|jpeg|png|webp)", filename, re.IGNORECASE
                    ):
                        continue
                    path = (image_root / filename).resolve()
                    if path.parent == image_root and path.is_file():
                        image_mapping[str(node)] = filename
                        image_files[filename] = path
            except (OSError, ValueError, TypeError) as error:
                app.state.image_error = True
                LOGGER.warning("Image mapping unavailable (%s)", type(error).__name__)
        yield

    app = FastAPI(title="MuseKG demo", lifespan=lifespan)
    app.state.system = system
    app.state.startup_error = None
    app.state.image_error = False
    origins = [
        value.strip()
        for value in os.getenv(
            "MUSEKG_ALLOWED_ORIGINS", "http://127.0.0.1:8000,http://localhost:8000"
        ).split(",")
        if value.strip()
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    @app.post("/api/chat", response_model=ChatResponse)
    def chat(request: ChatRequest) -> ChatResponse:
        if app.state.startup_error or system.kg is None or not system.kg.number_of_nodes():
            raise HTTPException(503, "Graph unavailable. Build it or set MUSEKG_GRAPH_PATH.")
        result = system.generate_answer(request.query)
        subgraph = system.kg.subgraph(result["subgraph_nodes"])
        nodes = []
        for node, attributes in subgraph.nodes(data=True):
            rendered = dict(attributes)
            rendered["id"] = str(node)
            rendered["label"] = rendered.get("title", str(node))
            rendered.setdefault("type", "node")
            # Only this explicit local mapping may supply a browser image URL.
            rendered.pop("image_url", None)
            for old, new in {
                "material_desc": "material",
                "production_date": "creation_date",
                "measurements": "dimensions",
                "accession_no": "accession_number",
                "history_cat": "category",
            }.items():
                if old in rendered:
                    rendered[new] = rendered[old]
            filename = image_mapping.get(str(node)) or image_mapping.get(
                str(attributes.get("source_id", ""))
            )
            if filename:
                rendered["image_url"] = "/images/" + quote(filename)
            nodes.append(rendered)
        edges = []
        for source, target in dict.fromkeys((u, v) for u, v in subgraph.edges()):
            for index, label in enumerate(relation_labels(subgraph, source, target)):
                edges.append(
                    {
                        "id": f"edge:{len(edges)}:{index}",
                        "source": str(source),
                        "target": str(target),
                        "label": label,
                    }
                )
        return ChatResponse(
            answer=result["answer"],
            context=result["context"],
            entity=result["entity_extracted"],
            graph_data={"nodes": nodes, "edges": edges},
            answer_mode=result["answer_mode"],
        )

    @app.get("/images/{filename}")
    def local_image(filename: str):
        path = image_files.get(filename)
        if path is None or path.resolve().parent != image_root or not path.is_file():
            raise HTTPException(404, "Image not found")
        return FileResponse(path)

    @app.get("/health")
    def health() -> dict:
        graph_loaded = (
            not app.state.startup_error
            and system.kg is not None
            and system.kg.number_of_nodes() > 0
        )
        degraded = not graph_loaded or system.model_error or app.state.image_error
        return {
            "status": "degraded" if degraded else "ok",
            "ready": bool(graph_loaded),
            "graph_loaded": bool(graph_loaded),
            "llm_enabled": system.llm_pipeline is not None,
            "model_requested": bool(model_name),
            "model_error": system.model_error,
            "images_loaded": len(image_files),
            "image_error": app.state.image_error,
            "error": app.state.startup_error,
        }

    @app.get("/api/examples")
    def examples() -> list[str]:
        if app.state.startup_error or system.kg is None:
            return []
        titles = []
        for _, data in system.kg.nodes(data=True):
            title = data.get("title")
            if data.get("type") == "object" and title and title not in titles:
                titles.append(title)
            if len(titles) == 3:
                break
        return [f"Tell me about {title}" for title in titles]

    app.mount("/", StaticFiles(directory=static_root, html=True), name="static")
    return app


app = create_app()
