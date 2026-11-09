"""Local graph retrieval with optional LLM extraction and answer generation."""

from __future__ import annotations

import difflib
import logging
import re
from typing import Any

import networkx as nx

from .evaluation import _display_name, normalise_text

LOGGER = logging.getLogger("musekg.demo")


def relation_labels(graph: nx.Graph, source: str, target: str) -> list[str]:
    """Read both the research graph's `relation` and released `relationship`."""
    data = graph.get_edge_data(source, target) or {}
    edges = data.values() if graph.is_multigraph() else [data]
    return [str(edge.get("relationship", edge.get("relation", "related_to"))) for edge in edges]


class MuseKGSystem:
    def __init__(self, graph_file: str, model_name: str | None = None):
        self.graph_file = graph_file
        self.model_name = model_name
        self.kg: nx.Graph | None = None
        self.llm_pipeline: Any = None
        self.model_error = False
        self._titles: dict[str, str] = {}
        self._ids: dict[str, str] = {}

    def load_resources(self) -> None:
        self.kg = None
        self.llm_pipeline = None
        self.model_error = False
        self._titles = {}
        self._ids = {}
        self.kg = nx.read_gexf(self.graph_file)
        if not self.kg.number_of_nodes():
            raise ValueError("The graph is empty")
        for node, data in self.kg.nodes(data=True):
            title = normalise_text(data.get("title"))
            # Object and image-label nodes can have the same title. Prefer objects.
            if title and (data.get("type") == "object" or title not in self._titles):
                self._titles[title] = node
            self._ids[normalise_text(node) or str(node)] = node
            if data.get("source_id") is not None:
                self._ids[normalise_text(data["source_id"]) or str(data["source_id"])] = node
        if self.model_name:
            try:
                import torch
                from transformers import pipeline

                self.llm_pipeline = pipeline(
                    "text-generation",
                    model=self.model_name,
                    device=0 if torch.cuda.is_available() else -1,
                    max_new_tokens=256,
                    trust_remote_code=False,
                )
            except Exception as error:  # noqa: BLE001 - model failure uses graph fallback
                # Never pass provider error strings (which may include credentials) to clients.
                LOGGER.warning("Model unavailable (%s); using graph answers", type(error).__name__)
                self.model_error = True

    def _match_entity(self, candidate: str) -> str | None:
        candidate = normalise_text(candidate) or ""
        if candidate in self._titles or candidate in self._ids:
            return candidate
        if len(candidate) < 4:
            return None
        partial = [title for title in self._titles if candidate in title]
        if len(partial) == 1:
            return partial[0]
        matches = difflib.get_close_matches(candidate, self._titles, n=1, cutoff=0.7)
        return matches[0] if matches else None

    def _complete(self, prompt: str, max_new_tokens: int = 128) -> str:
        result = self.llm_pipeline(
            prompt, max_new_tokens=max_new_tokens, do_sample=False, return_full_text=False
        )
        return str(result[0]["generated_text"]).removeprefix(prompt).strip()

    def _extract_entity(self, query: str) -> str:
        lowered = normalise_text(query) or ""
        if lowered in self._titles or lowered in self._ids:
            return lowered
        contained = [
            title
            for title in self._titles
            if re.search(rf"(?<!\w){re.escape(title)}(?!\w)", lowered)
        ]
        if contained:
            return max(contained, key=len)
        # Preserve the research demo's fast paths before asking a model.
        patterns = [
            r"(?:object|artifact) (?:with title|titled|named) ['\"](.+?)['\"]",
            r"what (?:is|are)(?: the)? .+? (?:of|for) (.+?)[?.!]*$",
            r"who (?:produced|made|created) (.+?)[?.!]*$",
            r"used to (?:make|create) (.+?)[?.!]*$",
            (
                r"(?:tell me(?: something)? about|show me(?: details of)?|describe|"
                r"(?:i want to know|what can you tell me) (?:more )?about) (.+?)[?.!]*$"
            ),
        ]
        candidates = []
        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                candidates.append(re.sub(r"^the\s+", "", match.group(1), flags=re.IGNORECASE))
        candidates.append(re.sub(r"^(what|who|where|when|how|which)\s+", "", lowered))
        for candidate in candidates:
            matched = self._match_entity(candidate)
            if matched:
                return matched
        if self.llm_pipeline is not None:
            try:
                extracted = self._complete(
                    "Extract only the museum object, person, or organisation name from the "
                    "question. Do not include attribute words such as material or dimensions. "
                    f"Return one name on one line.\nQuestion: {query}\nName:",
                    max_new_tokens=64,
                )
                if extracted:
                    matched = self._match_entity(extracted.splitlines()[0].strip(" '\".,?!"))
                    if matched:
                        return matched
            except Exception as error:  # noqa: BLE001 - model failure uses graph fallback
                LOGGER.warning("Entity extraction failed (%s)", type(error).__name__)
        return lowered

    def _context(self, node: str) -> tuple[str, list[str]]:
        assert self.kg is not None
        data = self.kg.nodes[node]
        lines = [f"The object titled '{data.get('title', node)}' has these properties:"]
        for key, value in data.items():
            if key not in {"id", "label", "source_id", "title", "type"}:
                lines.append(f"- {key}: {value}")
        nodes = [node]
        for neighbor in self.kg.neighbors(node):
            nodes.append(neighbor)
            for relationship in relation_labels(self.kg, node, neighbor):
                lines.append(f"- {relationship}: {_display_name(self.kg, neighbor)}")
        return "\n".join(lines), nodes

    def _deterministic_answer(self, query: str, node: str) -> str:
        assert self.kg is not None
        lowered = query.casefold()
        data = self.kg.nodes[node]
        attributes = {
            "material": ("material_desc", "material"),
            "accession": ("accession_no", "accession_number"),
            "museum": ("museum",),
            "category": ("history_cat", "category"),
            "description": ("description",),
            "dimension": ("measurements", "dimensions"),
            "measurement": ("measurements", "dimensions"),
            "date": ("production_date", "creation_date"),
            "collection": ("collection",),
            "type": ("object_type",),
        }
        for word, keys in attributes.items():
            if word in lowered:
                for key in keys:
                    if data.get(key):
                        return str(data[key])
        if any(word in lowered for word in ("who produced", "who made", "creator", "producer")):
            for neighbor in self.kg.neighbors(node):
                if set(relation_labels(self.kg, node, neighbor)) & {
                    "has_primary_producer",
                    "has_primary_organisation",
                }:
                    return _display_name(self.kg, neighbor)
        available = [
            f"{key}: {data[key]}"
            for key in ("description", "material_desc", "material", "production_date", "museum")
            if data.get(key)
        ]
        return "; ".join(available) or "No descriptive attributes are available."

    def generate_answer(self, query: str) -> dict[str, Any]:
        if self.kg is None:
            raise RuntimeError("MuseKG resources have not been loaded")
        entity = self._extract_entity(query)
        node = self._titles.get(entity)
        if node is None:
            node = self._ids.get(entity)
        if node is None:
            return {
                "query": query,
                "entity_extracted": entity,
                "context": "No matching object was found.",
                "answer": "I could not find a matching object in this collection.",
                "subgraph_nodes": [],
                "answer_mode": "graph",
            }
        context, nodes = self._context(node)
        answer = self._deterministic_answer(query, node)
        mode = "graph"
        if self.llm_pipeline is not None:
            general = any(p in query.casefold() for p in ("tell me about", "describe", "show me"))
            instruction = "Summarise this object" if general else "Answer the question concisely"
            prompt = (
                f"{instruction} using only the supplied context. If the context does not contain "
                f"the answer, say so.\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"
            )
            try:
                completion = self._complete(prompt)
                if completion:
                    answer = completion if general else completion.splitlines()[0]
                    mode = "llm"
            except Exception as error:  # noqa: BLE001 - model failure uses graph fallback
                LOGGER.warning("Answer generation failed (%s); using graph", type(error).__name__)
        return {
            "query": query,
            "entity_extracted": entity,
            "context": context,
            "answer": answer,
            "subgraph_nodes": nodes,
            "answer_mode": mode,
        }
