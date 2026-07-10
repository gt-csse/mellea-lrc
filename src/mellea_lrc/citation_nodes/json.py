"""JSON projections for citation-node traces."""

from __future__ import annotations

from dataclasses import asdict
from types import MappingProxyType
from typing import TYPE_CHECKING

from mellea_lrc.core.citations import citation_kind
from mellea_lrc.serialization.json import SCHEMA_VERSION

if TYPE_CHECKING:
    from mellea_lrc.citation_nodes.types import (
        CitationNode,
        CitationNodeDocument,
        CitationNodeInput,
        CitationStep,
        JSONValue,
    )
    from mellea_lrc.core.citations import CanonicalCitation


def citation_node_document_to_json(document: CitationNodeDocument) -> dict[str, object]:
    """Project a citation-node document into strict JSON-compatible data."""
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "citation_node_document",
        "text": document.text,
        "nodes": [citation_node_to_json(node) for node in document.nodes],
    }


def citation_node_to_json(node: CitationNode) -> dict[str, object]:
    """Project one citation node into strict JSON-compatible data."""
    return {
        "citation_id": node.citation_id,
        "status": node.status.value,
        "input": citation_node_input_to_json(node.input),
        "steps": [citation_step_to_json(step) for step in node.steps],
    }


def citation_node_input_to_json(node_input: CitationNodeInput) -> dict[str, object]:
    """Project citation-node input into strict JSON-compatible data."""
    return {
        "citation_id": node_input.citation_id,
        "citation_span": {
            "start": node_input.citation_span.start,
            "end": node_input.citation_span.end,
        },
        "matched_locator_text": node_input.matched_locator_text,
        "matched_citation_text": node_input.matched_citation_text,
        "citation": citation_to_json(node_input.citation),
        "resolves_to": node_input.resolves_to,
    }


def citation_step_to_json(step: CitationStep) -> dict[str, object]:
    """Project one citation step into strict JSON-compatible data."""
    return {
        "step_id": step.step_id,
        "operation": step.operation,
        "status": step.status.value,
        "depends_on": list(step.depends_on),
        "lane": step.lane,
        "summary": step.summary,
        "data": json_value_to_builtin(step.data),
        "error": step.error,
    }


def citation_to_json(citation: CanonicalCitation) -> dict[str, object]:
    """Project a canonical citation dataclass into JSON-compatible data."""
    payload = asdict(citation)
    payload["type"] = citation_kind(citation).value
    return payload


def json_value_to_builtin(value: JSONValue) -> object:
    """Convert frozen JSON-like values back into builtin JSON containers."""
    if isinstance(value, MappingProxyType):
        return {key: json_value_to_builtin(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [json_value_to_builtin(item) for item in value]
    return value
