"""JSON-ready serialization for one validated document."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from enum import Enum
from typing import TYPE_CHECKING, TypeAlias

from mellea_lrc.core.citations import citation_kind

if TYPE_CHECKING:
    from mellea_lrc.validation.types import ValidatedDocument

JsonValue: TypeAlias = str | int | float | bool | list["JsonValue"] | dict[str, "JsonValue"] | None
SCHEMA_VERSION = 1


def serialize_validated_document(document: ValidatedDocument) -> dict[str, JsonValue]:
    """Project one ``ValidatedDocument`` into a visualization-ready JSON object.

    The extracted source is serialized once. Each validation progression then
    refers to its source citation by ``citation_id`` and carries a flat ordered
    list of nodes. ``depends_on`` is preserved verbatim for graph edges.
    """
    source = document.source
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "validated_document",
        "source": {
            "source_metadata": _serialize_value(source.source_metadata),
            "text": source.text,
            "preprocessing_metadata": _serialize_value(source.preprocessing_metadata),
            "citations": [
                {
                    "citation_id": citation.citation_id,
                    "span": _serialize_value(citation.span),
                    "matched_text": citation.matched_text,
                    "citation": {
                        "citation_type": citation_kind(citation.citation).value,
                        **_serialize_dataclass(citation.citation),
                    },
                    "resolves_to": citation.resolves_to,
                }
                for citation in source.citations
            ],
            "extraction_metadata": _serialize_value(source.extraction_metadata),
        },
        "citations": [
            {
                "citation_id": progression.citation_id,
                "nodes": [
                    {
                        "node_type": type(node).__name__,
                        **_serialize_dataclass(node),
                    }
                    for node in progression.nodes
                ],
            }
            for progression in document.citations
        ],
    }


def _serialize_dataclass(value: object) -> dict[str, JsonValue]:
    if not is_dataclass(value) or isinstance(value, type):
        msg = f"Expected a dataclass instance, got {type(value).__name__}"
        raise TypeError(msg)
    return {field.name: _serialize_value(getattr(value, field.name)) for field in fields(value)}


def _serialize_value(value: object) -> JsonValue:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Enum):
        return _serialize_value(value.value)
    if is_dataclass(value) and not isinstance(value, type):
        return _serialize_dataclass(value)
    if isinstance(value, Mapping):
        return {str(key): _serialize_value(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_serialize_value(item) for item in value]
    msg = f"Cannot serialize {type(value).__name__} as JSON"
    raise TypeError(msg)
