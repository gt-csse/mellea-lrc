"""JSON-ready serialization for reusable mellea-lrc artifacts."""

from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING, TypeAlias, cast

from mellea_lrc.core.citations import citation_kind

if TYPE_CHECKING:
    from mellea_lrc.core.citations import CanonicalCitation
    from mellea_lrc.extraction.types import DocumentExtraction, ExtractedCitation

JsonValue: TypeAlias = str | int | float | bool | None | dict[str, "JsonValue"] | list["JsonValue"]


def _serialize_citation(citation: CanonicalCitation) -> dict[str, JsonValue]:
    payload = cast("dict[str, JsonValue]", asdict(citation))
    payload["type"] = citation_kind(citation).value
    return payload


def serialize_extracted_citation(item: ExtractedCitation) -> dict[str, JsonValue]:
    """Serialize one extracted citation into a UI-agnostic JSON-ready dict."""
    return {
        "citation_id": item.citation_id,
        "span": cast("dict[str, JsonValue]", asdict(item.span)),
        "matched_text": item.matched_text,
        "citation": _serialize_citation(item.citation),
        "resolves_to": item.resolves_to,
    }


def serialize_document_extraction(result: DocumentExtraction) -> dict[str, JsonValue]:
    """Serialize a full extraction artifact without annotation-tool assumptions."""
    return {
        "source_path": result.source_path,
        "text": result.text,
        "preprocessing": cast("dict[str, JsonValue]", asdict(result.preprocessed.metadata)),
        "citations": [serialize_extracted_citation(item) for item in result.citations],
        "counts": {
            "total": len(result.citations),
            "full": len(result.full_citations),
            "by_type": _count_by_type(result),
        },
    }


def _count_by_type(result: DocumentExtraction) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in result.citations:
        kind = citation_kind(item.citation).value
        counts[kind] = counts.get(kind, 0) + 1
    return counts
