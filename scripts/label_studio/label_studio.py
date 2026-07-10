"""Label Studio prediction serialization for pre-annotation."""

from dataclasses import asdict
from typing import Any

from mellea_lrc.core.citations import (
    FullCaseCitation,
    FullJournalCitation,
    FullLawCitation,
    IdCitation,
    ReferenceCitation,
    ShortCaseCitation,
    SupraCitation,
    UnknownCitation,
)
from mellea_lrc.extraction.types import ExtractedCitation, ExtractedDocument

MODEL_VERSION = "eyecite-pre-annotation"


def _field_values(item: ExtractedCitation) -> dict[str, str | None]:
    citation = item.citation
    if isinstance(citation, FullCaseCitation):
        return asdict(citation)
    if isinstance(citation, FullLawCitation):
        return asdict(citation)
    if isinstance(citation, FullJournalCitation):
        return asdict(citation)
    if isinstance(citation, ShortCaseCitation):
        return asdict(citation)
    if isinstance(citation, SupraCitation):
        return asdict(citation)
    if isinstance(citation, IdCitation):
        return asdict(citation)
    if isinstance(citation, ReferenceCitation):
        return asdict(citation)
    if isinstance(citation, UnknownCitation):
        return {}
    msg = f"Unsupported citation type: {type(citation).__name__}"
    raise TypeError(msg)


def _label_result(extraction: ExtractedDocument, item: ExtractedCitation) -> dict[str, Any]:
    annotated_text = extraction.text[item.citation_span.start : item.citation_span.end]
    return {
        "id": item.citation_id,
        "from_name": "label",
        "to_name": "text",
        "type": "labels",
        "value": {
            "start": item.citation_span.start,
            "end": item.citation_span.end,
            "text": annotated_text,
            "labels": [item.citation.kind.value],
        },
    }


def _field_results(item: ExtractedCitation) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for field_name, value in _field_values(item).items():
        results.append(
            {
                "id": item.citation_id,
                "from_name": field_name,
                "to_name": "text",
                "type": "textarea",
                "value": {"text": [value if value is not None else ""]},
            }
        )
    return results


def _relation_results(extraction: ExtractedDocument) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for item in extraction.citations:
        if item.resolves_to is None:
            continue
        results.append(
            {
                "from_id": item.citation_id,
                "to_id": item.resolves_to,
                "type": "relation",
                "direction": "right",
            }
        )
    return results


def to_label_studio_prediction(extraction: ExtractedDocument) -> dict[str, Any]:
    """Convert a document extraction into a Label Studio prediction dict."""
    results: list[dict[str, Any]] = []
    for item in extraction.citations:
        results.append(_label_result(extraction, item))
        results.extend(_field_results(item))
    results.extend(_relation_results(extraction))

    return {
        "model_version": MODEL_VERSION,
        "score": 1.0,
        "result": results,
    }
