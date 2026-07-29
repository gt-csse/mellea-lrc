"""JSON round-trip support for extracted documents."""

from __future__ import annotations

from collections.abc import Mapping

from mellea_lrc.core.citations import (
    CanonicalCitation,
    CitationKind,
    FullCaseCitation,
    FullJournalCitation,
    FullLawCitation,
    IdCitation,
    ReferenceCitation,
    ShortCaseCitation,
    SupraCitation,
    UnknownCitation,
    citation_kind,
)
from mellea_lrc.core.documents import SourceFormat, SourceMetadata
from mellea_lrc.core.spans import Span
from mellea_lrc.extraction.types import (
    ExtractedCitation,
    ExtractedDocument,
    ExtractionBackend,
    ExtractionMetadata,
)
from mellea_lrc.preprocessing.types import (
    PreprocessingBackend,
    PreprocessingMetadata,
)
from mellea_lrc.serialization._json import JsonValue, require_list, require_mapping, serialize_dataclass

SCHEMA_VERSION = 2
_ARTIFACT_TYPE = "extracted_document"

_CITATION_TYPES: dict[CitationKind, type[CanonicalCitation]] = {
    CitationKind.FULL_CASE: FullCaseCitation,
    CitationKind.FULL_LAW: FullLawCitation,
    CitationKind.FULL_JOURNAL: FullJournalCitation,
    CitationKind.SHORT_CASE: ShortCaseCitation,
    CitationKind.SUPRA: SupraCitation,
    CitationKind.ID: IdCitation,
    CitationKind.REFERENCE: ReferenceCitation,
    CitationKind.UNKNOWN: UnknownCitation,
}


def serialize_extracted_document(document: ExtractedDocument) -> dict[str, JsonValue]:
    """Project one ``ExtractedDocument`` into a recoverable JSON artifact."""
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": _ARTIFACT_TYPE,
        "source_metadata": serialize_dataclass(document.source_metadata),
        "text": document.text,
        "preprocessing_metadata": serialize_dataclass(document.preprocessing_metadata),
        "citations": [
            {
                "citation_id": citation.citation_id,
                "span": serialize_dataclass(citation.span),
                "locator_span": serialize_dataclass(citation.locator_span),
                "matched_text": citation.matched_text,
                "citation": {
                    "citation_type": citation_kind(citation.citation).value,
                    **serialize_dataclass(citation.citation),
                },
                "resolves_to": citation.resolves_to,
            }
            for citation in document.citations
        ],
        "extraction_metadata": serialize_dataclass(document.extraction_metadata),
    }


def deserialize_extracted_document(payload: Mapping[str, object]) -> ExtractedDocument:
    """Recover one ``ExtractedDocument`` from its serialized artifact."""
    _require_artifact(payload, artifact_type=_ARTIFACT_TYPE)
    source_metadata = require_mapping(payload.get("source_metadata"), name="source_metadata")
    preprocessing_metadata = require_mapping(
        payload.get("preprocessing_metadata"), name="preprocessing_metadata"
    )
    extraction_metadata = require_mapping(payload.get("extraction_metadata"), name="extraction_metadata")
    citations = require_list(payload.get("citations"), name="citations")
    text = payload.get("text")
    if not isinstance(text, str):
        msg = "text must be a string"
        raise ValueError(msg)

    return ExtractedDocument(
        source_metadata=SourceMetadata(
            path=_optional_string(source_metadata.get("path"), name="source_metadata.path"),
            format=SourceFormat(
                _required_string(source_metadata.get("format"), name="source_metadata.format")
            ),
            header=_optional_string(source_metadata.get("header"), name="source_metadata.header"),
        ),
        text=text,
        preprocessing_metadata=PreprocessingMetadata(
            backend=PreprocessingBackend(
                _required_string(preprocessing_metadata.get("backend"), name="preprocessing_metadata.backend")
            ),
            backend_version=_optional_string(
                preprocessing_metadata.get("backend_version"), name="preprocessing_metadata.backend_version"
            ),
        ),
        citations=tuple(_deserialize_citation(item) for item in citations),
        extraction_metadata=ExtractionMetadata(
            backend=ExtractionBackend(
                _required_string(extraction_metadata.get("backend"), name="extraction_metadata.backend")
            ),
            backend_version=_optional_string(
                extraction_metadata.get("backend_version"), name="extraction_metadata.backend_version"
            ),
        ),
    )


def _deserialize_citation(value: object) -> ExtractedCitation:
    payload = require_mapping(value, name="citation")
    span = require_mapping(payload.get("span"), name="citation.span")
    locator_span = require_mapping(payload.get("locator_span"), name="citation.locator_span")
    citation_payload = require_mapping(payload.get("citation"), name="citation.citation")
    kind = CitationKind(
        _required_string(citation_payload.get("citation_type"), name="citation.citation_type")
    )
    citation_type = _CITATION_TYPES[kind]
    citation_fields = {key: value for key, value in citation_payload.items() if key != "citation_type"}
    return ExtractedCitation(
        citation_id=_required_string(payload.get("citation_id"), name="citation.citation_id"),
        span=Span(
            start=_required_integer(span.get("start"), name="citation.span.start"),
            end=_required_integer(span.get("end"), name="citation.span.end"),
        ),
        locator_span=Span(
            start=_required_integer(locator_span.get("start"), name="citation.locator_span.start"),
            end=_required_integer(locator_span.get("end"), name="citation.locator_span.end"),
        ),
        matched_text=_required_string(payload.get("matched_text"), name="citation.matched_text"),
        citation=citation_type(**citation_fields),
        resolves_to=_optional_string(payload.get("resolves_to"), name="citation.resolves_to"),
    )


def _require_artifact(payload: Mapping[str, object], *, artifact_type: str) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION:
        msg = f"Unsupported serialization schema version: {payload.get('schema_version')!r}"
        raise ValueError(msg)
    if payload.get("artifact_type") != artifact_type:
        msg = f"Expected artifact_type={artifact_type!r}"
        raise ValueError(msg)


def _required_string(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        msg = f"{name} must be a string"
        raise ValueError(msg)
    return value


def _optional_string(value: object, *, name: str) -> str | None:
    if value is None:
        return None
    return _required_string(value, name=name)


def _required_integer(value: object, *, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        msg = f"{name} must be an integer"
        raise ValueError(msg)
    return value
