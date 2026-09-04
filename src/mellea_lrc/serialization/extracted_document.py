"""JSON round-trip support for extracted documents."""

from __future__ import annotations

from collections.abc import Mapping

from mellea_lrc.core.citations import (
    CanonicalCitation,
    CitationDate,
    CitationKind,
    DocketCitation,
    FullCaseCitation,
    FullJournalCitation,
    FullLawCitation,
    IdCitation,
    ReferenceCitation,
    Reporter,
    ShortCaseCitation,
    SupraCitation,
    UnknownCitation,
    citation_kind,
)
from mellea_lrc.core.documents import SourceFormat, SourceMetadata
from mellea_lrc.core.spans import Span
from mellea_lrc.extraction.reading.relaxation import Relaxation
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

SCHEMA_VERSION = 3
_ARTIFACT_TYPE = "extracted_document"

_CITATION_TYPES: dict[CitationKind, type[CanonicalCitation]] = {
    CitationKind.FULL_CASE: FullCaseCitation,
    CitationKind.FULL_LAW: FullLawCitation,
    CitationKind.FULL_JOURNAL: FullJournalCitation,
    CitationKind.DOCKET: DocketCitation,
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
                "full_span": serialize_dataclass(citation.full_span),
                "locator_span": serialize_dataclass(citation.locator_span),
                "matched_text": citation.matched_text,
                "citation": {
                    "citation_type": citation_kind(citation.citation).value,
                    **serialize_dataclass(citation.citation),
                },
                "resolves_to": citation.resolves_to,
                "authority_id": citation.authority_id,
                "colocation_id": citation.colocation_id,
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
            relaxation=Relaxation(
                _required_string(extraction_metadata.get("relaxation"), name="extraction_metadata.relaxation")
            ),
        ),
    )


def _deserialize_citation(value: object) -> ExtractedCitation:
    payload = require_mapping(value, name="citation")
    full_span = require_mapping(payload.get("full_span"), name="citation.full_span")
    locator_span = require_mapping(payload.get("locator_span"), name="citation.locator_span")
    citation_payload = require_mapping(payload.get("citation"), name="citation.citation")
    kind = CitationKind(
        _required_string(citation_payload.get("citation_type"), name="citation.citation_type")
    )
    citation_type = _CITATION_TYPES[kind]
    citation_fields = {key: value for key, value in citation_payload.items() if key != "citation_type"}
    # Two fields hold objects rather than strings, so both have to be rebuilt.
    if isinstance(citation_fields.get("date"), Mapping):
        citation_fields["date"] = _deserialize_date(citation_fields["date"])
    if isinstance(citation_fields.get("reporter"), Mapping):
        citation_fields["reporter"] = _deserialize_reporter(citation_fields["reporter"])
    return ExtractedCitation(
        citation_id=_required_string(payload.get("citation_id"), name="citation.citation_id"),
        full_span=Span(
            start=_required_integer(full_span.get("start"), name="citation.full_span.start"),
            end=_required_integer(full_span.get("end"), name="citation.full_span.end"),
        ),
        locator_span=Span(
            start=_required_integer(locator_span.get("start"), name="citation.locator_span.start"),
            end=_required_integer(locator_span.get("end"), name="citation.locator_span.end"),
        ),
        matched_text=_required_string(payload.get("matched_text"), name="citation.matched_text"),
        citation=citation_type(**citation_fields),
        resolves_to=_optional_string(payload.get("resolves_to"), name="citation.resolves_to"),
        authority_id=_optional_string(payload.get("authority_id"), name="citation.authority_id"),
        colocation_id=_optional_string(payload.get("colocation_id"), name="citation.colocation_id"),
    )


def _deserialize_date(payload: Mapping[str, object]) -> CitationDate:
    return CitationDate(
        year=_required_string(payload.get("year"), name="citation.date.year"),
        month=_optional_string(payload.get("month"), name="citation.date.month"),
        day=_optional_string(payload.get("day"), name="citation.date.day"),
    )


def _deserialize_reporter(payload: Mapping[str, object]) -> Reporter:
    is_scotus = payload.get("is_scotus")
    return Reporter(
        as_written=_required_string(payload.get("as_written"), name="citation.reporter.as_written"),
        short_name=_optional_string(payload.get("short_name"), name="citation.reporter.short_name"),
        name=_optional_string(payload.get("name"), name="citation.reporter.name"),
        cite_type=_optional_string(payload.get("cite_type"), name="citation.reporter.cite_type"),
        is_scotus=bool(is_scotus),
        editions=tuple(
            _required_string(value, name="citation.reporter.editions")
            for value in require_list(payload.get("editions", []), name="citation.reporter.editions")
        ),
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
