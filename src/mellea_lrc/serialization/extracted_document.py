"""JSON round-trip support for extracted documents."""

from __future__ import annotations

from collections.abc import Mapping

from mellea_lrc.core.case_names import CaseName
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
from mellea_lrc.core.pin_cites import PinCite
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

SCHEMA_VERSION = 10
"""What an artifact of this shape is called, so a reader refuses one it cannot read.

Version 10 is what a citation became when it started carrying its own position.

A citation entry is now ``citation_id``, ``citation``, and the three fields that
belong to its place in a *document* rather than to the citation -- ``resolves_to``,
``root_id``, ``colocation_id``. Everything else moved **inside** ``citation``,
where it is written once: ``span``, ``locator_span``, ``matched_text``,
``case_name`` as a whole name rather than a bare span, and ``pin_cite`` as the
written form with its position and the pages it claims. Version 8 wrote most of
those beside the citation, where a reading and the position it was read from
could go out of step -- which is what happened when a reader repaired a name.
"""
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
                "citation": _serialize_citation(citation.citation),
                "resolves_to": citation.resolves_to,
                "root_id": citation.root_id,
                "colocation_id": citation.colocation_id,
            }
            for citation in document.citations
        ],
        "unread_case_names": [serialize_dataclass(span) for span in document.unread_case_names],
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
        unread_case_names=tuple(
            _optional_span(item, name="unread_case_names")
            for item in require_list(payload.get("unread_case_names", []), name="unread_case_names")
            if item is not None
        ),
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


def _deserialize_citation(payload: Mapping[str, object]) -> ExtractedCitation:
    """One citation, from the one place the artifact writes it."""
    citation_payload = require_mapping(payload.get("citation"), name="citation.citation")
    kind = CitationKind(
        _required_string(citation_payload.get("citation_type"), name="citation.citation_type")
    )
    fields = {key: value for key, value in citation_payload.items() if key != "citation_type"}
    # Everything that is an object rather than a string has to be rebuilt.
    for name, rebuild in (
        ("date", _deserialize_date),
        ("reporter", _deserialize_reporter),
    ):
        if isinstance(fields.get(name), Mapping):
            fields[name] = rebuild(fields[name])
    for name in ("span", "locator_span"):
        fields[name] = _optional_span(fields.get(name), name=f"citation.{name}")
    fields["case_name"] = _read_case_name(fields.get("case_name"), name="citation.case_name")
    # `UnknownCitation` states no pin cite at all, so the key is not written for
    # it and must not be invented here.
    if "pin_cite" in fields:
        fields["pin_cite"] = _read_pin_cite(fields["pin_cite"])
    return ExtractedCitation(
        citation_id=_required_string(payload.get("citation_id"), name="citation.citation_id"),
        citation=_CITATION_TYPES[kind](**fields),
        resolves_to=_optional_string(payload.get("resolves_to"), name="citation.resolves_to"),
        root_id=_optional_string(payload.get("root_id"), name="citation.root_id"),
        colocation_id=_optional_string(payload.get("colocation_id"), name="citation.colocation_id"),
    )


def _serialize_citation(citation: CanonicalCitation) -> dict[str, object]:
    """A citation and everything it knows, including where it is written."""
    return {
        "citation_type": citation_kind(citation).value,
        **serialize_dataclass(citation),
        "case_name": _serialize_case_name(citation.case_name),
        **({"pin_cite": _serialize_pin_cite(citation.pin_cite)} if hasattr(citation, "pin_cite") else {}),
    }


def _serialize_pin_cite(pin_cite: PinCite | None) -> dict[str, object] | None:
    """The written form, where it is, and the pages it claims."""
    if pin_cite is None:
        return None
    return {
        "span": serialize_dataclass(pin_cite.span) if pin_cite.span else None,
        "text": pin_cite.text,
        "pages": [serialize_dataclass(pages) for pages in pin_cite.pages],
    }


def _read_pin_cite(value: object) -> PinCite | None:
    """Rebuild a pin cite from a payload, reading its pages from the text again."""
    if not isinstance(value, Mapping):
        return None
    return PinCite.read(
        _required_string(value.get("text"), name="citation.pin_cite.text"),
        _optional_span(value.get("span"), name="citation.pin_cite.span"),
    )


def _serialize_case_name(name: CaseName | None) -> dict[str, object] | None:
    """A case name as its four parts, or `None` where the citation states none."""
    if name is None:
        return None
    return {
        "span": serialize_dataclass(name.span),
        "text": name.text,
        "plaintiff": name.plaintiff,
        "defendant": name.defendant,
    }


def _read_case_name(value: object, *, name: str) -> CaseName | None:
    """Rebuild a case name from a payload, whichever shape wrote it."""
    if not isinstance(value, Mapping):
        return None
    span = _optional_span(value.get("span"), name=f"{name}.span")
    if span is None:
        return None
    return CaseName(
        span=span,
        text=_required_string(value.get("text"), name=f"{name}.text"),
        plaintiff=_optional_string(value.get("plaintiff"), name=f"{name}.plaintiff"),
        defendant=_optional_string(value.get("defendant"), name=f"{name}.defendant"),
    )


def _case_name(payload: Mapping[str, object]) -> CaseName | None:
    """The case name a citation payload states, from either shape it may be in.

    `case_name` is the whole value. A payload written before it has only
    `case_name_span`, and a name is rebuilt from the span with the parties left
    unread, which is what that payload recorded.
    """
    whole = _read_case_name(payload.get("case_name"), name="citation.case_name")
    if whole is not None:
        return whole
    span = _optional_span(payload.get("case_name_span"), name="citation.case_name_span")
    return None if span is None else CaseName(span=span, text="")


def _optional_span(value: object, *, name: str) -> Span | None:
    if value is None:
        return None
    span = require_mapping(value, name=name)
    return Span(
        start=_required_integer(span.get("start"), name=f"{name}.start"),
        end=_required_integer(span.get("end"), name=f"{name}.end"),
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
