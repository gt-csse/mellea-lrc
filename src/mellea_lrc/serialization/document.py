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
from mellea_lrc.core.findings import Finding, FindingKind
from mellea_lrc.core.pin_cites import PinCite, PinCiteKind, PinCitePages
from mellea_lrc.core.record import CitationRecord, Correction, Node, Reads
from mellea_lrc.core.spans import Span
from mellea_lrc.extraction.reading.relaxation import Relaxation
from mellea_lrc.extraction.types import (
    CitationRecord,
    Document,
    ExtractionBackend,
    ExtractionMetadata,
)
from mellea_lrc.preprocessing.types import (
    PreprocessingBackend,
    PreprocessingMetadata,
)
from mellea_lrc.serialization._json import JsonValue, require_list, require_mapping, serialize_dataclass

SCHEMA_VERSION = 13
"""What an artifact of this shape is called, so a reader refuses one it cannot read.

Version 11 is the citation record. A document holds records rather than
extracted citations, and each entry is::

    citation_id            what this pass assigned, which never changes
    source                 the citation as the rules read it, frozen
    stated                 the same citation as currently read
    resolves_to, root_id, colocation_id
    authority_id, found    filled by validation, absent until then
    trace                  every node, and the corrections it justified

`source` and `stated` are the same shape, so a reader diffs them field by field
to see what was changed and why. Both carry their own position -- ``span``,
``locator_span``, ``matched_text``, ``case_name`` and ``pin_cite`` are fields of
the citation, written once, because a reading and the position it was read from
go out of step the moment they are stored apart.
"""
_ARTIFACT_TYPE = "document"

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


def serialize_document(document: Document) -> dict[str, JsonValue]:
    """Project one ``Document`` into a recoverable JSON artifact."""
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": _ARTIFACT_TYPE,
        "source_metadata": serialize_dataclass(document.source_metadata),
        "text": document.text,
        "preprocessing_metadata": serialize_dataclass(document.preprocessing_metadata),
        "citations": [_serialize_record(record) for record in document.citations],
        "unread_case_names": [serialize_dataclass(span) for span in document.unread_case_names],
        "findings": [_serialize_finding(finding) for finding in document.findings],
        "passes": list(document.passes),
        "extraction_metadata": serialize_dataclass(document.extraction_metadata),
    }


def _serialize_finding(finding: Finding) -> dict[str, object]:
    """One thing a pass learned that belongs to no citation."""
    return {
        "kind": finding.kind.value,
        "stage": finding.stage,
        "made_by": finding.made_by,
        "message": finding.message,
        "span": serialize_dataclass(finding.span) if finding.span is not None else None,
        "citation": _serialize_citation(finding.citation) if finding.citation is not None else None,
    }


def _read_findings(value: object) -> tuple[Finding, ...]:
    """The findings back, each with the reading it could not build."""
    if not isinstance(value, list):
        return ()
    return tuple(
        Finding(
            kind=FindingKind(_required_string(entry.get("kind"), name="findings.kind")),
            stage=_required_string(entry.get("stage"), name="findings.stage"),
            made_by=_required_string(entry.get("made_by"), name="findings.made_by"),
            message=_required_string(entry.get("message"), name="findings.message"),
            span=_optional_span(entry.get("span"), name="findings.span"),
            citation=(
                _read_citation(entry.get("citation"), name="findings.citation")
                if entry.get("citation") is not None
                else None
            ),
        )
        for entry in value
        if isinstance(entry, Mapping)
    )


def deserialize_document(payload: Mapping[str, object]) -> Document:
    """Recover one ``Document`` from its serialized artifact."""
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

    return Document(
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
        findings=_read_findings(payload.get("findings")),
        passes=tuple(
            _required_string(name, name="passes") for name in (payload.get("passes") or [])
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


def _deserialize_citation(payload: Mapping[str, object]) -> CitationRecord:
    """One record: what the rules read, what it is now, and the trace between."""
    source = _read_citation(payload.get("source"), name="citation.source")
    stated_payload = payload.get("stated")
    return CitationRecord(
        citation_id=_required_string(payload.get("citation_id"), name="citation.citation_id"),
        source=source,
        stated=(
            _read_citation(stated_payload, name="citation.stated")
            if isinstance(stated_payload, Mapping)
            else source
        ),
        resolves_to=_optional_string(payload.get("resolves_to"), name="citation.resolves_to"),
        root_id=_optional_string(payload.get("root_id"), name="citation.root_id"),
        colocation_id=_optional_string(payload.get("colocation_id"), name="citation.colocation_id"),
        authority_id=_optional_string(payload.get("authority_id"), name="citation.authority_id"),
        trace=_read_trace(payload.get("trace")),
    )


def _serialize_record(record: CitationRecord) -> dict[str, object]:
    """A record, with `stated` written only where it differs from `source`."""
    return {
        "citation_id": record.citation_id,
        "source": _serialize_citation(record.source),
        **({"stated": _serialize_citation(record.stated)} if record.stated != record.source else {}),
        "resolves_to": record.resolves_to,
        "root_id": record.root_id,
        "colocation_id": record.colocation_id,
        **({"authority_id": record.authority_id} if record.authority_id else {}),
        **({"trace": [_serialize_node(node) for node in record.trace]} if record.trace else {}),
    }


def _serialize_node(node: Node) -> dict[str, object]:
    """One node, and the corrections it justified."""
    return {
        "node_id": node.node_id,
        "reads": node.reads.value,
        "stage": node.stage,
        "made_by": node.made_by,
        "outcome": node.outcome,
        "message": node.message,
        "depends_on": list(node.depends_on),
        "corrections": [
            {
                "field": correction.field,
                "before": _serialize_value(correction.before),
                "after": _serialize_value(correction.after),
                "reason": correction.reason,
            }
            for correction in node.corrections
        ],
        # Written back as it arrived. Nothing here reads it: it is the node's
        # own record of what it did, in whatever shape the stage that made it
        # keeps, and interpreting it would make `core` know that stage's types.
        **({"details": dict(node.details)} if node.details else {}),
    }


def _serialize_value(value: object) -> object:
    """A corrected field's value, whatever kind of thing it is."""
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, CaseName):
        return _serialize_case_name(value)
    if isinstance(value, PinCite):
        return _serialize_pin_cite(value)
    return serialize_dataclass(value)


def _read_value(field: str, value: object) -> object:
    """The same, back. Which field it is says what shape to read it as."""
    if field == "case_name":
        return _read_case_name(value, name="citation.trace.correction")
    if field == "pin_cite":
        return _read_pin_cite(value)
    return value


def _read_trace(value: object) -> tuple[Node, ...]:
    """Every node on a record, in the order they happened."""
    if not isinstance(value, list):
        return ()
    return tuple(
        Node(
            node_id=_required_string(entry.get("node_id"), name="citation.trace.node_id"),
            reads=Reads(_required_string(entry.get("reads"), name="citation.trace.reads")),
            stage=_required_string(entry.get("stage"), name="citation.trace.stage"),
            made_by=_required_string(entry.get("made_by"), name="citation.trace.made_by"),
            outcome=_required_string(entry.get("outcome"), name="citation.trace.outcome"),
            message=_optional_string(entry.get("message"), name="citation.trace.message"),
            depends_on=tuple(entry.get("depends_on") or ()),
            corrections=tuple(
                Correction(
                    field=_required_string(item.get("field"), name="citation.trace.field"),
                    before=_read_value(str(item.get("field")), item.get("before")),
                    after=_read_value(str(item.get("field")), item.get("after")),
                    reason=_required_string(item.get("reason"), name="citation.trace.reason"),
                )
                for item in (entry.get("corrections") or [])
                if isinstance(item, Mapping)
            ),
            details=dict(entry["details"]) if isinstance(entry.get("details"), Mapping) else {},
        )
        for entry in value
        if isinstance(entry, Mapping)
    )


def _read_citation(value: object, *, name: str) -> CanonicalCitation:
    """One citation, from the one place the artifact writes it."""
    citation_payload = require_mapping(value, name=name)
    kind = CitationKind(_required_string(citation_payload.get("citation_type"), name=f"{name}.citation_type"))
    fields = {key: value for key, value in citation_payload.items() if key != "citation_type"}
    for field_name, rebuild in (("date", _deserialize_date), ("reporter", _deserialize_reporter)):
        if isinstance(fields.get(field_name), Mapping):
            fields[field_name] = rebuild(fields[field_name])
    for field_name in ("span", "locator_span"):
        fields[field_name] = _optional_span(fields.get(field_name), name=f"{name}.{field_name}")
    fields["case_name"] = _read_case_name(fields.get("case_name"), name=f"{name}.case_name")
    # `UnknownCitation` states no pin cite at all, so the key is not written for
    # it and must not be invented here.
    if "pin_cite" in fields:
        fields["pin_cite"] = _read_pin_cite(fields["pin_cite"])
    return _CITATION_TYPES[kind](**fields)


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
    """Rebuild a pin cite from a payload, pages included.

    The pages are **read back, not recomputed.** They used to be derived from
    the text again, which is right for a pin cite the rules read and wrong for
    every one a reader has since answered about: the whole finding of
    `written_but_no_page` is that these characters claim no page, and
    recomputing turns `74950` straight back into page 74,950. A stage that
    cannot write down what it decided has not decided anything.
    """
    if not isinstance(value, Mapping):
        return None
    return PinCite(
        span=_optional_span(value.get("span"), name="citation.pin_cite.span"),
        text=_required_string(value.get("text"), name="citation.pin_cite.text"),
        pages=_read_pin_cite_pages(value.get("pages")),
    )


def _optional_int(value: object, *, name: str) -> int | None:
    """An integer a payload may leave out, which a page with no number does."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        msg = f"{name} must be an integer"
        raise ValueError(msg)
    return value


def _read_pin_cite_pages(value: object) -> tuple[PinCitePages, ...]:
    """Which pages a pin cite claims, as the artifact holds them."""
    if not isinstance(value, list):
        return ()
    return tuple(
        PinCitePages(
            first=_optional_int(item.get("first"), name="citation.pin_cite.pages.first"),
            last=_optional_int(item.get("last"), name="citation.pin_cite.pages.last"),
            kind=PinCiteKind(_required_string(item.get("kind"), name="citation.pin_cite.pages.kind")),
            footnote=_optional_string(item.get("footnote"), name="citation.pin_cite.pages.footnote"),
            note=_optional_string(item.get("note"), name="citation.pin_cite.pages.note"),
        )
        for item in value
        if isinstance(item, Mapping)
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
