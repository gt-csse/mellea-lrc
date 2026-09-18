"""JSON round-trip support for extracted documents."""

from __future__ import annotations

from collections.abc import Mapping

from mellea_lrc.core.case_names import CaseName
from mellea_lrc.core.citations import (
    CanonicalCitation,
    CitationDate,
    CitationKind,
    DocketCitation,
    DocketEntry,
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
from mellea_lrc.core.record import (
    CitationRecord,
    Correction,
    DateExploration,
    Judgement,
    Node,
    Question,
    Reads,
    Resolution,
    unjudged,
)
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

SCHEMA_VERSION = 17
_ARTIFACT_TYPE = "document"
_SUPPORTED_SCHEMA_VERSIONS = frozenset({15, 16, SCHEMA_VERSION})

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
        **({"nodes": [_serialize_node(node) for node in document.nodes]} if document.nodes else {}),
        "locators": [
            {
                "citation_id": locator.citation_id,
                "kind": locator.kind.value,
                "span": serialize_dataclass(locator.span),
                "text": locator.text,
            }
            for locator in document.locators
        ],
        "colocations": [list(group) for group in document.colocations],
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
        "node_id": finding.node_id,
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
            node_id=_optional_string(entry.get("node_id"), name="findings.node_id"),
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

    document = Document(
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
        nodes=_read_trace(payload.get("nodes")),
        unread_case_names=tuple(
            _optional_span(item, name="unread_case_names")
            for item in require_list(payload.get("unread_case_names", []), name="unread_case_names")
            if item is not None
        ),
        findings=_read_findings(payload.get("findings")),
        passes=tuple(_required_string(name, name="passes") for name in (payload.get("passes") or [])),
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
    return document


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
        found=_read_resolution(payload.get("found"), name="citation.found"),
        corrections=_read_corrections(payload.get("corrections")),
        judgements=_read_judgements(payload.get("judgements")),
        withdrawn_by=_optional_string(payload.get("withdrawn_by"), name="citation.withdrawn_by"),
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
        **({"found": _serialize_resolution(record.found)} if record.found is not None else {}),
        # What the pipeline currently says, each naming the node that said it.
        # Written flat rather than inside the trace: the trace is a graph, and
        # a reader after the current state should never have to walk one.
        "judgements": {
            question.value: {
                "outcome": judgement.outcome,
                "node_id": judgement.node_id,
                "message": judgement.message,
            }
            for question, judgement in record.judgements.items()
        },
        **({"withdrawn_by": record.withdrawn_by} if record.withdrawn_by else {}),
        **(
            {"corrections": [_serialize_correction(item) for item in record.corrections]}
            if record.corrections
            else {}
        ),
        **({"trace": [_serialize_node(node) for node in record.trace]} if record.trace else {}),
    }


def _serialize_resolution(resolution: Resolution) -> dict[str, object]:
    """What an archive holds, written only where it adds to the four fields every one has."""
    return {
        "cluster_id": resolution.cluster_id,
        "case_name": resolution.case_name,
        "date_filed": resolution.date_filed,
        "court_id": resolution.court_id,
        "node_id": resolution.node_id,
        **({"opinion_ids": list(resolution.opinion_ids)} if resolution.opinion_ids else {}),
        **({"citations": list(resolution.citations)} if resolution.citations else {}),
        **({"dates": serialize_dataclass(resolution.dates)} if resolution.dates is not None else {}),
        **(
            {"duplicates": [_serialize_resolution(item) for item in resolution.duplicates]}
            if resolution.duplicates
            else {}
        ),
        **({"docket_id": resolution.docket_id} if resolution.docket_id else {}),
        **({"govinfo_package_id": resolution.govinfo_package_id} if resolution.govinfo_package_id else {}),
    }


def _read_resolution(value: object, *, name: str) -> Resolution | None:
    if value is None:
        return None
    fields = require_mapping(value, name=name)
    return Resolution(
        cluster_id=_optional_string(fields.get("cluster_id"), name=f"{name}.cluster_id"),
        case_name=_optional_string(fields.get("case_name"), name=f"{name}.case_name"),
        date_filed=_optional_string(fields.get("date_filed"), name=f"{name}.date_filed"),
        court_id=_optional_string(fields.get("court_id"), name=f"{name}.court_id"),
        node_id=_required_string(fields.get("node_id"), name=f"{name}.node_id"),
        opinion_ids=tuple(
            _required_string(item, name=f"{name}.opinion_ids")
            for item in require_list(fields.get("opinion_ids", []), name=f"{name}.opinion_ids")
        ),
        citations=tuple(
            _required_string(item, name=f"{name}.citations")
            for item in require_list(fields.get("citations", []), name=f"{name}.citations")
        ),
        dates=_read_dates(fields.get("dates"), name=f"{name}.dates"),
        duplicates=tuple(
            resolution
            for item in require_list(fields.get("duplicates", []), name=f"{name}.duplicates")
            if (resolution := _read_resolution(item, name=f"{name}.duplicates")) is not None
        ),
        docket_id=_optional_string(fields.get("docket_id"), name=f"{name}.docket_id"),
        govinfo_package_id=_optional_string(
            fields.get("govinfo_package_id"), name=f"{name}.govinfo_package_id"
        ),
    )


def _read_dates(value: object, *, name: str) -> DateExploration | None:
    if value is None:
        return None
    fields = require_mapping(value, name=name)
    return DateExploration(
        stated=_required_string(fields.get("stated"), name=f"{name}.stated"),
        stated_precision=_required_string(fields.get("stated_precision"), name=f"{name}.stated_precision"),
        record_date_filed=_optional_string(fields.get("record_date_filed"), name=f"{name}.record_date_filed"),
        other_dates=_optional_string(fields.get("other_dates"), name=f"{name}.other_dates"),
        phrases_by_opinion=tuple(
            (str(pair[0]), tuple(str(phrase) for phrase in pair[1]))
            for pair in require_list(fields.get("phrases_by_opinion", []), name=f"{name}.phrases_by_opinion")
        ),
        matched_phrase=_optional_string(fields.get("matched_phrase"), name=f"{name}.matched_phrase"),
        matched_opinion_id=_optional_string(
            fields.get("matched_opinion_id"), name=f"{name}.matched_opinion_id"
        ),
    )


def _serialize_correction(correction: Correction) -> dict[str, object]:
    return {
        "field": correction.field,
        "before": _serialize_value(correction.before),
        "after": _serialize_value(correction.after),
        "reason": correction.reason,
        "node_id": correction.node_id,
    }


def _read_corrections(value: object) -> tuple[Correction, ...]:
    """Every change to `stated`, in the order they were made."""
    if not isinstance(value, list):
        return ()
    return tuple(
        Correction(
            field=_required_string(item.get("field"), name="citation.corrections.field"),
            before=_read_value(str(item.get("field")), item.get("before")),
            after=_read_value(str(item.get("field")), item.get("after")),
            reason=_required_string(item.get("reason"), name="citation.corrections.reason"),
            node_id=_required_string(item.get("node_id"), name="citation.corrections.node_id"),
        )
        for item in value
        if isinstance(item, Mapping)
    )


def _read_judgements(value: object) -> dict[Question, Judgement]:
    """One answer per question, with every question the pipeline asks present."""
    answers = unjudged()
    if not isinstance(value, Mapping):
        return answers
    for question in Question:
        written = value.get(question.value)
        if not isinstance(written, Mapping):
            continue
        answers[question] = Judgement(
            outcome=_required_string(written.get("outcome"), name="citation.judgements.outcome"),
            node_id=_optional_string(written.get("node_id"), name="citation.judgements.node_id"),
            message=_optional_string(written.get("message"), name="citation.judgements.message"),
        )
    return answers


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
    if isinstance(fields.get("docket_entry"), Mapping):
        fields["docket_entry"] = _deserialize_docket_entry(
            fields["docket_entry"], name=f"{name}.docket_entry"
        )
    fields["case_name"] = _read_case_name(fields.get("case_name"), name=f"{name}.case_name")
    # `UnknownCitation` states no pin cite at all, so the key is not written for
    # it and must not be invented here.
    if "pin_cite" in fields:
        fields["pin_cite"] = _read_pin_cite(fields["pin_cite"])
    return _CITATION_TYPES[kind](**fields)


def _deserialize_docket_entry(value: Mapping[str, object], *, name: str) -> DocketEntry:
    """Recover the optional written entry reference on a docket citation."""
    span = _optional_span(value.get("span"), name=f"{name}.span")
    if span is None:
        msg = f"{name}.span must be a span"
        raise ValueError(msg)
    return DocketEntry(
        number=_required_string(value.get("number"), name=f"{name}.number"),
        span=span,
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
    if payload.get("schema_version") not in _SUPPORTED_SCHEMA_VERSIONS:
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
