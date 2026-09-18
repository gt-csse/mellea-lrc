"""Independent, checkpointable readings of complete case locators.

A root locator is admitted before the project reads the surrounding case name,
court, date, or pin cite.  This module exposes those admissions as small,
``Document -> Document`` stages.  Each stage adds a citation-owned trace node,
then recalculates only the low-level structure that follows from locator spans:
co-location and root pointers.

The intended chain is::

    start_locator_document(preprocessed)
      -> find_full_reporter_locators(document)
      -> find_docket_locators(document)
      -> hunt_full_reporter_locators(document)  # optional plugin
      -> hunt_docket_locators(document)         # optional plugin

The two hunting calls are deliberately not coupled to the deterministic
readers.  A caller can checkpoint, deserialize, and continue at every arrow.
Field readers live in :mod:`mellea_lrc.extraction.stages` and run only after
this chain has settled the locator graph.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from mellea_lrc.core.citations import DocketCitation, FullCaseCitation
from mellea_lrc.core.record import Node, Reads
from mellea_lrc.core.spans import Span
from mellea_lrc.extraction.eyecite_extractor import grow_roots
from mellea_lrc.extraction.reading.unread_names import unread_case_names
from mellea_lrc.extraction.rules import ExtractionRules, stable
from mellea_lrc.extraction.structure.citation_tree import assign_roots
from mellea_lrc.extraction.structure.colocation import assign_colocation
from mellea_lrc.extraction.types import CitationRecord, Document, ExtractionMetadata
from mellea_lrc.preprocessing.types import PreprocessedDocument

if TYPE_CHECKING:
    from collections.abc import Iterable

REPORTER_RULE_STAGE = "full_reporter_locator_rule"
DOCKET_RULE_STAGE = "docket_locator_rule"
REPORTER_SITE_STAGE = "full_reporter_locator_site_hunting"
DOCKET_SITE_STAGE = "docket_locator_site_hunting"

_MADE_BY = "mellea_lrc.extraction.locator_stages"


def start_locator_document(
    preprocessed: PreprocessedDocument,
    *,
    rules: ExtractionRules | None = None,
) -> Document:
    """Create an empty, serializable document for independent locator stages."""
    effective = stable(rules)
    return Document(
        source_metadata=preprocessed.source_metadata,
        text=preprocessed.text,
        preprocessing_metadata=preprocessed.preprocessing_metadata,
        index_spans=preprocessed.index_spans,
        citations=(),
        passes=(),
        extraction_metadata=ExtractionMetadata(relaxation=effective.relaxation),
    )


def find_full_reporter_locators(
    document: Document,
    *,
    rules: ExtractionRules | None = None,
) -> Document:
    """Append rule-read reporter locator occurrences and recompute structure.

    The output contains only locator fields.  Case names, courts, dates, and
    pin cites are intentionally blank until their explicit readers run.
    """
    candidates = _read_locator_candidates(document, rules=rules, include_dockets=False)
    return _admit(document, candidates, kind=FullCaseCitation, stage=REPORTER_RULE_STAGE, rules=rules)


def find_docket_locators(
    document: Document,
    *,
    rules: ExtractionRules | None = None,
) -> Document:
    """Append rule-read federal CM/ECF docket locator occurrences.

    This does not audit a docket or require its court.  It records the exact
    locator first; court and date resolution remains a later operation.
    """
    candidates = _read_locator_candidates(document, rules=rules, include_dockets=True)
    return _admit(document, candidates, kind=DocketCitation, stage=DOCKET_RULE_STAGE, rules=rules)


def mark_full_reporter_locator_hunting_skipped(
    document: Document,
    *,
    reason: str,
) -> Document:
    """Persist a deliberate no-op for an optional reporter-hunting stage.

    A skipped stage is document-level configuration evidence, not evidence for
    any particular citation.  It therefore has one document node and creates
    no synthetic citation trace nodes.
    """
    node_id = f"{REPORTER_SITE_STAGE}:not_run"
    if any(node.node_id == node_id for node in document.nodes):
        return _with_pass(document, REPORTER_SITE_STAGE)
    node = Node(
        node_id=node_id,
        reads=Reads.DOCUMENT,
        stage=REPORTER_SITE_STAGE,
        made_by=_MADE_BY,
        outcome="not_run",
        message=reason,
        details={"enabled": False},
    )
    return _with_pass(replace(document, nodes=(*document.nodes, node)), REPORTER_SITE_STAGE)


def rebuild_locator_structure(
    document: Document,
    *,
    rules: ExtractionRules | None = None,
) -> Document:
    """Recompute co-location and roots from the locator records already held.

    This is the only structural work locator admission performs.  It deliberately
    does not call a court, date, name, pin-cite, or docket-audit reader.
    """
    effective = stable(rules)
    citations = tuple(document.citations)
    if effective.colocation_reader is not None:
        citations = effective.colocation_reader(document.text, citations)
    else:
        citations = assign_colocation(document.text, citations)
    citations = assign_roots(citations)
    return replace(
        document,
        citations=citations,
        unread_case_names=unread_case_names(document.text, citations),
    )


def _read_locator_candidates(
    document: Document,
    *,
    rules: ExtractionRules | None,
    include_dockets: bool,
) -> tuple[CitationRecord, ...]:
    """Use eyecite's tokenizer while retaining only the requested locator kind.

    ``grow_roots`` remains the sole conversion from eyecite's internal objects
    into canonical citations.  All project field readers are disabled here, so
    their output cannot leak into a locator checkpoint.  The canonical objects
    are then trimmed to locator data by :func:`_locator_only`.
    """
    effective = stable(rules)
    tokenizer_factory = effective.tokenizer_factory
    if not include_dockets:
        from mellea_lrc.extraction.reading.relaxation import tokenizer_for

        tokenizer_factory = tokenizer_for
    locator_rules = replace(
        effective,
        read_dockets=include_dockets,
        tokenizer_factory=tokenizer_factory,
        case_name_reader=None,
        case_name_field_reader=None,
        pin_cite_reader=None,
        colocation_reader=None,
        docket_auditor=None,
        court_reader=None,
        date_reader=None,
    )
    preprocessed = PreprocessedDocument(
        source_metadata=document.source_metadata,
        text=document.text,
        preprocessing_metadata=document.preprocessing_metadata,
        index_spans=document.index_spans,
    )
    read = grow_roots(preprocessed, rules=locator_rules)
    wanted = DocketCitation if include_dockets else FullCaseCitation
    return tuple(record for record in read.citations if isinstance(record.source, wanted))


def _admit(
    document: Document,
    candidates: Iterable[CitationRecord],
    *,
    kind: type[FullCaseCitation] | type[DocketCitation],
    stage: str,
    rules: ExtractionRules | None,
) -> Document:
    """Append unseen candidates with their exact rule-admission evidence."""
    held = {record.citation_id for record in document.citations}
    admitted: list[CitationRecord] = []
    for candidate in candidates:
        if candidate.citation_id in held:
            continue
        citation = _locator_only(candidate.source)
        if not isinstance(citation, kind):
            continue
        record = CitationRecord(citation_id=candidate.citation_id, source=citation)
        record.observe(
            Node(
                node_id=f"{stage}:{record.citation_id}",
                reads=Reads.DOCUMENT,
                stage=stage,
                made_by=_MADE_BY,
                outcome="identified",
                details={
                    "locator": {
                        "span": {"start": citation.locator_span.start, "end": citation.locator_span.end},
                        "text": document.text[citation.locator_span.start : citation.locator_span.end],
                    },
                    "citation_type": type(citation).__name__,
                    "reader": "eyecite",
                },
            )
        )
        admitted.append(record)
        held.add(record.citation_id)

    appended = replace(
        document,
        citations=tuple(sorted((*document.citations, *admitted), key=_citation_order)),
    )
    return _with_pass(rebuild_locator_structure(appended, rules=rules), stage)


def _locator_only(citation: FullCaseCitation | DocketCitation) -> FullCaseCitation | DocketCitation:
    """Remove fields whose readers run after the locator chain."""
    locator_span = citation.locator_span
    if locator_span is None:
        msg = "A locator-stage citation must have a locator span"
        raise ValueError(msg)
    if isinstance(citation, FullCaseCitation):
        return replace(
            citation,
            span=locator_span,
            case_name=None,
            plaintiff=None,
            defendant=None,
            pin_cite=None,
            extra=None,
            date=None,
            court=None,
            parenthetical=None,
            antecedent=None,
        )
    return replace(
        citation,
        span=_docket_span(citation, locator_span),
        case_name=None,
        plaintiff=None,
        defendant=None,
        court=None,
        court_name=None,
        court_text=None,
        pin_cite=None,
        date=None,
        parenthetical=None,
    )


def _docket_span(citation: DocketCitation, locator_span: Span) -> Span:
    """Include an adjacent docket entry without treating later fields as read."""
    if citation.docket_entry is None:
        return locator_span
    return Span(start=citation.docket_entry.span.start, end=locator_span.end)


def _citation_order(record: CitationRecord) -> tuple[int, int, str]:
    return record.full_span.start, record.full_span.end, record.citation_id


def _with_pass(document: Document, stage: str) -> Document:
    """Record a stage once, so resuming a completed checkpoint is idempotent."""
    if stage in document.passes:
        return document
    return replace(document, passes=(*document.passes, stage))
