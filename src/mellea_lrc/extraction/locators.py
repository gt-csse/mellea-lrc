"""Find exact full reporter and explicitly labelled federal docket locators."""

from __future__ import annotations

import re

from mellea_lrc.model.citations import FullDocketCitation, FullReporterCitation, ShortReporterCitation
from mellea_lrc.model.document import Document
from mellea_lrc.model.span import Span
from mellea_lrc.preprocessing.document_index import is_within
from mellea_lrc.reporter_reading import full_reporter_readings, short_reporter_readings
from mellea_lrc.text_match import fuzzy_literal

FULL_REPORTER_LOCATORS_STAGE = "full_reporter_locators"
SHORT_REPORTER_CITATIONS_STAGE = "short_reporter_citations"
DOCKET_LOCATORS_STAGE = "docket_locators"

_PREFIXES = ("No. ", "Case No. ", "Civil Action No. ", "Civ. A. No. ", "Docket No. ")
DOCKET_PREFIX_PATTERN = (
    r"\b(?:" + "|".join(fuzzy_literal(prefix, whitespace=True) for prefix in _PREFIXES) + ")"
)
# Office/year/type/sequence, with optional judge codes. Other jurisdictions and
# malformed forms belong to a later, independently reviewable hunting pass.
_CMECF = r"(?:\d{1,3}[:-])?\d{2}-[A-Za-z]{2,4}-\d{1,6}(?:-[A-Za-z]{2,5}){0,2}"
_DOCKET = re.compile(rf"{DOCKET_PREFIX_PATTERN}(?P<number>{_CMECF})(?![A-Za-z0-9:/\\-])", re.IGNORECASE)


def _overlaps(span: Span, document: Document) -> bool:
    for item in document.citations:
        site = item.site_span
        if site.start < span.end and span.start < site.end:
            return True
    return False


def find_full_reporter_locators(document: Document) -> Document:
    """Create one typed occurrence for each shared-reader reporter span.

    The field re-reads its exact quote with the same reader. That keeps its
    normalized identity recoverable from a serialized citation even if eyecite
    used surrounding document context while finding the span.
    """
    if FULL_REPORTER_LOCATORS_STAGE in document.stage_runs:
        raise ValueError(f"Stage already completed: {FULL_REPORTER_LOCATORS_STAGE}")
    if "colocations" in document.stage_runs:
        raise ValueError("Discover all locators before resolving colocations")
    for reading in sorted(full_reporter_readings(document.text), key=lambda item: item.span):
        span = Span(*reading.span)
        if is_within(span, document.index_spans) or _overlaps(span, document):
            continue
        identifier = f"reporter:{span.start}:{span.end}"
        document = document.add_citation(
            FullReporterCitation.from_locator(
                citation_id=identifier,
                stage=FULL_REPORTER_LOCATORS_STAGE,
                source=document.text,
                span=span,
            )
        )
    return document.complete(FULL_REPORTER_LOCATORS_STAGE)


def find_short_reporter_citations(document: Document) -> Document:
    """Record eyecite short-case sites as short citations, without growing leaves.

    This optional stage does not participate in full-locator colocation or
    root formation. Later leaf growth may attach its occurrences to roots.
    """
    if SHORT_REPORTER_CITATIONS_STAGE in document.stage_runs:
        raise ValueError(f"Stage already completed: {SHORT_REPORTER_CITATIONS_STAGE}")
    if "roots" not in document.stage_runs:
        raise ValueError("Form full roots before finding short reporter citations")
    for reading in sorted(short_reporter_readings(document.text), key=lambda item: item.span):
        span = Span(*reading.span)
        if is_within(span, document.index_spans) or _overlaps(span, document):
            continue
        identifier = f"short-reporter:{span.start}:{span.end}"
        document = document.add_citation(
            ShortReporterCitation.from_short_locator(
                citation_id=identifier,
                stage=SHORT_REPORTER_CITATIONS_STAGE,
                source=document.text,
                span=span,
            )
        )
    return document.complete(SHORT_REPORTER_CITATIONS_STAGE)


def find_docket_locators(document: Document) -> Document:
    """Create courtless docket occurrences from labelled CM/ECF numbers."""
    if DOCKET_LOCATORS_STAGE in document.stage_runs:
        raise ValueError(f"Stage already completed: {DOCKET_LOCATORS_STAGE}")
    if "colocations" in document.stage_runs:
        raise ValueError("Discover all locators before resolving colocations")
    for match in _DOCKET.finditer(document.text):
        span = Span(*match.span())
        if is_within(span, document.index_spans) or _overlaps(span, document):
            continue
        identifier = f"docket:{span.start}:{span.end}"
        document = document.add_citation(
            FullDocketCitation.from_locator(
                citation_id=identifier,
                stage=DOCKET_LOCATORS_STAGE,
                source=document.text,
                span=span,
                number_span=Span(*match.span("number")),
            )
        )
    return document.complete(DOCKET_LOCATORS_STAGE)
