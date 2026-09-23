"""Find exact full reporter and explicitly labelled federal docket locators."""

from __future__ import annotations

import re

from eyecite import get_citations
from eyecite.models import FullCaseCitation

from mellea_lrc.extraction.rules import ExtractionRules
from mellea_lrc.model.citations import FullCitationKind
from mellea_lrc.model.document import Document
from mellea_lrc.model.operations import CitationField
from mellea_lrc.model.span import Span
from mellea_lrc.preprocessing.document_index import is_within
from mellea_lrc.text_match import fuzzy_literal

_PREFIXES = ("No. ", "Case No. ", "Civil Action No. ", "Civ. A. No. ", "Docket No. ")
_PREFIX = r"\b(?:" + "|".join(fuzzy_literal(prefix, whitespace=True) for prefix in _PREFIXES) + ")"
# Office/year/type/sequence, with optional judge codes. Other jurisdictions and
# malformed forms belong to a later, independently reviewable hunting pass.
_CMECF = r"(?:\d{1,3}[:-])?\d{2}-[A-Za-z]{2,4}-\d{1,6}(?:-[A-Za-z]{2,5}){0,2}"
_DOCKET = re.compile(rf"{_PREFIX}(?P<number>{_CMECF})(?![A-Za-z0-9:/\\-])", re.IGNORECASE)
_ENTRY = re.compile(r"\b(?:Doc(?:ument)?\.?|Dkt\.?|ECF)\s*(?:No\.?\s*)?(?P<number>\d+(?:-\d+)?)", re.I)
_ENTRY_JOIN = re.compile(r"^[\s,;:\[\]()]{0,12}$")


def _overlaps(span: Span, document: Document) -> bool:
    return any(
        item.locator_span is not None
        and item.locator_span.start < span.end
        and span.start < item.locator_span.end
        for item in document.citations
    )


def find_full_reporter_locators(document: Document, rules: ExtractionRules | None = None) -> Document:
    """Create one typed occurrence for each eyecite full case reporter span."""
    del rules  # Eyecite's default reporter tokenizer is the stable first pass.
    stage = "full_reporter_locators"
    if stage in document.completed_stages:
        return document
    if "colocations" in document.completed_stages:
        raise ValueError("Discover all locators before resolving colocations")
    found = sorted(
        (citation for citation in get_citations(document.text) if isinstance(citation, FullCaseCitation)),
        key=lambda citation: citation.span(),
    )
    for match in found:
        start, end = match.span()
        span = Span(start, end)
        if is_within(span, document.index_spans) or _overlaps(span, document):
            continue
        identifier = f"reporter:{start}:{end}"
        document = document.create_citation(stage, identifier, FullCitationKind.REPORTER)
        document = document.update_fields(
            stage,
            identifier,
            {
                CitationField.LOCATOR_SPAN: span,
                CitationField.LOCATOR_TEXT: document.text[start:end],
                CitationField.VOLUME: match.groups.get("volume"),
                CitationField.REPORTER: match.groups.get("reporter"),
                CitationField.PAGE: match.groups.get("page"),
            },
        )
    return document.complete(stage)


def find_docket_locators(document: Document, rules: ExtractionRules | None = None) -> Document:
    """Create courtless docket occurrences from labelled CM/ECF numbers."""
    del rules
    stage = "docket_locators"
    if stage in document.completed_stages:
        return document
    if "colocations" in document.completed_stages:
        raise ValueError("Discover all locators before resolving colocations")
    for match in _DOCKET.finditer(document.text):
        span = Span(*match.span())
        if is_within(span, document.index_spans) or _overlaps(span, document):
            continue
        changes = {
            CitationField.LOCATOR_SPAN: span,
            CitationField.LOCATOR_TEXT: document.text[span.start : span.end],
            CitationField.DOCKET_NUMBER: match.group("number"),
        }
        before = tuple(_ENTRY.finditer(document.text, max(0, span.start - 96), span.start))
        if before and _ENTRY_JOIN.fullmatch(document.text[before[-1].end() : span.start]):
            entry = before[-1]
            changes[CitationField.DOCKET_ENTRY] = entry.group("number")
            changes[CitationField.DOCKET_ENTRY_SPAN] = Span(*entry.span())
        identifier = f"docket:{span.start}:{span.end}"
        document = document.create_citation(stage, identifier, FullCitationKind.DOCKET)
        document = document.update_fields(stage, identifier, changes)
    return document.complete(stage)
