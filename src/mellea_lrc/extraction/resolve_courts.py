"""Read written courts from colocation-bounded parentheticals."""

from __future__ import annotations

import re

from mellea_lrc.config.extraction import ExtractionRules, stable
from mellea_lrc.extraction._contextual import (
    ContextParenthetical,
    date_spans,
    looks_like_court_label,
    parenthetical_after,
    supported_date_span,
)
from mellea_lrc.extraction.resolve_colocations import SUBSTAGE as COLOCATIONS
from mellea_lrc.model.citations import FullCitationVariant, FullReporterCitation
from mellea_lrc.model.document import Document
from mellea_lrc.model.span import Span

SUBSTAGE = "grow_roots.read_root_fields.resolve_courts"

_DATE_EVENT = re.compile(r"\s+\b(?:filed|decided|issued)\b\s*$", re.I)


def resolve_courts(document: Document, rules: ExtractionRules | None = None) -> Document:
    """Read one written court after each locator group, or a safe inference.

    A dated parenthetical with no court text becomes ``NOT_STATED`` unless a
    normalized Supreme Court reporter supplies a unique inference. Missing or
    prose context remains unread. Citation-shaped labels are retained even
    when they fail court normalization.
    """
    if SUBSTAGE in document.substage_runs:
        raise ValueError(f"Substage already completed: {SUBSTAGE}")
    if COLOCATIONS not in document.substage_runs:
        raise ValueError("Resolve colocations before reading contextual fields")
    config = rules or stable()
    for citation in document.full_locators:
        if citation.court:
            continue
        context = parenthetical_after(document, citation, config.post_locator_window)
        if context is None:
            continue
        date_span = supported_date_span(context)
        if date_span is None and date_spans(context.body):
            continue
        court_span = _court_span(context, date_span[0] if date_span else len(context.body))
        if court_span is None and date_span is None:
            continue
        updated = citation.record(SUBSTAGE)
        if court_span is not None:
            updated = updated.with_court(document.text, court_span)
        elif (court_id := _infer_court(citation)) is not None:
            updated = updated.with_inferred_court(court_id)
        else:
            updated = updated.with_court_not_stated()
        document = document.replace_citation(updated)
    return document.complete_substage(SUBSTAGE)


def _court_span(context: ContextParenthetical, end: int) -> Span | None:
    region = context.body[:end]
    if event := _DATE_EVENT.search(region):
        region = region[: event.start()]
    leading = len(region) - len(region.lstrip(" ,;"))
    written = region.strip(" ,;")
    if not looks_like_court_label(written):
        return None
    start = context.body_start + leading
    return Span(start, start + len(written))


def _infer_court(citation: FullCitationVariant) -> str | None:
    """Infer only the one reporter-to-court identity guaranteed by eyecite."""
    if not isinstance(citation, FullReporterCitation):
        return None
    locator = citation.locator[-1]
    if not locator.normalizable:
        return None
    return "scotus" if locator.get_normalized().reporter.is_scotus else None
