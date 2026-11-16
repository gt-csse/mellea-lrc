"""Read written decision dates from colocation-bounded parentheticals."""

from __future__ import annotations

from mellea_lrc.config.extraction import ExtractionRules, stable
from mellea_lrc.extraction._contextual import date_spans, parenthetical_after, supported_date_span
from mellea_lrc.extraction.resolve_colocations import SUBSTAGE as COLOCATIONS
from mellea_lrc.model.citations.fields.court import court_id_if_unique
from mellea_lrc.model.document import Document
from mellea_lrc.model.span import Span

SUBSTAGE = "grow_roots.read_root_fields.resolve_dates"


def resolve_dates(document: Document, rules: ExtractionRules | None = None) -> Document:
    """Read one unambiguous written date after each full locator group.

    An explicit written court with no date records ``NOT_STATED``. Missing,
    truncated, prose, or competing context remains unread for later review.
    """
    if SUBSTAGE in document.substage_runs:
        raise ValueError(f"Substage already completed: {SUBSTAGE}")
    if COLOCATIONS not in document.substage_runs:
        raise ValueError("Resolve colocations before reading contextual fields")
    config = rules or stable()
    for citation in document.full_locators:
        if citation.date:
            continue
        context = parenthetical_after(document, citation, config.post_locator_window)
        if context is None:
            continue
        if match := supported_date_span(context):
            start, end = match
            updated = citation.record(SUBSTAGE).with_date(
                document.text,
                Span(context.body_start + start, context.body_start + end),
            )
        elif not date_spans(context.body) and court_id_if_unique(context.body.strip(" ,;")) is not None:
            updated = citation.record(SUBSTAGE).with_date_not_stated()
        else:
            continue
        document = document.replace_citation(updated)
    return document.complete_substage(SUBSTAGE)
