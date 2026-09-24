"""Read dates from bounded post-locator parentheticals."""

from __future__ import annotations

from mellea_lrc.config.extraction import ExtractionRules, stable
from mellea_lrc.extraction._support.context_windows import dated_parenthetical, require_structure
from mellea_lrc.model.citations.fields.date import FULL_DATE_RE, YEAR_RE
from mellea_lrc.model.document import Document
from mellea_lrc.model.span import Span

STAGE = "dates"


def resolve_dates(document: Document, rules: ExtractionRules | None = None) -> Document:
    """Read an exact day or year from the bounded post-site parenthetical."""
    if STAGE in document.stage_runs:
        raise ValueError(f"Stage already completed: {STAGE}")
    require_structure(document)
    config = rules or stable()
    for citation in document.full_locators:
        found = dated_parenthetical(document, citation, config.post_locator_window)
        if not found:
            continue
        parenthetical, start = found
        body = parenthetical.group("body")
        match = FULL_DATE_RE.search(body) or YEAR_RE.search(body)
        if match is None:
            continue
        span = Span(
            start + parenthetical.start("body") + match.start(),
            start + parenthetical.start("body") + match.end(),
        )
        document = document.replace_citation(citation.record(STAGE).with_date(document.text, span))
    return document.complete(STAGE)
