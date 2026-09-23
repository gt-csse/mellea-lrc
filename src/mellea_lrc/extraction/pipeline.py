"""One readable composition of the independent first-pass extraction stages."""

from __future__ import annotations

from mellea_lrc.extraction.context import (
    resolve_case_names,
    resolve_courts,
    resolve_dates,
    resolve_pin_cites,
)
from mellea_lrc.extraction.locators import find_docket_locators, find_full_reporter_locators
from mellea_lrc.extraction.roots import form_roots
from mellea_lrc.extraction.rules import ExtractionRules, stable
from mellea_lrc.extraction.structure import resolve_colocations
from mellea_lrc.model.document import Document


def grow_roots(
    document: Document,
    *,
    rules: ExtractionRules | None = None,
) -> Document:
    """Run the current rule-based stages through root formation."""
    config = rules or stable()
    document = find_full_reporter_locators(document)
    document = find_docket_locators(document)
    document = resolve_colocations(document, config)
    document = resolve_case_names(document, config)
    document = resolve_courts(document, config)
    document = resolve_dates(document, config)
    document = resolve_pin_cites(document, config)
    return form_roots(document)
