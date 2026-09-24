"""One readable composition of the independent first-pass extraction stages."""

from __future__ import annotations

from mellea_lrc.config.extraction import ExtractionRules, stable
from mellea_lrc.extraction._site_hunting.review import DocketSiteReviewer
from mellea_lrc.extraction.case_names import resolve_case_names
from mellea_lrc.extraction.colocations import resolve_colocations
from mellea_lrc.extraction.courts import resolve_courts
from mellea_lrc.extraction.dates import resolve_dates
from mellea_lrc.extraction.docket_entries import resolve_docket_entries
from mellea_lrc.extraction.docket_locator import find_docket_locators
from mellea_lrc.extraction.docket_site_hunting import hunt_docket_locators
from mellea_lrc.extraction.full_reporter_locator import find_full_reporter_locators
from mellea_lrc.extraction.pin_cites import resolve_pin_cites
from mellea_lrc.extraction.roots import form_roots
from mellea_lrc.model.document import Document


async def grow_roots(
    document: Document,
    *,
    rules: ExtractionRules | None = None,
    hunt_dockets: bool = False,
    reviewer: DocketSiteReviewer | None = None,
) -> Document:
    """Read full locators, optionally hunt dockets, then form roots once."""
    config = rules or stable()
    document = find_full_reporter_locators(document)
    document = find_docket_locators(document)
    if hunt_dockets:
        document = await hunt_docket_locators(document, reviewer=reviewer)
    document = resolve_docket_entries(document)
    document = resolve_colocations(document, config)
    document = resolve_case_names(document, config)
    document = resolve_courts(document, config)
    document = resolve_dates(document, config)
    document = resolve_pin_cites(document, config)
    return form_roots(document)
