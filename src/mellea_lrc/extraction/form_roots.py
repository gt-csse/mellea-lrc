"""Group full citation occurrences by their exact normalized locator keys."""

from mellea_lrc.extraction.resolve_colocations import SUBSTAGE as COLOCATIONS
from mellea_lrc.extraction.resolve_courts import SUBSTAGE as COURTS
from mellea_lrc.model.citations import FullDocketCitation, FullReporterCitation
from mellea_lrc.model.citations.fields.base import current_reading
from mellea_lrc.model.document import Document

SUBSTAGE = "grow_roots.form_root_groups.form_roots"


def _reporter_key(citation: FullReporterCitation) -> tuple[str, ...] | None:
    reading = citation.locator[-1]
    if not reading.normalizable:
        return None
    locator = reading.get_normalized()
    return (
        "reporter",
        str(locator.volume),
        "".join(locator.edition.casefold().split()),
        locator.page.casefold(),
    )


def _docket_key(citation: FullDocketCitation) -> tuple[str, ...] | None:
    court = current_reading(citation.court)
    locator = citation.locator[-1]
    if court is None or not court.normalizable or not locator.normalizable:
        return None
    return "docket", court.get_normalized().id, locator.get_normalized().docket_number.casefold()


def form_roots(document: Document) -> Document:
    """Group exact reporter and court-qualified docket matches in source order."""
    if SUBSTAGE in document.substage_runs:
        raise ValueError(f"Substage already completed: {SUBSTAGE}")
    if COLOCATIONS not in document.substage_runs:
        raise ValueError("Resolve colocations before forming roots")

    if COURTS not in document.substage_runs:
        raise ValueError("Resolve courts before forming roots")

    known: dict[tuple[str, ...], str] = {}
    for citation in document.full_locators:
        key = _reporter_key(citation) if isinstance(citation, FullReporterCitation) else _docket_key(citation)
        root_id = citation.id if key is None else known.setdefault(key, citation.id)
        document = document.replace_citation(citation.record(SUBSTAGE).with_root(root_id))
    return document.complete_substage(SUBSTAGE)
