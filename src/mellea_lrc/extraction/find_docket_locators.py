"""Create native docket occurrences from explicitly labelled CM/ECF numbers."""

from mellea_lrc.model.citations import FullDocketCitation
from mellea_lrc.model.document import Document
from mellea_lrc.model.span import Span
from mellea_lrc.parsing.dockets import docket_readings

SUBSTAGE = "grow_roots.discover_root_locators.find_docket_locators"


def find_docket_locators(document: Document) -> Document:
    """Save courtless docket occurrences with exact locator and number spans."""
    if SUBSTAGE in document.substage_runs:
        raise ValueError(f"Substage already completed: {SUBSTAGE}")
    if "grow_roots.read_root_fields.resolve_colocations" in document.substage_runs:
        raise ValueError("Discover all locators before resolving colocations")
    for reading in docket_readings(document.text):
        span = Span(*reading.span)
        if any(span.overlaps(item.site_span) for item in document.citations):
            continue
        identifier = f"docket:{span.start}:{span.end}"
        document = document.add_citation(
            FullDocketCitation.from_locator(
                citation_id=identifier,
                substage=SUBSTAGE,
                source=document.text,
                span=span,
                number_span=Span(*reading.number_span),
            )
        )
    return document.complete_substage(SUBSTAGE)
