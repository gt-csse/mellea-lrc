"""Group created short reporter sites before reading their shared case names."""

from eyecite.models import ShortCaseCitation

from mellea_lrc.config.extraction import ExtractionRules, stable
from mellea_lrc.extraction.context.colocations import colocation_readings
from mellea_lrc.extraction.context.leaves import preceding_name, require_leaves
from mellea_lrc.extraction.short_reporter_locator import SUBSTAGE as CREATION_SUBSTAGE
from mellea_lrc.model.citations import ShortReporterCitation
from mellea_lrc.model.document import Document
from mellea_lrc.model.span import Span
from mellea_lrc.parsing.events import events

SUBSTAGE = "grow_leaves.short_reporter_citations.colocations"


def resolve_short_reporter_colocations(document: Document, rules: ExtractionRules | None = None) -> Document:
    """Apply the full locator grouping rule to already bounded short sites.

    Pinpoints are complete before this pass. Other recognized authorities
    and independently written names interrupt the short sequence; proximity
    never merges or assigns roots.
    """
    require_leaves(document, SUBSTAGE)
    if CREATION_SUBSTAGE not in document.substage_runs:
        raise ValueError("Create short reporter citations before resolving colocations")
    config = rules or stable()
    sites = [
        (
            citation.id,
            citation.short_locator_span,
            citation.short_locator[-1].get_normalized().edition
            if citation.short_locator[-1].normalizable
            else None,
        )
        for citation in document.short_reporters
    ]
    barriers = [
        Span(*event.span()) for event in events(document.text) if not isinstance(event, ShortCaseCitation)
    ]
    barriers.extend(c.site_span for c in document.citations if not isinstance(c, ShortReporterCitation))
    # A following short reporter with its own written name starts a new
    # occurrence even when that name fits inside the distance allowance.
    # Recognize its source boundary here; field writing stays in the next substage.
    barriers.extend(
        span
        for citation in document.short_reporters
        if (span := preceding_name(document, citation.short_locator_span)) is not None
    )
    by_id = {citation.id: citation for citation in document.short_reporters}
    for group in colocation_readings(
        document.text, sites, config.colocation_max_meaningful_gap, barriers=barriers
    ):
        for identifier in group.citation_ids:
            document = document.replace_citation(by_id[identifier].record(SUBSTAGE).with_colocation(group.id))
    return document.complete_substage(SUBSTAGE)
