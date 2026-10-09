"""Group adjacent locator occurrences without asserting they share identity."""

from __future__ import annotations

from mellea_lrc.config.extraction import ExtractionRules, stable
from mellea_lrc.extraction.context.colocations import colocation_readings
from mellea_lrc.model.citations import FullReporterCitation
from mellea_lrc.model.document import Document

SUBSTAGE = "grow_roots.field_reading.colocations"


def resolve_colocations(document: Document, rules: ExtractionRules | None = None) -> Document:
    """Assign groups after all locator discovery, before any context read.

    A group is a parsing boundary and candidate parallel-citation site. It is
    never itself a finding that its identifiers refer to the same case.
    """
    if SUBSTAGE in document.substage_runs:
        raise ValueError(f"Substage already completed: {SUBSTAGE}")
    if not {
        "grow_roots.locator_discovery.full_reporter_locators",
        "grow_roots.locator_discovery.docket_locators",
    } & set(document.substage_runs):
        raise ValueError("Discover at least one kind of full locator before resolving colocations")
    config = rules or stable()
    sites = [
        (
            citation.id,
            citation.locator_span,
            citation.locator[-1].get_normalized().edition
            if isinstance(citation, FullReporterCitation) and citation.locator[-1].normalizable
            else None,
        )
        for citation in document.full_locators
    ]
    by_id = {citation.id: citation for citation in document.full_locators}
    for group in colocation_readings(document.text, sites, config.colocation_max_meaningful_gap):
        for identifier in group.citation_ids:
            document = document.replace_citation(by_id[identifier].record(SUBSTAGE).with_colocation(group.id))
    return document.complete_substage(SUBSTAGE)
