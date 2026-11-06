"""Group adjacent locator occurrences for contextual field reading."""

from mellea_lrc.config.extraction import ExtractionRules, stable
from mellea_lrc.model.citations import FullReporterCitation
from mellea_lrc.model.document import Document
from mellea_lrc.parsing.colocations import colocation_groups

SUBSTAGE = "grow_roots.read_root_fields.resolve_colocations"


def resolve_colocations(document: Document, rules: ExtractionRules | None = None) -> Document:
    """Assign parsing groups after locator discovery, before reading context."""
    if SUBSTAGE in document.substage_runs:
        raise ValueError(f"Substage already completed: {SUBSTAGE}")
    if not {
        "grow_roots.discover_root_locators.find_full_reporter_locators",
        "grow_roots.discover_root_locators.find_docket_locators",
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
    for group in colocation_groups(document.text, sites, config.colocation_max_meaningful_gap):
        group_id = f"colocation:{group[0]}"
        for identifier in group:
            document = document.replace_citation(by_id[identifier].record(SUBSTAGE).with_colocation(group_id))
    return document.complete_substage(SUBSTAGE)
