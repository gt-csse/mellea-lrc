"""Read each short reporter's case name before its complete colocation group."""

from mellea_lrc.extraction.context.leaves import preceding_name, require_leaves
from mellea_lrc.extraction.short_reporter_colocations import SUBSTAGE as COLOCATION_SUBSTAGE
from mellea_lrc.model.citations import latest
from mellea_lrc.model.document import Document

SUBSTAGE = "grow_leaves.short_reporter_citations.case_names"


def resolve_short_reporter_case_names(document: Document) -> Document:
    """Quote one preceding name for every group member, including singletons.

    Colocation only relaxes the name-reading window. Each citation keeps its
    own locator, pinpoint, normalized reporter and eventual root attachment.
    """
    require_leaves(document, SUBSTAGE)
    if COLOCATION_SUBSTAGE not in document.substage_runs:
        raise ValueError("Resolve short reporter colocations before reading case names")
    groups = {group.id: group.citation_ids for group in document.short_reporter_colocations}
    by_id = {citation.id: citation for citation in document.short_reporters}
    names = {}
    for citation in document.short_reporters:
        group_id = latest(citation.colocation_id)
        first = by_id[groups[group_id][0]] if group_id is not None else citation
        if first.id not in names:
            names[first.id] = preceding_name(document, first.short_locator_span)
        updated = citation.record(SUBSTAGE)
        if (span := names[first.id]) is not None:
            updated = updated.with_case_name(document.text, span)
        document = document.replace_citation(updated)
    return document.complete_substage(SUBSTAGE)
