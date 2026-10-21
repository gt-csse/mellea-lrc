"""Read each short reporter's name with document-aware rules or saved ML spans."""

from mellea_lrc.config.field_extraction import FieldExtractionConfig, FieldSource
from mellea_lrc.extraction.context.leaves import preceding_name, require_leaves
from mellea_lrc.extraction.resolve_short_reporter_colocations import SUBSTAGE as COLOCATION_SUBSTAGE
from mellea_lrc.model.citations import latest
from mellea_lrc.model.document import Document

SUBSTAGE = "grow_leaves.grow_short_reporter_leaves.resolve_case_names"


def resolve_short_reporter_case_names(
    document: Document,
    *,
    field_extraction: FieldExtractionConfig | None = None,
) -> Document:
    """Read before a short-locator group; retain each member's separate pin and attribution."""
    require_leaves(document, SUBSTAGE)
    if COLOCATION_SUBSTAGE not in document.substage_runs:
        raise ValueError("Resolve short reporter colocations before reading case names")
    fields = field_extraction or FieldExtractionConfig.from_env()
    citations = document.short_reporters
    if fields.case_name is FieldSource.ML:
        spans = [
            fields.validate_prediction(citation.get_ml_field_prediction()).case_name for citation in citations
        ]
    else:
        groups = {group.id: group.citation_ids for group in document.short_reporter_colocations}
        by_id = {citation.id: citation for citation in citations}
        names = {}
        spans = []
        for citation in citations:
            group_id = latest(citation.colocation_id)
            first = by_id[groups[group_id][0]] if group_id is not None else citation
            if first.id not in names:
                names[first.id] = preceding_name(document, first.short_locator_span)
            spans.append(names[first.id])
    for citation, span in zip(citations, spans, strict=True):
        updated = citation.record(SUBSTAGE)
        if span is not None:
            updated = updated.with_case_name(document.text, span)
        document = document.replace_citation(updated)
    return document.complete_substage(SUBSTAGE)
