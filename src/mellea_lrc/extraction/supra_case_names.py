"""Read the written case-name field of each created supra citation."""

from mellea_lrc.extraction.context.leaves import require_leaves
from mellea_lrc.model.citations import SupraCitation
from mellea_lrc.model.document import Document
from mellea_lrc.model.span import Span
from mellea_lrc.parsing.supra import supra_readings

SUBSTAGE = "grow_leaves.supra_citations.case_names"


def resolve_supra_case_names(document: Document) -> Document:
    require_leaves(document, SUBSTAGE)
    for citation in document.short_citations:
        if not isinstance(citation, SupraCitation):
            continue
        site = citation.site_span
        readings = supra_readings(document.text[site.start : site.end])
        if len(readings) != 1:
            raise ValueError("A saved supra citation must retain one source-shaped reference")
        name = readings[0].antecedent_span
        span = Span(site.start + name[0], site.start + name[1])
        document = document.replace_citation(citation.record(SUBSTAGE).with_case_name(document.text, span))
    return document.complete_substage(SUBSTAGE)
