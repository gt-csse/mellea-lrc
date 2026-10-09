"""Read each created supra citation's optional pinpoint field."""

from mellea_lrc.extraction.context.leaves import require_leaves
from mellea_lrc.model.citations import SupraCitation
from mellea_lrc.model.document import Document
from mellea_lrc.model.span import Span
from mellea_lrc.parsing.supra import supra_readings

SUBSTAGE = "grow_leaves.supra_citations.pin_cites"


def resolve_supra_pin_cites(document: Document) -> Document:
    require_leaves(document, SUBSTAGE)
    for citation in document.short_citations:
        if not isinstance(citation, SupraCitation):
            continue
        site = citation.site_span
        readings = supra_readings(document.text[site.start : site.end])
        if len(readings) != 1:
            raise ValueError("A saved supra citation must retain one source-shaped reference")
        if pin := readings[0].pin_span:
            span = Span(site.start + pin[0], site.start + pin[1])
            document = document.replace_citation(citation.record(SUBSTAGE).with_pin_cite(document.text, span))
    return document.complete_substage(SUBSTAGE)
