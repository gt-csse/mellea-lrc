"""Create named supra citations before field reading and attribution."""

from mellea_lrc.extraction.context.leaves import aliases, require_leaves
from mellea_lrc.model.citations import SupraCitation
from mellea_lrc.model.document import Document
from mellea_lrc.model.span import Span
from mellea_lrc.parsing.supra import supra_readings

SUBSTAGE = "grow_leaves.supra_citations.discovery"


def find_supra_citations(document: Document) -> Document:
    require_leaves(document, SUBSTAGE)
    for reading in supra_readings(document.text, names=tuple(aliases(document))):
        span = Span(*reading.span)
        if not any(span.overlaps(c.site_span) for c in document.citations):
            document = document.add_citation(
                SupraCitation.from_source(source=document.text, span=span, substage=SUBSTAGE)
            )
    return document.complete_substage(SUBSTAGE)
