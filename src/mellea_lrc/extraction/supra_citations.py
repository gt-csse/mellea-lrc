"""Discover supra references independently of root attachment."""

from eyecite.models import SupraCitation as EyeciteSupra

from mellea_lrc.extraction.leaf_reading import events, require_leaves
from mellea_lrc.model.citations import SupraCitation
from mellea_lrc.model.document import Document
from mellea_lrc.model.span import Span

STAGE = "29_supra_citations"


def find_supra_citations(document: Document) -> Document:
    require_leaves(document, STAGE)
    for event in events(document.text):
        if not isinstance(event, EyeciteSupra):
            continue
        span = Span(*event.full_span())
        if not any(span.overlaps(c.site_span) for c in document.citations):
            document = document.add_citation(
                SupraCitation.from_source(source=document.text, span=span, stage=STAGE)
            )
    return document.complete(STAGE)
