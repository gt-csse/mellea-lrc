"""Discover Id/Ibid sites; resolve their antecedents in a later stage."""

import re

from eyecite.models import IdCitation as EyeciteId

from mellea_lrc.extraction.leaf_reading import events, pin_after, require_leaves
from mellea_lrc.model.citations import IdCitation
from mellea_lrc.model.document import Document
from mellea_lrc.model.span import Span

STAGE = "30_id_citations"


def id_pin_span(source: str, position: int, end: int) -> Span | None:
    """An Id. pin needs an explicit join or paragraph/star marker.

    A bare number following whitespace can be a footnote label or list item,
    rather than a pin. Do not consume it merely because PIN_PREFIX permits
    unlabelled page numbers for other citation kinds.
    """
    if re.match(r"\s*,?\s*(?:at\s+|(?=[¶*]))", source[position:end], re.I) is None:
        return None
    return pin_after(source, position, end)


def find_id_citations(document: Document) -> Document:
    require_leaves(document, STAGE)
    for event in events(document.text):
        if not isinstance(event, EyeciteId):
            continue
        start, end = event.span()
        if pin := id_pin_span(document.text, event.token.end, event.token.end + 100):
            end = pin.end
        span = Span(start, end)
        if not any(span.overlaps(c.site_span) for c in document.citations):
            document = document.add_citation(
                IdCitation.from_source(source=document.text, span=span, stage=STAGE)
            )
    return document.complete(STAGE)
