"""Read Id/Ibid spans and pinpoints; attribute them in the next stage."""

import re

from eyecite.models import IdCitation as EyeciteId

from mellea_lrc.extraction.context.leaves import pin_after, require_leaves
from mellea_lrc.matching.literal import fuzzy_literal
from mellea_lrc.model.citations import IdCitation
from mellea_lrc.model.document import Document
from mellea_lrc.model.span import Span
from mellea_lrc.parsing.events import events

STAGE = "32_id_citations"
_AT_JOIN = fuzzy_literal("at ", whitespace=True, newline=True)


def id_pin_span(source: str, position: int, end: int) -> Span | None:
    """An Id. pin needs an explicit join or paragraph/star marker.

    A bare number following whitespace can be a footnote label or list item,
    rather than a pin. Do not consume it merely because PIN_PREFIX permits
    unlabelled page numbers for other citation kinds.
    """
    if re.match(rf"\s*,?\s*(?:{_AT_JOIN}|(?=[¶*]))", source[position:end], re.I) is None:
        return None
    return pin_after(source, position, end)


def find_id_citations(document: Document) -> Document:
    require_leaves(document, STAGE)
    source_events = events(document.text)
    boundaries = sorted(
        {c.site_span.start for c in document.citations} | {event.span()[0] for event in source_events}
    )
    for event in source_events:
        if not isinstance(event, EyeciteId):
            continue
        start, end = event.span()
        # A pinpoint may not consume the following authority. Use all eyecite
        # events, including noncase sources, as well as existing citation sites.
        limit = min(
            event.token.end + 100,
            next((position for position in boundaries if position > start), len(document.text)),
        )
        pin = id_pin_span(document.text, event.token.end, limit)
        if pin is not None:
            end = pin.end
        else:
            # Eyecite may treat an unlabelled footnote number as a pin. Our
            # explicit-join policy must govern both the field and the site.
            end = event.token.end
        span = Span(start, end)
        if not any(span.overlaps(c.site_span) for c in document.citations):
            document = document.add_citation(
                IdCitation.from_source(source=document.text, span=span, stage=STAGE, pin_span=pin)
            )
    return document.complete(STAGE)
