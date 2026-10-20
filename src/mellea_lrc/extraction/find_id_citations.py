"""Read Id/Ibid spans and pinpoints; attribute them in the next substage."""

from eyecite.models import IdCitation as EyeciteId

from mellea_lrc.extraction.context.leaves import require_leaves
from mellea_lrc.model.citations import IdCitation
from mellea_lrc.model.document import Document
from mellea_lrc.parsing.events import events
from mellea_lrc.parsing.pin_cite import id_pin_after
from mellea_lrc.parsing.span import Span

SUBSTAGE = "grow_leaves.grow_id_leaves.find_id_citations"


def find_id_citations(document: Document) -> Document:
    require_leaves(document, SUBSTAGE)
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
        reading = id_pin_after(document.text, event.token.end, limit)
        pin = Span(*reading) if reading is not None else None
        if pin is not None:
            end = pin.end
        else:
            # Eyecite may treat an unlabelled footnote number as a pin. Our
            # explicit-join policy must govern both the field and the site.
            end = event.token.end
        span = Span(start, end)
        if not any(span.overlaps(c.site_span) for c in document.citations):
            document = document.add_citation(
                IdCitation.from_source(source=document.text, span=span, substage=SUBSTAGE, pin_span=pin)
            )
    return document.complete_substage(SUBSTAGE)
