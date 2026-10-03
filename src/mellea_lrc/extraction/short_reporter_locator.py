"""Create short reporter occurrences before their attribution."""

import re

from mellea_lrc.extraction.context.leaves import pin_after
from mellea_lrc.model.citations import ShortReporterCitation
from mellea_lrc.model.document import Document
from mellea_lrc.model.span import Span
from mellea_lrc.parsing.reporters import short_reporter_readings

STAGE = "28_short_reporter_citations"


def find_short_reporter_citations(document: Document) -> Document:
    """Create source-grounded short reporters with their locator and pin.

    Normalization belongs to each quoted field. No root assignment or
    semantic attribution is made here; those belong to the later attribution stage.
    """
    if STAGE in document.stage_runs:
        raise ValueError(f"Stage already completed: {STAGE}")
    if "10_roots" not in document.stage_runs:
        raise ValueError("Form full roots before finding short reporter citations")
    for reading in sorted(short_reporter_readings(document.text), key=lambda item: item.span):
        span = Span(*reading.span)
        if any(span.overlaps(item.site_span) for item in document.citations):
            continue
        # The locator span includes the pinpoint that identifies this as a
        # short form. Only the separate pin field normalizes its targets.
        marker = re.search(r"\bat\s+", document.text[span.start : span.end], re.I)
        pin = pin_after(document.text, span.start + marker.end(), span.end) if marker else None
        identifier = f"short-reporter:{span.start}:{span.end}"
        document = document.add_citation(
            ShortReporterCitation.from_short_locator(
                citation_id=identifier,
                stage=STAGE,
                source=document.text,
                span=span,
                pin_cite_span=pin,
            )
        )
    return document.complete(STAGE)
