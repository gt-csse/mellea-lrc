"""Short reporter citation stage and its eyecite reading."""

from __future__ import annotations

import re
from dataclasses import dataclass

from eyecite import get_citations
from eyecite.models import ShortCaseCitation

from mellea_lrc.extraction.full_reporter_locator import _reporter_tokenizer
from mellea_lrc.extraction.leaf_reading import pin_after
from mellea_lrc.model.citations import ShortReporterCitation
from mellea_lrc.model.document import Document
from mellea_lrc.model.span import Span

STAGE = "28_short_reporter_citations"


@dataclass(frozen=True, slots=True)
class ShortReporterReading:
    """A short-case eyecite result indexed to unchanged source text."""

    span: tuple[int, int]
    citation: ShortCaseCitation


def short_reporter_readings(source: str) -> tuple[ShortReporterReading, ...]:
    """Read short reporter sites, including their first pinpoint page."""
    readings: list[ShortReporterReading] = []
    for citation in get_citations(source, tokenizer=_reporter_tokenizer()):
        if not isinstance(citation, ShortCaseCitation):
            continue
        start, end = citation.span_with_pincite()
        # Eyecite identifies the reporter; the common pin grammar completes
        # spaced ranges, lists and footnotes without changing source offsets.
        marker = re.search(r"\bat\s+", source[start:end], re.I)
        if marker and (pin := pin_after(source, start + marker.end(), end + 100)):
            end = pin.end
        readings.append(ShortReporterReading(span=(start, end), citation=citation))
    return tuple(readings)


def find_short_reporter_citations(document: Document) -> Document:
    """Record eyecite short-case sites as short citations, without growing leaves.

    This optional stage does not participate in full-locator colocation or
    root formation. Later leaf growth may attach its occurrences to roots.
    """
    if STAGE in document.stage_runs:
        raise ValueError(f"Stage already completed: {STAGE}")
    if "10_roots" not in document.stage_runs:
        raise ValueError("Form full roots before finding short reporter citations")
    for reading in sorted(short_reporter_readings(document.text), key=lambda item: item.span):
        span = Span(*reading.span)
        if any(span.overlaps(item.site_span) for item in document.citations):
            continue
        identifier = f"short-reporter:{span.start}:{span.end}"
        document = document.add_citation(
            ShortReporterCitation.from_short_locator(
                citation_id=identifier,
                stage=STAGE,
                source=document.text,
                span=span,
            )
        )
    return document.complete(STAGE)
