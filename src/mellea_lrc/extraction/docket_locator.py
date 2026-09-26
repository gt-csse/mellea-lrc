"""Rule-based docket locator stage and its narrow CM/ECF reader."""

from __future__ import annotations

import re
from dataclasses import dataclass

from mellea_lrc.model.citations import FullDocketCitation
from mellea_lrc.model.document import Document
from mellea_lrc.model.span import Span
from mellea_lrc.text_match import fuzzy_literal

STAGE = "docket_locators"

_PREFIXES = ("No. ", "Case No. ", "Civil Action No. ", "Civ. A. No. ", "Docket No. ")
DOCKET_PREFIX_PATTERN = (
    r"\b(?:" + "|".join(fuzzy_literal(prefix, whitespace=True) for prefix in _PREFIXES) + ")"
)
# Office/year/type/sequence, with optional judge codes. Other jurisdictions and
# malformed forms belong to a later, independently reviewable hunting pass.
_CMECF = r"(?:\d{1,3}[:-])?\d{2}-[A-Za-z]{2,4}-\d{1,6}(?:-[A-Za-z]{2,5}){0,2}"
_DOCKET = re.compile(rf"{DOCKET_PREFIX_PATTERN}(?P<number>{_CMECF})(?![A-Za-z0-9:/\\-])", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class DocketReading:
    """A docket locator and number span in unchanged source text."""

    span: tuple[int, int]
    number_span: tuple[int, int]


def docket_readings(source: str) -> tuple[DocketReading, ...]:
    """Read only explicitly labelled federal CM/ECF docket identifiers."""
    return tuple(DocketReading(match.span(), match.span("number")) for match in _DOCKET.finditer(source))


def find_docket_locators(document: Document) -> Document:
    """Create courtless docket occurrences from labelled CM/ECF numbers."""
    if STAGE in document.stage_runs:
        raise ValueError(f"Stage already completed: {STAGE}")
    if "colocations" in document.stage_runs:
        raise ValueError("Discover all locators before resolving colocations")
    for reading in docket_readings(document.text):
        span = Span(*reading.span)
        if any(span.overlaps(item.site_span) for item in document.citations):
            continue
        identifier = f"docket:{span.start}:{span.end}"
        document = document.add_citation(
            FullDocketCitation.from_locator(
                citation_id=identifier,
                stage=STAGE,
                source=document.text,
                span=span,
                number_span=Span(*reading.number_span),
            )
        )
    return document.complete(STAGE)
