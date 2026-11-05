"""Source-only reader for explicitly labelled federal CM/ECF identifiers."""

from __future__ import annotations

import re
from dataclasses import dataclass

from mellea_lrc.matching.literal_to_regex import fuzzy_literal

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
