"""Written citation dates and their validated calendar readings."""

from __future__ import annotations

import re
from datetime import date
from typing import Self

from pydantic import BaseModel, ConfigDict, model_validator

from mellea_lrc.model.citations.fields.base import CitationField, normalization_record, source_quote
from mellea_lrc.model.span import Span

YEAR_RE = re.compile(r"(?<!\d)(?:1[6789]\d{2}|20\d{2}|21\d{2})(?!\d)")
FULL_DATE_RE = re.compile(
    r"\b(?P<month>Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r"\.?\s+(?P<day>\d{1,2}),?\s+(?P<year>(?:1[6789]|20|21)\d{2})\b",
    re.I,
)
_MONTHS = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}


class CitationDate(BaseModel):
    """A decision date, possibly only a year or month."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    year: int
    month: int | None = None
    day: int | None = None

    @model_validator(mode="after")
    def _validate_calendar_date(self) -> Self:
        if self.day is not None and self.month is None:
            raise ValueError("Citation date cannot have a day without a month")
        try:
            date(
                self.year,
                self.month if self.month is not None else 1,
                self.day if self.day is not None else 1,
            )
        except ValueError as exc:
            raise ValueError(f"Invalid citation date: {exc}") from exc
        return self


def normalize_date(quote: str) -> CitationDate:
    """Parse a complete written day or year; never return a null date."""
    if match := FULL_DATE_RE.fullmatch(quote):
        return CitationDate(
            year=int(match.group("year")),
            month=_MONTHS[match.group("month")[:3].lower()],
            day=int(match.group("day")),
        )
    if YEAR_RE.fullmatch(quote):
        return CitationDate(year=int(quote))
    raise ValueError(f"Cannot normalize citation date: {quote!r}")


class DateField(CitationField[CitationDate]):
    """An exact written date whose normalized calendar value is checked."""

    quote: str
    span: Span

    @classmethod
    def from_source(cls, source: str, span: Span, *, node_id: str) -> Self:
        quote = source_quote(source, span)
        return cls(
            node_id=node_id,
            quote=quote,
            span=span,
            **normalization_record(lambda: normalize_date(quote)),
        )

    @model_validator(mode="after")
    def _validate_normalization(self) -> Self:
        self.validate_normalization(lambda: normalize_date(self.quote))
        return self
