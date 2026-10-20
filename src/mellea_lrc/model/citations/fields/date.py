"""Written citation dates and their validated calendar readings."""

from __future__ import annotations

from datetime import date
from typing import Self

from pydantic import BaseModel, ConfigDict, model_validator

from mellea_lrc.model.citations.fields.base import CitationField, normalization_record, source_quote
from mellea_lrc.parsing.dates import FULL_DATE_RE, MONTH_YEAR_RE, YEAR_RE
from mellea_lrc.parsing.span import Span

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
    """Parse a written day, month, or year; never return a null date."""
    if match := FULL_DATE_RE.fullmatch(quote):
        return CitationDate(
            year=int(match.group("year")),
            month=_MONTHS[match.group("month")[:3].lower()],
            day=int(match.group("day")),
        )
    if match := MONTH_YEAR_RE.fullmatch(quote):
        return CitationDate(
            year=int(match.group("year")),
            month=_MONTHS[match.group("month")[:3].lower()],
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
