"""Typed, source-grounded readings of citation fields."""

from __future__ import annotations

from typing import Generic, Self, TypeVar

from pydantic import BaseModel, ConfigDict, model_validator

from mellea_lrc.model.citations.date import CitationDate
from mellea_lrc.model.span import Span

T = TypeVar("T")


class CitationField(BaseModel, Generic[T]):
    """One immutable quote and its normalized interpretation at a decision node."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    node_id: str
    normalized: T
    quote: str | None = None
    span: Span | None = None

    @classmethod
    def from_source(cls, source: str, span: Span, *, normalized: T, node_id: str) -> Self:
        """Quote exact source text and store the producer's interpretation."""
        if span.start == span.end or span.end > len(source):
            raise ValueError("Field span is outside the source or empty")
        return cls(node_id=node_id, quote=source[span.start : span.end], span=span, normalized=normalized)

    @model_validator(mode="after")
    def _validate_quote_pair(self) -> Self:
        if (self.quote is None) != (self.span is None):
            raise ValueError("Quote and span must be present together")
        if self.quote == "":
            raise ValueError("A quoted field cannot be empty")
        return self

    def validate_source(self, source: str) -> None:
        """Check that a stored quote still matches the preprocessed text."""
        if self.span is not None and (
            self.span.start == self.span.end
            or self.span.end > len(source)
            or source[self.span.start : self.span.end] != self.quote
        ):
            raise ValueError("Field quote does not match its source span")


class QuotedField(CitationField[T], Generic[T]):
    """A reading that must point to written text."""

    @model_validator(mode="after")
    def _require_quote(self) -> Self:
        if self.quote is None:
            raise ValueError("Written field requires a quote and span")
        return self


class LocatorField(QuotedField[str]):
    """The full written locator, before its components are read."""


class VolumeField(QuotedField[str]):
    """A reporter's written volume."""


class ReporterField(QuotedField[str]):
    """A reporter abbreviation or database label."""


class PageField(QuotedField[str]):
    """A reporter's written first page or database identifier."""


class DocketNumberField(QuotedField[str]):
    """An opaque docket number; equivalence is not normalization."""


class DocketEntryField(QuotedField[str]):
    """A labelled docket entry whose normalized value is its entry number."""


class CaseNameField(QuotedField[str]):
    """A written case name and its current programmatic reading."""


class CourtField(CitationField[str]):
    """A normalized court ID, written or inferred from a reporter."""

    @classmethod
    def inferred(cls, court_id: str, *, node_id: str) -> Self:
        """Record a court derived without a separately written court label."""
        return cls(node_id=node_id, normalized=court_id)


class DateField(QuotedField[CitationDate]):
    """A written decision date and its parsed calendar components."""


class PinCiteField(QuotedField[str]):
    """A written pinpoint reference and its parsed value."""
