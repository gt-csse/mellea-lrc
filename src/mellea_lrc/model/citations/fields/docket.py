"""Source-grounded docket locator and optional entry reference."""

from __future__ import annotations

import re
from typing import Self

from pydantic import BaseModel, ConfigDict, model_validator

from mellea_lrc.model.citations.fields.base import CitationField, source_quote
from mellea_lrc.model.span import Span

_ENTRY_NUMBER = re.compile(
    r"(?:Doc(?:ument)?\.?|Dkt\.?|ECF)\s*(?:No\.?\s*)?(?P<number>\d+(?:-\d+)?)",
    re.I,
)


class DocketLocatorValue(BaseModel):
    """An opaque docket number; equivalence belongs to identity review."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    docket_number: str

    @model_validator(mode="after")
    def _require_number(self) -> Self:
        if not self.docket_number.strip():
            raise ValueError("Docket locator has no normalized number")
        return self


class FullDocketLocator(CitationField[DocketLocatorValue]):
    """The prefixed written docket locator and its opaque number."""

    quote: str
    span: Span
    number_span: Span

    @classmethod
    def from_source(cls, source: str, span: Span, number_span: Span, *, node_id: str) -> Self:
        quote = source_quote(source, span)
        if not (span.start <= number_span.start < number_span.end <= span.end):
            raise ValueError("Docket number span must be inside the locator")
        number = source_quote(source, number_span)
        return cls(
            node_id=node_id,
            quote=quote,
            span=span,
            number_span=number_span,
            normalized=DocketLocatorValue(docket_number=number),
        )

    @model_validator(mode="after")
    def _validate_normalization(self) -> Self:
        if not (self.span.start <= self.number_span.start < self.number_span.end <= self.span.end):
            raise ValueError("Docket number span must be inside the locator")
        start = self.number_span.start - self.span.start
        end = self.number_span.end - self.span.start
        if self.quote[start:end] != self.normalized.docket_number:
            raise ValueError("Docket number normalization does not match its locator quote")
        return self


class DocketEntryField(CitationField[str]):
    """A labelled docket entry with a parsed entry number."""

    quote: str
    span: Span

    @classmethod
    def from_source(cls, source: str, span: Span, *, node_id: str) -> Self:
        quote = source_quote(source, span)
        match = _ENTRY_NUMBER.fullmatch(quote)
        if match is None:
            raise ValueError(f"Cannot normalize docket entry: {quote!r}")
        return cls(node_id=node_id, quote=quote, span=span, normalized=match.group("number"))

    @model_validator(mode="after")
    def _validate_normalization(self) -> Self:
        match = _ENTRY_NUMBER.fullmatch(self.quote)
        if match is None or self.normalized != match.group("number"):
            raise ValueError("Docket entry normalization does not match its quote")
        return self
