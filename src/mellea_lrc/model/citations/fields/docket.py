"""Source-grounded docket locator and optional entry reference."""

from __future__ import annotations

import re
from typing import Self

from pydantic import BaseModel, ConfigDict, model_validator

from mellea_lrc.model.citations.fields.base import CitationField, normalization_record, source_quote
from mellea_lrc.model.span import Span

DOCKET_ENTRY_PATTERN = re.compile(
    r"\b(?:Doc(?:ument)?\.?|Dkt\.?|ECF|D\.I\.)\s*(?:No\.?\s*)?(?P<number>\d+(?:-\d+)?)(?![A-Za-z0-9/-]|\.\d)",
    re.I,
)


def normalize_docket_entry(quote: str) -> str:
    match = DOCKET_ENTRY_PATTERN.fullmatch(quote)
    if match is None:
        raise ValueError(f"Cannot normalize docket entry: {quote!r}")
    return match.group("number")


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
            **normalization_record(lambda: DocketLocatorValue(docket_number=number)),
        )

    @model_validator(mode="after")
    def _validate_normalization(self) -> Self:
        if not (self.span.start <= self.number_span.start < self.number_span.end <= self.span.end):
            raise ValueError("Docket number span must be inside the locator")
        start = self.number_span.start - self.span.start
        end = self.number_span.end - self.span.start
        self.validate_normalization(lambda: DocketLocatorValue(docket_number=self.quote[start:end]))
        return self


class DocketEntryField(CitationField[str]):
    """A labelled docket entry with a parsed entry number."""

    quote: str
    span: Span

    @classmethod
    def from_source(cls, source: str, span: Span, *, node_id: str) -> Self:
        quote = source_quote(source, span)
        return cls(
            node_id=node_id,
            quote=quote,
            span=span,
            **normalization_record(lambda: normalize_docket_entry(quote)),
        )

    @model_validator(mode="after")
    def _validate_normalization(self) -> Self:
        self.validate_normalization(lambda: normalize_docket_entry(self.quote))
        return self
