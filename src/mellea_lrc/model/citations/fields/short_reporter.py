"""An exact short reporter quote and its normalized reporter identity."""

from __future__ import annotations

from functools import lru_cache
from typing import Self

from eyecite.models import Reporter
from pydantic import BaseModel, ConfigDict, model_validator

from mellea_lrc.model.citations.fields.base import CitationField, normalization_record, source_quote
from mellea_lrc.model.span import Span
from mellea_lrc.parsing.reporters import short_reporter_readings


class ShortReporterLocatorValue(BaseModel):
    """A short reporter's volume and edition. Pinpoints have their own field."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    volume: int
    reporter: Reporter
    edition: str

    @model_validator(mode="after")
    def _validate_parts(self) -> Self:
        if self.volume < 1 or not self.edition.strip():
            raise ValueError("Short reporter locator has incomplete normalized parts")
        if not self.reporter.short_name or not self.reporter.name:
            raise ValueError("Reporter has no normalized identity")
        return self


@lru_cache(maxsize=8192)
def normalize_short_reporter_locator(quote: str) -> ShortReporterLocatorValue:
    """Normalize exactly one whole short reporter locator."""
    matches = [reading for reading in short_reporter_readings(quote) if reading.span == (0, len(quote))]
    if len(matches) != 1:
        raise ValueError(f"Cannot normalize short reporter locator: {quote!r}")
    citation = matches[0].citation
    edition = citation.edition_guess
    if edition is None:
        raise ValueError(f"Cannot normalize ambiguous short reporter locator: {quote!r}")
    volume_text = citation.groups.get("volume")
    reporter_text = citation.groups.get("reporter")
    if not volume_text or not reporter_text:
        raise ValueError(f"Cannot normalize short reporter locator: {quote!r}")
    reporter_start = quote.find(reporter_text, len(volume_text))
    if not quote.startswith(volume_text) or reporter_start < 0:
        raise ValueError(f"Reporter components do not match short locator quote: {quote!r}")
    return ShortReporterLocatorValue(
        volume=int(volume_text),
        reporter=edition.reporter,
        edition=edition.short_name,
    )


class ShortReporterLocator(CitationField[ShortReporterLocatorValue]):
    """One written short locator with exact offsets in its source."""

    quote: str
    span: Span

    @classmethod
    def from_source(cls, source: str, span: Span, *, node_id: str) -> Self:
        quote = source_quote(source, span)
        return cls(
            node_id=node_id,
            quote=quote,
            span=span,
            **normalization_record(lambda: normalize_short_reporter_locator(quote)),
        )

    @model_validator(mode="after")
    def _validate_normalization(self) -> Self:
        self.validate_normalization(lambda: normalize_short_reporter_locator(self.quote))
        return self
