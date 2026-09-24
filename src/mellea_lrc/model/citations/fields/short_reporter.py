"""A short reporter citation and its eyecite-normalized reporter and pin page."""

from __future__ import annotations

from functools import lru_cache
from typing import Self

from eyecite.models import Reporter
from pydantic import BaseModel, ConfigDict, model_validator

from mellea_lrc.model.citations.fields.base import CitationField, normalization_record, source_quote
from mellea_lrc.model.span import Span
from mellea_lrc.reporter_reading import short_reporter_readings


class ShortReporterLocatorValue(BaseModel):
    """Reporter identity and the first pinpoint page in a short citation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    volume: int
    reporter: Reporter
    edition: str
    pin_page: str

    @model_validator(mode="after")
    def _validate_parts(self) -> Self:
        if self.volume < 1 or not self.edition.strip() or not self.pin_page.strip():
            raise ValueError("Short reporter locator has incomplete normalized parts")
        if not self.reporter.short_name or not self.reporter.name:
            raise ValueError("Reporter has no normalized identity")
        return self


@lru_cache(maxsize=8192)
def normalize_short_reporter_locator(quote: str) -> ShortReporterLocatorValue:
    """Read one complete short citation through the shared eyecite service."""
    matches = [reading for reading in short_reporter_readings(quote) if reading.span == (0, len(quote))]
    if len(matches) != 1:
        raise ValueError(f"Cannot normalize short reporter locator: {quote!r}")
    match = matches[0].citation

    edition = match.edition_guess
    if edition is None:
        raise ValueError(f"Cannot normalize ambiguous short reporter locator: {quote!r}")
    volume_text = match.groups.get("volume")
    reporter_text = match.groups.get("reporter")
    page_text = match.groups.get("page")
    pin_page = match.corrected_page()
    if not volume_text or not reporter_text or not page_text or not pin_page:
        raise ValueError(f"Cannot normalize short reporter locator: {quote!r}")
    offset = 0
    for part in (volume_text, reporter_text, page_text):
        position = quote.find(part, offset)
        if position < 0:
            raise ValueError(f"Reporter component does not match short locator quote: {quote!r}")
        offset = position + len(part)
    try:
        volume = int(volume_text)
    except ValueError as exc:
        raise ValueError(f"Cannot normalize reporter volume: {volume_text!r}") from exc
    return ShortReporterLocatorValue(
        volume=volume,
        reporter=edition.reporter,
        edition=edition.short_name,
        pin_page=pin_page,
    )


class ShortReporterLocator(CitationField[ShortReporterLocatorValue]):
    """One exact short reporter quote with its normalized reporter identity."""

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
