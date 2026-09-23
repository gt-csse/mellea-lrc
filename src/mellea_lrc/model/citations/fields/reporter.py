"""A full reporter locator and its eyecite-normalized reporter identity."""

from __future__ import annotations

from functools import lru_cache
from typing import Self

from eyecite import get_citations
from eyecite.models import FullCaseCitation, Reporter
from pydantic import BaseModel, ConfigDict, model_validator

from mellea_lrc.model.citations.fields.base import CitationField, source_quote
from mellea_lrc.model.span import Span


class ReporterLocatorValue(BaseModel):
    """Normalized parts of one complete reporter or database locator."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    volume: int
    reporter: Reporter
    edition: str
    page: str

    @model_validator(mode="after")
    def _validate_parts(self) -> Self:
        if self.volume < 1 or not self.edition.strip() or not self.page.strip():
            raise ValueError("Full reporter locator has incomplete normalized parts")
        if not self.reporter.short_name or not self.reporter.name:
            raise ValueError("Reporter has no normalized identity")
        return self


@lru_cache(maxsize=8192)
def normalize_reporter_locator(quote: str) -> ReporterLocatorValue:
    """Re-read one locator in isolation using eyecite's reporter edition."""
    matches = [
        cite
        for cite in get_citations(quote)
        if isinstance(cite, FullCaseCitation) and cite.span() == (0, len(quote))
    ]
    if len(matches) != 1:
        raise ValueError(f"Cannot normalize full reporter locator: {quote!r}")
    match = matches[0]

    # corrected_reporter() falls back to raw text when the edition is unknown;
    # require an actual eyecite edition before recording a normalized value.
    edition = match.edition_guess
    if edition is None:
        raise ValueError(f"Cannot normalize ambiguous reporter locator: {quote!r}")
    volume_text = match.groups.get("volume")
    reporter_text = match.groups.get("reporter")
    page_text = match.groups.get("page")
    page = match.corrected_page()
    if not volume_text or not reporter_text or not page_text or not page:
        raise ValueError(f"Cannot normalize full reporter locator: {quote!r}")
    offset = 0
    for part in (volume_text, reporter_text, page_text):
        position = quote.find(part, offset)
        if position < 0:
            raise ValueError(f"Reporter component does not match locator quote: {quote!r}")
        offset = position + len(part)
    try:
        volume = int(volume_text)
    except ValueError as exc:
        raise ValueError(f"Cannot normalize reporter volume: {volume_text!r}") from exc
    return ReporterLocatorValue(
        volume=volume,
        reporter=edition.reporter,
        edition=edition.short_name,
        page=page,
    )


class FullReporterLocator(CitationField[ReporterLocatorValue]):
    """One exact written locator with a normalized Reporter object."""

    quote: str
    span: Span

    @classmethod
    def from_source(cls, source: str, span: Span, *, node_id: str) -> Self:
        quote = source_quote(source, span)
        return cls(node_id=node_id, quote=quote, span=span, normalized=normalize_reporter_locator(quote))

    @model_validator(mode="after")
    def _validate_normalization(self) -> Self:
        if self.normalized != normalize_reporter_locator(self.quote):
            raise ValueError("Reporter locator normalization does not match its quote")
        return self
