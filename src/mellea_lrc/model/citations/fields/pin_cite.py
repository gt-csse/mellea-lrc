"""The typed page or paragraph targets of a pinpoint citation."""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Self, TypeAlias

from pydantic import BaseModel, ConfigDict, model_validator

from mellea_lrc.model.citations.fields.base import CitationField, normalization_record, source_quote
from mellea_lrc.model.span import Span

_PAGE_PIN = re.compile(r"(?P<star>\*)?(?P<first>\d+)(?:[-–](?P<last>\d+))?\Z")


class PinCiteKind(StrEnum):
    PAGE = "page"
    STAR = "star"
    PARAGRAPH = "paragraph"


class PinCiteTarget(BaseModel):
    """One inclusive pinpoint range, independent of its written notation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    first: int
    last: int
    kind: PinCiteKind
    footnote: str | None = None

    @model_validator(mode="after")
    def _valid_range(self) -> PinCiteTarget:
        if self.first < 1 or self.last < self.first:
            raise ValueError("Pin cite pages must form a positive, ascending range")
        if self.footnote == "":
            raise ValueError("Pin cite footnote cannot be empty")
        return self


PinCiteValue: TypeAlias = tuple[PinCiteTarget, ...]


def normalize_pin_cite(quote: str) -> PinCiteValue:
    """Normalize a written page/star-page pin or raise on failure."""
    match = _PAGE_PIN.fullmatch(quote)
    if match is None:
        raise ValueError(f"Cannot normalize pin cite: {quote!r}")
    first_text = match.group("first")
    last_text = match.group("last")
    first = int(first_text)
    last = first
    if last_text is not None:
        last = int(last_text)
        if len(last_text) < len(first_text):
            place = 10 ** len(last_text)
            last += first // place * place
            if last < first:
                last += place
    return (
        PinCiteTarget(
            first=first,
            last=last,
            kind=PinCiteKind.STAR if match.group("star") else PinCiteKind.PAGE,
        ),
    )


class PinCiteField(CitationField[PinCiteValue]):
    """An exact pinpoint quote with validated normalized targets."""

    quote: str
    span: Span

    @classmethod
    def from_source(cls, source: str, span: Span, *, node_id: str) -> Self:
        quote = source_quote(source, span)
        return cls(
            node_id=node_id,
            quote=quote,
            span=span,
            **normalization_record(lambda: normalize_pin_cite(quote)),
        )

    @model_validator(mode="after")
    def _validate_normalization(self) -> Self:
        self.validate_normalization(lambda: normalize_pin_cite(self.quote))
        return self
