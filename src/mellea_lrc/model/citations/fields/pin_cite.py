"""The typed page or paragraph targets of a pinpoint citation."""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Self, TypeAlias

from pydantic import BaseModel, ConfigDict, model_validator

from mellea_lrc.model.citations.fields.base import CitationField, normalization_record, source_quote
from mellea_lrc.model.span import Span

# PDF text extraction may insert horizontal space on either side of a range
# separator. Share this bounded relaxation with the reader so the exact quote
# it captures is also a quote this normalizer can interpret.
PIN_RANGE_JOIN = r"[^\S\r\n]*[-–][^\S\r\n]*"
_HSPACE = r"[^\S\r\n]"
_NUMBER = rf"(?:\*|¶{{1,2}})?{_HSPACE}*\d+(?:{PIN_RANGE_JOIN}\d+)?"
_NOTE = (
    rf"(?:{_HSPACE}+"
    rf"(?:(?:&|and){_HSPACE}*)?"
    rf"(?:n{{1,2}}\.|fn\.?)"
    rf"{_HSPACE}*\d+(?:{PIN_RANGE_JOIN}\d+)?)?"
)
_ITEM = rf"{_NUMBER}{_NOTE}"
# The extractor uses the same syntax to find a bounded candidate. The
# normalizer below still requires every character of that candidate to parse.
PIN_PREFIX = re.compile(rf"^\s*,?\s*(?:at{_HSPACE}+)?(?P<pin>{_ITEM}(?:,\s*{_ITEM})*)", re.I)
_AT = re.compile(rf"at{_HSPACE}+", re.I)
_PART = re.compile(
    rf"(?P<label>\*|¶{{1,2}})?{_HSPACE}*(?P<first>\d+)"
    rf"(?:{PIN_RANGE_JOIN}(?P<last>\d+))?"
    rf"(?:{_HSPACE}+(?:(?:&|and){_HSPACE}*)?(?:n{{1,2}}\.|fn\.?)"
    rf"{_HSPACE}*(?P<footnote>\d+(?:{PIN_RANGE_JOIN}\d+)?))?\Z",
    re.I,
)


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
    """Normalize complete written targets, or leave the whole quote for review."""
    text = quote.strip()
    if prefix := _AT.match(text):
        text = text[prefix.end() :]
    parts = text.split(",")
    targets: list[PinCiteTarget] = []
    kind = PinCiteKind.PAGE
    for index, part in enumerate(parts):
        match = _PART.fullmatch(part.strip())
        if match is None:
            raise ValueError(f"Cannot normalize pin cite: {quote!r}")
        label = match.group("label")
        if label == "*":
            kind = PinCiteKind.STAR
        elif label in {"¶", "¶¶"}:
            kind = PinCiteKind.PARAGRAPH
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
        footnote = match.group("footnote")
        # A comma after a footnote can list more pages or more footnotes. Do
        # not silently assign those numbers to the wrong kind of target.
        if footnote is not None and index < len(parts) - 1:
            raise ValueError(f"Ambiguous page/footnote list: {quote!r}")
        targets.append(
            PinCiteTarget(
                first=first,
                last=last,
                kind=kind,
                footnote=re.sub(PIN_RANGE_JOIN, "-", footnote) if footnote else None,
            )
        )
    return tuple(targets)


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
