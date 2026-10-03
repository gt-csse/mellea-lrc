"""Typed, source-grounded Id and supra references."""

from __future__ import annotations

import re
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, model_validator

from mellea_lrc.model.citations.fields.base import CitationField, normalization_record, source_quote
from mellea_lrc.model.span import Span
from mellea_lrc.parsing.markers import ID_MARKER
from mellea_lrc.parsing.supra import supra_readings


class IdValue(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    keyword: Literal["id", "ibid"]


class SupraValue(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    antecedent: str | None
    volume: int | None


def normalize_id(quote: str) -> IdValue:
    match = re.match(ID_MARKER, quote, re.I)
    if match is None:
        raise ValueError("Id citation has no supported Id. or Ibid. marker")
    keyword = re.match(r"id|ibid", match.group(1), re.I)
    assert keyword is not None
    return IdValue(keyword=keyword.group().lower())


def normalize_supra(quote: str) -> SupraValue:
    matches = supra_readings(quote)
    if len(matches) != 1:
        raise ValueError("Supra quote must contain one supra reference")
    reading = matches[0]
    antecedent = quote[slice(*reading.antecedent_span)]
    return SupraValue(antecedent=" ".join(antecedent.split()), volume=reading.volume)


class IdField(CitationField[IdValue]):
    quote: str
    span: Span

    @classmethod
    def from_source(cls, source: str, span: Span, *, node_id: str) -> Self:
        quote = source_quote(source, span)
        return cls(
            node_id=node_id, quote=quote, span=span, **normalization_record(lambda: normalize_id(quote))
        )

    @model_validator(mode="after")
    def _normalization(self) -> Self:
        self.validate_normalization(lambda: normalize_id(self.quote))
        return self


class SupraField(CitationField[SupraValue]):
    quote: str
    span: Span

    @classmethod
    def from_source(cls, source: str, span: Span, *, node_id: str) -> Self:
        quote = source_quote(source, span)
        return cls(
            node_id=node_id, quote=quote, span=span, **normalization_record(lambda: normalize_supra(quote))
        )

    @model_validator(mode="after")
    def _normalization(self) -> Self:
        self.validate_normalization(lambda: normalize_supra(self.quote))
        return self
