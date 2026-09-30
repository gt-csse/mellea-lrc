"""Typed, source-grounded Id and supra references."""

from __future__ import annotations

import re
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, model_validator

from mellea_lrc.model.citations.fields.base import CitationField, normalization_record, source_quote
from mellea_lrc.model.span import Span


class IdValue(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    keyword: Literal["id", "ibid"]


class SupraValue(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    antecedent: str | None
    volume: int | None


def normalize_id(quote: str) -> IdValue:
    match = re.match(r"(?P<keyword>id|ibid)\s*\.", quote, re.I)
    if match is None:
        raise ValueError("Id citation has no Id. or Ibid. marker")
    return IdValue(keyword=match.group("keyword").lower())


def normalize_supra(quote: str) -> SupraValue:
    from eyecite import get_citations
    from eyecite.models import SupraCitation

    matches = [c for c in get_citations(quote) if isinstance(c, SupraCitation)]
    if len(matches) != 1:
        raise ValueError("Supra quote must contain one supra reference")
    metadata = matches[0].metadata
    return SupraValue(antecedent=metadata.antecedent_guess, volume=metadata.volume)


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
