"""Quoted case-name readings and their rule or model normalization."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import model_validator

from mellea_lrc.model.citations.fields.base import CitationField, normalization_record, source_quote
from mellea_lrc.parsing.case_names import CaseName, CaseNameKind
from mellea_lrc.parsing.span import Span


class CaseNameField(CitationField[CaseName]):
    """A quoted name with a rule or model reading of its written form."""

    quote: str
    span: Span
    normalized_by: Literal["rule", "model"] = "rule"

    @classmethod
    def from_source(cls, source: str, span: Span, *, node_id: str) -> Self:
        quote = source_quote(source, span)
        return cls(
            node_id=node_id,
            quote=quote,
            span=span,
            **normalization_record(lambda: CaseName.from_quote(quote)),
        )

    @classmethod
    def from_model(cls, source: str, span: Span, normalized: CaseName, *, node_id: str) -> Self:
        """Keep the exact source span and the model's typed normalization."""
        return cls(
            node_id=node_id,
            quote=source_quote(source, span),
            span=span,
            normalized_by="model",
            normalizable=True,
            normalized=normalized,
            normalization_error=None,
        )

    @model_validator(mode="after")
    def _validate_normalization(self) -> Self:
        if self.normalizable and self.get_normalized().kind is CaseNameKind.NOT_STATED:
            raise ValueError("An unstated case name cannot be a quoted field reading")
        if self.normalized_by == "rule":
            self.validate_normalization(lambda: CaseName.from_quote(self.quote))
        return self
