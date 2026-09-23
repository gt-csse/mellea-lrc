"""Structured, validated readings of written case names."""

from __future__ import annotations

import re
from enum import Enum
from typing import Self

from pydantic import BaseModel, ConfigDict, model_validator

from mellea_lrc.model.citations.fields.base import CitationField, source_quote
from mellea_lrc.model.span import Span

_VERSUS = re.compile(r"\s+v\.\s+")
_IN_RE = re.compile(r"In re\s+(.+)", re.IGNORECASE)
_EX_PARTE = re.compile(r"Ex parte\s+(.+)", re.IGNORECASE)


class CaseNameKind(str, Enum):
    """The grammatical form of a case name."""

    ADVERSARIAL = "adversarial"
    IN_RE = "in_re"
    EX_PARTE = "ex_parte"


class CaseName(BaseModel):
    """The parties or subject represented by a written case name."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: CaseNameKind
    plaintiff: str | None = None
    defendant: str | None = None
    subject: str | None = None

    @classmethod
    def from_quote(cls, quote: str) -> Self:
        """Read a written name, preserving punctuation while folding whitespace."""
        name = " ".join(quote.split())
        if not name:
            raise ValueError("Case name is empty")

        if match := _IN_RE.fullmatch(name):
            return cls(kind=CaseNameKind.IN_RE, subject=match.group(1))
        if match := _EX_PARTE.fullmatch(name):
            return cls(kind=CaseNameKind.EX_PARTE, subject=match.group(1))

        separators = list(_VERSUS.finditer(name))
        if len(separators) != 1:
            raise ValueError("Adversarial case name requires exactly one 'v.' separator")
        separator = separators[0]
        return cls(
            kind=CaseNameKind.ADVERSARIAL,
            plaintiff=name[: separator.start()],
            defendant=name[separator.end() :],
        )

    @model_validator(mode="after")
    def _validate_form(self) -> Self:
        for part in (self.plaintiff, self.defendant, self.subject):
            if part is not None and (not part or part != " ".join(part.split())):
                raise ValueError("Case name parts must be nonempty and use normalized whitespace")
            if part is not None and not any(character.isalnum() for character in part):
                raise ValueError("Case name parts must contain a name")

        if self.kind is CaseNameKind.ADVERSARIAL:
            if self.plaintiff is None or self.defendant is None or self.subject is not None:
                raise ValueError("Adversarial case name requires plaintiff and defendant only")
            if _VERSUS.search(self.plaintiff) or _VERSUS.search(self.defendant):
                raise ValueError("Adversarial case name has an ambiguous 'v.' separator")
        elif self.subject is None or self.plaintiff is not None or self.defendant is not None:
            raise ValueError("Subject case name requires a subject only")
        return self

    def as_citation(self) -> str:
        """Render the normalized case name in citation form."""
        if self.kind is CaseNameKind.ADVERSARIAL:
            return f"{self.plaintiff} v. {self.defendant}"
        prefix = "In re" if self.kind is CaseNameKind.IN_RE else "Ex parte"
        return f"{prefix} {self.subject}"


class CaseNameField(CitationField[CaseName]):
    """A quoted name whose normalized parties must match that quote."""

    quote: str
    span: Span

    @classmethod
    def from_source(cls, source: str, span: Span, *, node_id: str) -> Self:
        quote = source_quote(source, span)
        return cls(node_id=node_id, quote=quote, span=span, normalized=CaseName.from_quote(quote))

    @model_validator(mode="after")
    def _validate_normalization(self) -> Self:
        if self.normalized != CaseName.from_quote(self.quote):
            raise ValueError("Case name normalization does not match its quote")
        return self
