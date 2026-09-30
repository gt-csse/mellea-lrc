"""Structured, validated readings of written case names."""

from __future__ import annotations

import re
from enum import Enum
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, model_validator

from mellea_lrc.model.citations.fields.base import (
    CitationField,
    normalization_record,
    require_all_json_properties,
    source_quote,
)
from mellea_lrc.model.span import Span

_VERSUS = re.compile(r"\s+[vV](?:\s*\.\s*|\s+)")
_IN_RE = re.compile(r"In re\s+(.+)", re.IGNORECASE)
_EX_PARTE = re.compile(r"Ex parte\s+(.+)", re.IGNORECASE)
_MATTER_OF = re.compile(r"Matter of\s+(.+)", re.IGNORECASE)


class CaseNameKind(str, Enum):
    """A written name's form, or the absence of a written name."""

    ADVERSARIAL = "adversarial"
    IN_RE = "in_re"
    EX_PARTE = "ex_parte"
    PARTIAL = "partial"
    NOT_STATED = "not_stated"


class CaseName(BaseModel):
    """The structured form of a complete or partial written case name."""

    model_config = ConfigDict(frozen=True, extra="forbid", json_schema_extra=require_all_json_properties)

    kind: CaseNameKind
    plaintiff: str | None = None
    defendant: str | None = None
    subject: str | None = None
    partial: str | None = None

    @classmethod
    def from_quote(cls, quote: str) -> Self:
        """Read a written name, preserving punctuation while folding whitespace."""
        name = " ".join(quote.split())
        if not name:
            raise ValueError("Case name is empty")

        if match := _IN_RE.fullmatch(name):
            return cls(kind=CaseNameKind.IN_RE, subject=match.group(1))
        if match := _MATTER_OF.fullmatch(name):
            return cls(kind=CaseNameKind.IN_RE, subject=match.group(1))
        if match := _EX_PARTE.fullmatch(name):
            return cls(kind=CaseNameKind.EX_PARTE, subject=match.group(1))

        separators = list(_VERSUS.finditer(name))
        if len(separators) == 1:
            separator = separators[0]
            return cls(
                kind=CaseNameKind.ADVERSARIAL,
                plaintiff=name[: separator.start()],
                defendant=name[separator.end() :],
            )
        if (
            separators
            or re.match(r"^[vV]\s*\.", name)
            or re.search(r"\s+[vV](?:\s*\.)?$", name)
            or re.search(r"\s+[vV][sS]\.?(?:\s+|$)", name)
        ):
            raise ValueError("Adversarial case name has an incomplete or ambiguous separator")
        if re.fullmatch(r"(?:In re|Ex parte|Matter of)", name, re.IGNORECASE):
            raise ValueError("Procedural case name has no subject")
        if not any(character.isalnum() for character in name):
            raise ValueError("Case name fragment does not contain a name")
        return cls(kind=CaseNameKind.PARTIAL, partial=name)

    @model_validator(mode="after")
    def _validate_form(self) -> Self:
        for part in (self.plaintiff, self.defendant, self.subject, self.partial):
            if part is not None and (not part or part != " ".join(part.split())):
                raise ValueError("Case name parts must be nonempty and use normalized whitespace")
            if part is not None and not any(character.isalnum() for character in part):
                raise ValueError("Case name parts must contain a name")

        if self.kind is CaseNameKind.ADVERSARIAL:
            if (
                self.plaintiff is None
                or self.defendant is None
                or self.subject is not None
                or self.partial is not None
            ):
                raise ValueError("Adversarial case name requires plaintiff and defendant only")
            if _VERSUS.search(self.plaintiff) or _VERSUS.search(self.defendant):
                raise ValueError("Adversarial case name has an ambiguous 'v.' separator")
        elif self.kind in {CaseNameKind.IN_RE, CaseNameKind.EX_PARTE}:
            if (
                self.subject is None
                or self.plaintiff is not None
                or self.defendant is not None
                or self.partial is not None
            ):
                raise ValueError("Subject case name requires a subject only")
        elif self.kind is CaseNameKind.PARTIAL:
            if (
                self.partial is None
                or self.plaintiff is not None
                or self.defendant is not None
                or self.subject is not None
            ):
                raise ValueError("Partial case name requires only its printed fragment")
            if (
                (
                    (separator := _VERSUS.search(self.partial)) is not None
                    and any(character.isalnum() for character in self.partial[: separator.start()])
                    and any(character.isalnum() for character in self.partial[separator.end() :])
                )
                or _IN_RE.fullmatch(self.partial)
                or _MATTER_OF.fullmatch(self.partial)
                or _EX_PARTE.fullmatch(self.partial)
            ):
                raise ValueError("A complete case name cannot use the partial form")
        elif self.kind is CaseNameKind.NOT_STATED:
            if any(part is not None for part in (self.plaintiff, self.defendant, self.subject, self.partial)):
                raise ValueError("An unstated case name has no printed parts")
        else:
            raise ValueError(f"Unknown case name kind: {self.kind}")
        return self

    def as_citation(self) -> str:
        """Render the normalized case name in citation form."""
        if self.kind is CaseNameKind.ADVERSARIAL:
            return f"{self.plaintiff} v. {self.defendant}"
        if self.kind is CaseNameKind.PARTIAL:
            if self.partial is None:
                raise ValueError("Partial case name has no printed fragment")
            return self.partial
        if self.kind is CaseNameKind.NOT_STATED:
            raise ValueError("An unstated case name has no citation text")
        if self.kind is CaseNameKind.IN_RE:
            return f"In re {self.subject}"
        if self.kind is CaseNameKind.EX_PARTE:
            return f"Ex parte {self.subject}"
        raise ValueError(f"Unknown case name kind: {self.kind}")


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
