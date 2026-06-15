"""Canonical citation representations shared across extraction and validation."""

from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar, TypeAlias


class CitationKind(StrEnum):
    """Canonical citation type names used in annotation and serialization."""

    FULL_CASE = "FullCaseCitation"
    FULL_LAW = "FullLawCitation"
    FULL_JOURNAL = "FullJournalCitation"
    SHORT_CASE = "ShortCaseCitation"
    SUPRA = "SupraCitation"
    ID = "IdCitation"
    REFERENCE = "ReferenceCitation"
    UNKNOWN = "UnknownCitation"


FULL_CITATION_KINDS = frozenset(
    {
        CitationKind.FULL_CASE,
        CitationKind.FULL_LAW,
        CitationKind.FULL_JOURNAL,
    }
)


class _CitationKindMixin:
    """Shared accessor for canonical citation type names."""

    _KIND: ClassVar[CitationKind]

    @property
    def kind(self) -> CitationKind:
        """Return the canonical citation type."""
        return self._KIND


@dataclass(frozen=True, slots=True)
class FullCaseCitation(_CitationKindMixin):
    """Complete citation to a reported case."""

    _KIND: ClassVar[CitationKind] = CitationKind.FULL_CASE

    plaintiff: str | None = None
    defendant: str | None = None
    volume: str | None = None
    reporter: str | None = None
    page: str | None = None
    pin_cite: str | None = None
    extra: str | None = None
    year: str | None = None
    court: str | None = None
    parenthetical: str | None = None


@dataclass(frozen=True, slots=True)
class FullLawCitation(_CitationKindMixin):
    """Citation to a statute, regulation, or code section."""

    _KIND: ClassVar[CitationKind] = CitationKind.FULL_LAW

    volume: str | None = None
    reporter: str | None = None
    page: str | None = None
    pin_cite: str | None = None
    year: str | None = None
    publisher: str | None = None
    parenthetical: str | None = None


@dataclass(frozen=True, slots=True)
class FullJournalCitation(_CitationKindMixin):
    """Citation to a law review or journal article."""

    _KIND: ClassVar[CitationKind] = CitationKind.FULL_JOURNAL

    volume: str | None = None
    reporter: str | None = None
    page: str | None = None
    pin_cite: str | None = None
    year: str | None = None
    parenthetical: str | None = None


@dataclass(frozen=True, slots=True)
class ShortCaseCitation(_CitationKindMixin):
    """Subsequent reference using volume + reporter + pin cite."""

    _KIND: ClassVar[CitationKind] = CitationKind.SHORT_CASE

    volume: str | None = None
    reporter: str | None = None
    page: str | None = None
    pin_cite: str | None = None
    court: str | None = None
    parenthetical: str | None = None


@dataclass(frozen=True, slots=True)
class SupraCitation(_CitationKindMixin):
    """Reference using party name + supra."""

    _KIND: ClassVar[CitationKind] = CitationKind.SUPRA

    pin_cite: str | None = None
    parenthetical: str | None = None


@dataclass(frozen=True, slots=True)
class IdCitation(_CitationKindMixin):
    """Reference using Id. or Ibid."""

    _KIND: ClassVar[CitationKind] = CitationKind.ID

    pin_cite: str | None = None
    parenthetical: str | None = None


@dataclass(frozen=True, slots=True)
class ReferenceCitation(_CitationKindMixin):
    """Bare party-name reference with no reporter information."""

    _KIND: ClassVar[CitationKind] = CitationKind.REFERENCE

    plaintiff: str | None = None
    defendant: str | None = None


@dataclass(frozen=True, slots=True)
class UnknownCitation(_CitationKindMixin):
    """Span that looks like a citation but cannot be parsed."""

    _KIND: ClassVar[CitationKind] = CitationKind.UNKNOWN


CanonicalCitation: TypeAlias = (
    FullCaseCitation
    | FullLawCitation
    | FullJournalCitation
    | ShortCaseCitation
    | SupraCitation
    | IdCitation
    | ReferenceCitation
    | UnknownCitation
)


def citation_kind(citation: CanonicalCitation) -> CitationKind:
    """Return the canonical type name for a citation."""
    return citation.kind


def is_full_citation(citation: CanonicalCitation) -> bool:
    """Return True when the citation is a self-contained bibliographic cite."""
    return citation.kind in FULL_CITATION_KINDS
