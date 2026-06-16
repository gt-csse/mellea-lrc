"""Canonical citation representations shared across extraction and validation.

These are project-level citation classes. Eyecite citations are converted into
these canonical types before downstream validation and serialization.
"""

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


# Full citations are self-contained enough for validation against case search;
# short citations generally need an antecedent before they can be validated.
FULL_CITATION_KINDS = frozenset(
    {
        CitationKind.FULL_CASE,
        CitationKind.FULL_LAW,
        CitationKind.FULL_JOURNAL,
    }
)


@dataclass(frozen=True, slots=True)
class FullCaseCitation:
    """Complete citation to a reported case."""

    kind: ClassVar[CitationKind] = CitationKind.FULL_CASE

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
class FullLawCitation:
    """Citation to a statute, regulation, or code section."""

    kind: ClassVar[CitationKind] = CitationKind.FULL_LAW

    volume: str | None = None
    reporter: str | None = None
    page: str | None = None
    pin_cite: str | None = None
    year: str | None = None
    publisher: str | None = None
    parenthetical: str | None = None


@dataclass(frozen=True, slots=True)
class FullJournalCitation:
    """Citation to a law review or journal article."""

    kind: ClassVar[CitationKind] = CitationKind.FULL_JOURNAL

    volume: str | None = None
    reporter: str | None = None
    page: str | None = None
    pin_cite: str | None = None
    year: str | None = None
    parenthetical: str | None = None


@dataclass(frozen=True, slots=True)
class ShortCaseCitation:
    """Subsequent reference using volume + reporter + pin cite."""

    kind: ClassVar[CitationKind] = CitationKind.SHORT_CASE

    volume: str | None = None
    reporter: str | None = None
    page: str | None = None
    pin_cite: str | None = None
    court: str | None = None
    parenthetical: str | None = None


@dataclass(frozen=True, slots=True)
class SupraCitation:
    """Reference using party name + supra."""

    kind: ClassVar[CitationKind] = CitationKind.SUPRA

    pin_cite: str | None = None
    parenthetical: str | None = None


@dataclass(frozen=True, slots=True)
class IdCitation:
    """Reference using Id. or Ibid."""

    kind: ClassVar[CitationKind] = CitationKind.ID

    pin_cite: str | None = None
    parenthetical: str | None = None


@dataclass(frozen=True, slots=True)
class ReferenceCitation:
    """Bare party-name reference with no reporter information."""

    kind: ClassVar[CitationKind] = CitationKind.REFERENCE

    plaintiff: str | None = None
    defendant: str | None = None


@dataclass(frozen=True, slots=True)
class UnknownCitation:
    """Span that looks like a citation but cannot be parsed."""

    kind: ClassVar[CitationKind] = CitationKind.UNKNOWN


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
