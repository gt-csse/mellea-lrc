"""Canonical citation representations shared across extraction and validation.

These are project-level citation classes. Eyecite citations are converted into
these canonical types before downstream validation and serialization.
"""

from dataclasses import dataclass
from enum import Enum
from typing import ClassVar, TypeAlias


class CitationKind(str, Enum):
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
class CitationDate:
    """The decision date a citation states, to whatever precision it states it.

    A filing writes `(2007)` for a reported case and `(D. Ariz. Oct. 31, 2024)`
    for an unpublished one, and the difference carries information: 58 of the
    583 case citations on `false-citation-bench` give a full date, and they are
    disproportionately the Westlaw and LEXIS citations, which are exactly the
    ones a year alone cannot tell apart.

    So the field is a date rather than a year. `year` is always present -- a
    date without one identifies nothing -- and `month` and `day` come together
    or not at all, which is what the corpus shows: no citation there states a
    month without a day.

    Values are kept as the citation wrote them. Comparing them to a retrieved
    opinion is a separate step that has to be testable on its own.
    """

    year: str
    month: str | None = None
    day: str | None = None

    @property
    def is_exact(self) -> bool:
        """Whether this names a single day rather than a year."""
        return self.month is not None and self.day is not None


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
    date: CitationDate | None = None
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
    date: CitationDate | None = None
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
    date: CitationDate | None = None
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
