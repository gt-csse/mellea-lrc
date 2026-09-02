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
    DOCKET = "DocketCitation"
    SHORT_CASE = "ShortCaseCitation"
    SUPRA = "SupraCitation"
    ID = "IdCitation"
    REFERENCE = "ReferenceCitation"
    UNKNOWN = "UnknownCitation"


# Full citations identify what they cite on their own; short citations
# generally need an antecedent before they can be validated. A docket citation
# belongs here on that test -- a docket number and its court name a case with
# no help from the text around them -- even though the reporter-keyed case
# search cannot look one up, which is a fact about that service and not about
# the citation.
FULL_CITATION_KINDS = frozenset(
    {
        CitationKind.FULL_CASE,
        CitationKind.FULL_LAW,
        CitationKind.FULL_JOURNAL,
        CitationKind.DOCKET,
    }
)


@dataclass(frozen=True, slots=True)
class Reporter:
    """The reporter a citation names, as written and as the databases know it.

    A filing writes one reporter several ways and extraction adds more. Across
    the 26 documents of `false-citation-bench` there are **47 distinct reporter
    spellings for 35 reporters**: `F.Supp.2d`, `F. Supp. 2d` and `F.  Supp.  2d`
    are one thing, so are `N.C.App.` and `N.C. App.`, and so are `Fed. Appx.`,
    `Fed. App'x` and `F. App'x` -- the last three being the filer's choice
    rather than converter damage, which no amount of whitespace repair would
    reconcile.

    So the field is a reporter rather than a string. `as_written` is what the
    document says, kept because this project records what was written; the rest
    is what reporters-db knows about it, by way of the edition eyecite matched.

    `cite_type` is worth having beyond tidiness: it says `federal`, `state`,
    `journal` or `leg_statute`, which is a sourced answer to "is this a case at
    all" in place of the reporter-name guessing done elsewhere.

    `short_name` is None when no edition matched -- an unknown reporter is
    recorded as written and not invented.
    """

    as_written: str
    short_name: str | None = None
    name: str | None = None
    cite_type: str | None = None
    is_scotus: bool = False

    @property
    def canonical(self) -> str:
        """The spelling to compare on: the database's, or ours if it has none."""
        return self.short_name or self.as_written

    def __str__(self) -> str:
        """The reporter as the document wrote it."""
        return self.as_written


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

    def __str__(self) -> str:
        """The date roughly as a citation writes it."""
        if self.is_exact:
            return f"{self.month} {self.day}, {self.year}"
        return self.year


@dataclass(frozen=True, slots=True)
class FullCaseCitation:
    """Complete citation to a reported case."""

    kind: ClassVar[CitationKind] = CitationKind.FULL_CASE

    plaintiff: str | None = None
    defendant: str | None = None
    volume: str | None = None
    reporter: Reporter | None = None
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
    reporter: Reporter | None = None
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
    reporter: Reporter | None = None
    page: str | None = None
    pin_cite: str | None = None
    date: CitationDate | None = None
    parenthetical: str | None = None


@dataclass(frozen=True, slots=True)
class DocketCitation:
    """A case identified by its docket number rather than by a reporter page.

    The court is not decoration here, it is half of the identifier: the same
    docket number exists in every district, and only the pair names a case. So
    both are on the citation, and a docket read without a court is not read at
    all.

    ``docket_number`` is kept as the filing wrote it, damage included --
    ``1:25cv-05745-RPK`` is a real citation in false-citation-bench, missing the
    hyphen its converter dropped. Normalizing it here would hide from a reader
    what the document actually says, which is the one thing a verification tool
    must not do.
    """

    kind: ClassVar[CitationKind] = CitationKind.DOCKET

    plaintiff: str | None = None
    defendant: str | None = None
    docket_number: str | None = None
    court: str | None = None
    """The courts-db identifier, e.g. ``nyed``."""
    court_name: str | None = None
    court_text: str | None = None
    """The court exactly as the filing wrote it, e.g. ``E.D.N.Y.``."""
    pin_cite: str | None = None
    date: CitationDate | None = None
    parenthetical: str | None = None


@dataclass(frozen=True, slots=True)
class ShortCaseCitation:
    """Subsequent reference using volume + reporter + pin cite."""

    kind: ClassVar[CitationKind] = CitationKind.SHORT_CASE

    volume: str | None = None
    reporter: Reporter | None = None
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
    | DocketCitation
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
