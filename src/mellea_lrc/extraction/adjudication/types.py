"""What a candidate is, and what a reader's answer about one is.

The layer's unit is a **candidate**: a span of the document that looks like a
citation, or like a citation defect, and that the deterministic pass did not
record. A candidate is a proposal. It enters the record only after a reader has
accepted it, and the record then says which generator proposed it and that a
reader agreed -- otherwise a recovered citation is indistinguishable from a
parsed one, and the next person measuring extraction is measuring the reviewer.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from mellea_lrc.core.spans import Span


class CandidateKind(str, Enum):
    """What a generator thinks it has found."""

    LOCATOR = "locator"
    """A volume, reporter and page the extractor did not read."""

    DOCKET = "docket"
    """A docket number naming a case."""

    EDITION = "edition"
    """A reporter abbreviation naming more than one reporter.

    Not a missing citation. The citation was read; which reporter it names is
    what nobody can settle from the page alone.
    """

    ORPHAN_SHORT_FORM = "orphan_short_form"
    """A short form for a case the filing never gives in full.

    Not a missing citation -- a defect in the one that is there, and worth
    reporting whether or not anything is recovered from it.
    """


@dataclass(frozen=True, slots=True)
class Candidate:
    """One span a generator proposes, with what a reader needs to judge it."""

    generator: str
    """The module that proposed it, kept on the record it becomes."""

    kind: CandidateKind
    span: Span
    """Indexes the document text, never a copy of it."""

    window: Span
    """The surrounding text a reader is shown, and the text a promotion re-reads."""

    note: str = ""
    """Why this generator proposed it, in a sentence."""


class Verdict(str, Enum):
    """A reader's answer."""

    ACCEPTED = "accepted"
    REJECTED = "rejected"
    UNDECIDED = "undecided"
    """The reader declined. Undecided is a real answer and is recorded as one."""


@dataclass(frozen=True, slots=True)
class Adjudication:
    """A candidate and what became of it."""

    candidate: Candidate
    verdict: Verdict
    reason: str = ""
    reviewer: str = ""
    """What answered -- a rule, a model and which one, or a person."""
