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
from typing import TYPE_CHECKING, Generic, TypeVar

from mellea_lrc.core.spans import Span

if TYPE_CHECKING:
    from mellea_lrc.llm.ivr import IvrRun

T = TypeVar("T")


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

    CASE_NAME = "case_name"
    """A case named where no citation was read.

    Three different things wear this shape and only a reader can tell them
    apart: a name belonging to a citation beside it that was read without one,
    a proper short form for a case given in full elsewhere, and a name that is
    not a citation at all.
    """

    PIN_CITE = "pin_cite"
    """A page claim that may be more, or other, than what was read.

    Four shapes and one question. A page cut short by a stop word, a page the
    pattern would not terminate, a page the converter damaged past being one,
    and a page on a reference citation eyecite refused to build. Only a reader
    can say which of them is in front of it.
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

    about: str | None = None
    """The `citation_id` this candidate concerns, where it concerns one.

    A site proposing a citation nothing read has nothing to name; a site
    proposing that a citation's page is wrong has to say which citation, because
    the reader's answer is a correction to that record rather than a new one.
    """


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


@dataclass(frozen=True, slots=True)
class SiteReview(Generic[T]):
    """One site-review answer together with the complete reproducible IVR run."""

    answer: T | None
    reason: str | None
    run: IvrRun
