r"""A reporter abbreviation that names more than one reporter.

`5 Cranch 137` is United States Reports and it is also District of Columbia
Reports, and nothing in the citation says which. eyecite settles it by asking
which edition's years contain the citation's year, and that fails in both
directions on the same reporter:

    Marbury v. Madison, 5 Cranch 137 (1803)   1803 is inside both ranges
                                              -> no edition recorded at all
    Doe v. Roe, 5 Cranch 137 (1830)           1830 is inside only one
                                              -> District of Columbia, confidently

So extraction records the ambiguity instead of resolving it, and this proposes
the choice. It is a question a reader answers easily -- Marbury is a Supreme
Court case and the surrounding sentence says so -- and one a date cannot answer
at all, because the two editions overlap by forty years.

None of the 2,607 case citations in this project's corpora is ambiguous, so this
generator proposes nothing today. It exists because the alternative is a wrong
answer nobody sees: the citation would carry a canonical reporter naming the
wrong court, and no column would report it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mellea_lrc.core.spans import Span
from mellea_lrc.extraction.adjudication.types import Candidate, CandidateKind

if TYPE_CHECKING:
    from collections.abc import Iterator

    from mellea_lrc.extraction.types import ExtractedDocument

WINDOW = 120
_GENERATOR = "ambiguous_editions"


def ambiguous_editions(document: ExtractedDocument) -> Iterator[Candidate]:
    """Propose citations whose reporter abbreviation names several reporters."""
    text = document.text
    for item in document.citations:
        reporter = getattr(item.citation, "reporter", None)
        if reporter is None or len(reporter.editions) < 2:
            continue
        yield Candidate(
            generator=_GENERATOR,
            kind=CandidateKind.EDITION,
            span=item.locator_span,
            window=Span(
                start=max(0, item.span.start - WINDOW),
                end=min(len(text), item.span.end + WINDOW),
            ),
            note=(f"{reporter.as_written!r} names " + " or ".join(repr(n) for n in reporter.editions)),
        )
