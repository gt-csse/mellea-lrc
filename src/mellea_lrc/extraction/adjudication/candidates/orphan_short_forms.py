r"""A short form for a case the filing never gives in full.

Bluebook Rule 10.9 permits `833 F.2d at 186` only after the case has been cited
in full in the same document. A filing that writes only the short form has made
a claim about a page of a case it never identified, and 27 of the 304 short
forms across the 77 mined filings do exactly that -- in 16 of 77 documents.

**Two different things produce this shape, and telling them apart is the
reviewer's job**, which is why this generator proposes rather than decides:

*   The filing never introduced the case. 20 of the 27 read this way, and
    checking is not subtle -- document 70764936_25 cites Twombly five times and
    Iqbal three, always as `550 U.S. at 555` and `556 U.S. at 678`, and the
    strings `544` and `662` do not occur in its 53,793 characters. Nothing is
    recoverable here. The candidate is the finding.
*   Extraction missed the full citation. 2 of the 27, both the same table row
    set in capitals, which `reporter_sites` proposes separately. Here the
    candidate is recoverable, and promoting it fixes the short form too.

A short form states a **pin cite**, not a first page, so neither kind can be
resolved by a lookup: `833 F.2d 186` would find nothing, or find whatever case
does begin there and answer confidently about the wrong one. What identifies it
is the party name, which is why this waits for a reviewer.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from mellea_lrc.core.citations import CitationKind
from mellea_lrc.core.spans import Span
from mellea_lrc.extraction.adjudication.types import Candidate, CandidateKind
from mellea_lrc.extraction.structure.citation_tree import build_citation_tree

if TYPE_CHECKING:
    from collections.abc import Iterator

    from mellea_lrc.extraction.types import ExtractedDocument

WINDOW = 150
_GENERATOR = "orphan_short_forms"


def _full_form_in_text(text: str, volume: str, reporter: str) -> bool:
    """Whether the document states this volume and reporter with some other page.

    Whitespace-tolerant and case-insensitive, because the point is to find a
    full citation extraction may have missed rather than one it read.
    """
    letters = r"\s*".join(re.escape(character) for character in reporter if not character.isspace())
    pattern = re.compile(rf"\b{re.escape(volume)}\s*{letters}\s+\d{{1,4}}\b", re.IGNORECASE)
    return bool(pattern.search(text))


def orphan_short_forms(document: ExtractedDocument) -> Iterator[Candidate]:
    """Propose short case citations that resolved to no authority."""
    text = document.text
    for item in build_citation_tree(document).unattributed:
        citation = item.citation
        if citation.kind is not CitationKind.SHORT_CASE:
            continue
        volume = getattr(citation, "volume", None)
        reporter = getattr(citation, "reporter", None)
        if not (volume and reporter):
            continue
        recoverable = _full_form_in_text(text, volume, reporter.as_written)
        yield Candidate(
            generator=_GENERATOR,
            kind=CandidateKind.ORPHAN_SHORT_FORM,
            span=item.locator_span,
            window=Span(
                start=max(0, item.full_span.start - WINDOW),
                end=min(len(text), item.full_span.end + WINDOW),
            ),
            note=(
                "the volume and reporter appear elsewhere with another page, so the full "
                "citation may be there and unread"
                if recoverable
                else "no full citation of this volume and reporter anywhere in the document"
            ),
        )
