r"""A case named where the deterministic pass read no citation.

Run over the document with every citation blanked, so what is left is by
definition what nothing else reached. Three different things stand in that
residue, they are indistinguishable by rule, and each is a different finding:

*   **A name belonging to the citation beside it.** `In re BYJU ' s Alpha,
    Inc. , 2024 WL 1455586` comes back with the party parsed as `Alpha, Inc.`,
    because extraction spaces the apostrophe out of `BYJU's` and the name
    search stops there. The citation is in the record; half its name is not,
    and the half that is missing is sitting right next to it.

*   **A proper short form.** Bluebook Rule 10.9 lets a filing write
    `Courts have granted anonymity in similar circumstances (see Doe v.
    Amazon.com , Doe v. Rose)` once it has given those citations in full. There
    is nothing at that position for a reporter-driven tokenizer to find -- no
    volume, no reporter, no docket number -- so the filing's reliance on the
    case leaves no trace at all.

*   **Not a citation.** The filing's own caption naming its own parties, a
    section heading, a roman-numeral list item where `v.` is the numeral five.

**No rule here decides which**, and that is the point: the generator proposes
the site and a reader answers. What separates the three is what the neighbouring
text and the rest of the document say, which is a reading.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mellea_lrc.extraction.adjudication.types import Candidate, CandidateKind
from mellea_lrc.extraction.reading.unread_names import unread_case_names
from mellea_lrc.model.spans import Span

if TYPE_CHECKING:
    from collections.abc import Iterator

    from mellea_lrc.model.document import Document

WINDOW = 220
_GENERATOR = "case_name_sites"


def case_name_sites(document: Document) -> Iterator[Candidate]:
    """Propose every case name standing outside the citations that were read."""
    text = document.text
    for span in unread_case_names(text, document.citations):
        yield Candidate(
            generator=_GENERATOR,
            kind=CandidateKind.CASE_NAME,
            span=span,
            window=Span(start=max(0, span.start - WINDOW), end=min(len(text), span.end + WINDOW)),
            note=f"{text[span.start : span.end]!r} names a case and no citation was read at it",
        )
