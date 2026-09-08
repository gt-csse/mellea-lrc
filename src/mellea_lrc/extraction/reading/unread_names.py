r"""Case names the extraction chain did not consume.

This runs last, over the document with every citation blanked, so what it looks
at is by definition what nothing else read. A case name standing in that
residue is one of two things, and neither is visible anywhere else in the
record:

*   a case the filing cites with no volume, reporter, page or docket number, so
    a reporter-driven tokenizer produces nothing at any relaxation and the
    citation leaves no trace at all -- document 013 lists five authorities that
    way, two of which `validation-v2.0` records as identity defects;
*   a case that *was* read, whose name eyecite did not reach. `In re BYJU ' s
    Alpha, Inc. , 2024 WL 1455586` comes back with the party parsed as
    `Alpha, Inc.`, because extraction spaces the apostrophe out of `BYJU's` and
    the name search stops there. The citation is in the record; half its name
    is not.

**No rule here decides which.** Masking already answers the only question this
could ask -- whether a locator was read at that position -- so there is nothing
to reconstruct and nothing to suppress. A textual short form, a table of
authorities entry, the caption naming the filing's own parties: all are reported
the same way, because separating them is a reading and this is the residue a
reader is given.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from mellea_lrc.core.spans import Span

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mellea_lrc.extraction.types import ExtractedCitation

# A party name: capitalised words, the punctuation a company name carries, and
# the lowercase words that sit inside one -- `U.S. Department of the Interior`,
# `Chugach Natives, Inc.`, `Bell Atl. Sys. Leasing Int'l, Inc.` A word may run
# across a spaced apostrophe, which is what extraction leaves of `BYJU's`.
_TOKEN = r"[A-Z][\w.'’&\-]*(?:[^\S\r\n]*['’][^\S\r\n]*\w[\w.'’&\-]*)*"
# `and` is deliberately absent: it joins two case names far more often than it
# sits inside one.
_INNER = r"(?:of|the|for|in|on|at|to|by|with|ex|rel\.|de|van|von|del|la|le)"
_PARTY = rf"{_TOKEN}(?:,?\s+(?:{_INNER}\s+){{0,2}}{_TOKEN}){{0,9}}"

# `v.` is the whole signal. `In re` and `Ex parte` are the case-name forms with
# no `v.` in them. `Matter of` is absent: `as a matter of law` is far more
# common in a brief than `Matter of Smith`.
_NAMES = (
    re.compile(rf"\b(?P<name>{_PARTY}\s+(?:v\.|vs\.|v\b)\s+{_PARTY})"),
    re.compile(rf"\b(?P<name>(?:In\s+re|Ex\s+parte)\s+{_PARTY})"),
)


def unread_case_names(text: str, citations: Sequence[ExtractedCitation]) -> tuple[Span, ...]:
    """Spans of `text` that name a case and lie outside every citation read."""
    blanked = list(text)
    for citation in citations:
        span = citation.full_span
        blanked[span.start : span.end] = " " * (span.end - span.start)
    masked = "".join(blanked)

    found: list[tuple[int, int]] = []
    for pattern in _NAMES:
        found.extend(match.span("name") for match in pattern.finditer(masked))

    kept: list[Span] = []
    for start, end in sorted(found):
        if kept and start < kept[-1].end:
            continue
        kept.append(Span(start=start, end=end))
    return tuple(kept)
