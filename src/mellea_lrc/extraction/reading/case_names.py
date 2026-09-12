r"""Where a citation's case name is written in the document.

eyecite does not report one. It reports the parties -- `plaintiff` and
`defendant` -- and it reports two offsets that bracket the name: `full_span_start`,
which is where the citation begins, and the start of the locator, which is where
the identifier begins. Everything between them is the name and whatever the
filing put after it.

So the name is not reconstructed from the parties, it is **located**. That
matters because a party pair cannot hold every name eyecite itself parses:
`In re Flint Water Cases` comes back with `defendant='Flint Water Cases'` and
the opening words dropped, `Ex parte Young` with `defendant='Young'`, and a
single-party short form with `defendant='Hassan'`. Rebuilding a name by joining
the fields with ` v. ` would produce a string the document does not contain,
and there would be nothing to point at.

Three things sit between the two offsets and are not the name:

*   **A signal or a sentence's own words.** `In Day v. Woodworth , 13 How. 363`
    opens the window at `In`, and `see also White v. McBride` at `see also`.
*   **A second identifier.** `Turner v. Murphy Oil USA, Inc., No. 05-4206, 2006
    WL 1984362` states a docket number before the reporter citation, and it is
    inside the window because the locator is the reporter one.
*   **Punctuation and the spacing extraction leaves**, which is trimmed off
    either end and never from the middle: `World Wide  Ass'n  of  Specialty
    Programs` is the name as the document has it, damage included.

What is left is reported as a span. Nothing is repaired and nothing is joined,
so the characters at the span are the characters the filing wrote.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from mellea_lrc.core.spans import Span

if TYPE_CHECKING:
    from eyecite.models import CitationBase

# What a filing puts in front of a case name that is not part of it: a Bluebook
# signal, or the word carrying the name into the sentence.
_LEAD = re.compile(
    r"^(?:see\s+also|see|accord|citing|quoting|e\.?\s?g\.?|cf\.|but\s+see|"
    r"compare|under|and|of|in)\s+(?!re\b)",
    re.I,
)
# Where the name stops and an identifier starts: a docket number, or a volume
# followed by a reporter. A party's own comma and digits survive, which is what
# keeps `United States v. Approximately 127,271 Bitcoin` whole.
IDENTIFIER = re.compile(r",\s*(?=No\s*\.|Case\s+No\s*\.|\d{1,4}\s+(?:WL|U\.\s?S\.|[A-Z][A-Za-z]*\.))")
# How a case with no adverse party is named. eyecite's span opens after it and
# its `defendant` drops it, so `In re Giftcraft Ltd.` parses as `Giftcraft
# Ltd.`; the filing's name is the whole of it.
_NO_PARTY = re.compile(r"(?:In\s+re|In\s+the\s+Matter\s+of|Matter\s+of|Ex\s+parte)\s+$", re.I)
# What a converted table puts between one entry and the next: the leader dots
# that run to the page number, and the rules of the table itself. A span that
# opens above one of these has run through a neighbouring entry.
_ENTRY_BREAK = re.compile(r"[….]{2,}|\|")
_OPENING = " \t\n,.;:|·•()"
# A period is not trimmed off the end. The window closes at the locator rather
# than at a sentence, so a name ending in one ends in an abbreviation --
# `Langston Equip. Assocs., Inc.`, `Louisville Land Co.` -- and dropping it
# would report a name the filing did not write.
_CLOSING = " \t\n,;:|·•("


def _trim(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start] in _OPENING:
        start += 1
    while end > start and text[end - 1] in _CLOSING:
        end -= 1
    return start, end


def locate_case_name(text: str, citation: CitationBase, locator: Span) -> Span | None:
    """The span of the case name this citation is written under, if there is one.

    `None` when the filing writes no name at this citation -- a bare `Id.`, a
    short form with no party beside it, or a table entry whose name the
    converter moved elsewhere.
    """
    start = getattr(citation, "full_span_start", None)
    if start is None or start >= locator.start:
        return None
    # An entry of a table of authorities ends at its leader dots, so anything
    # before the last of them belongs to the entry above.
    window = text[start : locator.start]
    breaks = [m.end() for m in _ENTRY_BREAK.finditer(window)]
    if breaks:
        start += breaks[-1]
    start, end = _trim(text, start, locator.start)
    lead = _LEAD.match(text[start:end])
    while lead:
        start, end = _trim(text, start + lead.end(), end)
        lead = _LEAD.match(text[start:end])
    cut = IDENTIFIER.search(text, start, end)
    if cut:
        start, end = _trim(text, start, cut.start())
    if end <= start or not re.search(r"[A-Za-z]", text[start:end]):
        return None
    opener = _NO_PARTY.search(text[max(0, start - 30) : start])
    if opener:
        start -= len(opener.group())
    return Span(start=start, end=end)
