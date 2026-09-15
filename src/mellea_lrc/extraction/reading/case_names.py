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
IDENTIFIER = re.compile(
    r",\s*(?=No\s*\.|Case\s+No\s*\.|\d{1,4}\s+(?:WL|U\.\s?S\.|[A-Z][A-Za-z]*\.)"
    # A public-domain citation: `Bosh v. Cherokee County Bldg. Auth., 2013 OK 9`.
    r"|\d{4}\s+[A-Z]{2}(?:\s+[A-Z]{2,4})?\s+\d+)"
)
# How a case with no adverse party is named. eyecite's span opens after it and
# its `defendant` drops it, so `In re Giftcraft Ltd.` parses as `Giftcraft
# Ltd.`; the filing's name is the whole of it.
_NO_PARTY = re.compile(r"(?:In\s+re|In\s+the\s+Matter\s+of|Matter\s+of|Ex\s+parte)\s+$", re.I)
# What a converted table puts between one entry and the next: the leader dots
# that run to the page number, and the rules of the table itself. A span that
# opens above one of these has run through a neighbouring entry.
_ENTRY_BREAK = re.compile(r"[….]{2,}|\|")
_WORD = re.compile(r"[A-Za-z]")
# The heading a filing writes above a name: `Case Law: Sedima, S.P.R.L. ...`.
_LABEL = re.compile(r"^[A-Z][A-Za-z ]{0,18}:\s*")
# The number of a list, or the tail of the sentence before: `1) In Garrett`,
# `ERISA). In Womack`. A word in a parenthesis with no period after it belongs
# to the name -- `Muscogee (Creek) Nation v. Pruitt`.
_ENUMERATOR = re.compile(r"^(?:\d{1,3}\)|[A-Za-z0-9]{1,8}\)\.)\s+")
# A located name that begins at the `v.` has lost its first party.
_OPENS_AT_VERSUS = re.compile(r"^vs?\.?(?=\s)", re.I)
# The word in front of it, and the punctuation a filing puts before a name: a
# quotation dash, an opening bracket, the space after a signal.
_PARTY_BEHIND = re.compile(r"[A-Z][\w.'’&-]*[^\S\r\n]*$")
# A located name that is nothing but the suffix of a party has lost the party.
# eyecite guesses a short form's antecedent from the one token in front of the
# citation, so `Service By Air, Inc., supra` comes back as `Inc.` -- a word in
# every other corporate caption, which identifies none of them.
_SUFFIX_ONLY = re.compile(
    r"^(?:Inc|LLC|L\.\s?L\.\s?C|Co|Corp|Ltd|L\.\s?P|LLP|PLLC|P\.\s?C|N\.\s?A|"
    r"Ass'?n|Assocs?|Grp|Group|Bros|Partners)\.?$",
    re.I,
)
# One word of a party, read backwards from its suffix: a capitalised word, or
# one of the small words a caption keeps between them. A word that is neither
# is the sentence in front of the name, and the name stops there.
_WORD_BEHIND = re.compile(
    r"(?:[A-Z][\w.'’&-]*|&|of|the|by|and|for|in|at|on|de|la|le|van|von)"
    r"[^\S\r\n]*,?[^\S\r\n]*$"
)
#: How far back the party in front of a suffix is looked for. Six words covers
#: every one in this corpus and stops a runaway from eating a sentence.
_PARTY_WORDS = 6
# The punctuation a filing puts in front of a name: a quotation mark, a
# quotation dash, an opening bracket, the space after a signal.
_OPENING = " \t\n,.;:|·•()-\"'“”‘’"
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


def _strip_lead(text: str, start: int, end: int) -> tuple[int, int]:
    """The span with the signals and carrying words in front of the name cut off."""
    start, end = _trim(text, start, end)
    lead = _LEAD.match(text[start:end])
    while lead:
        start, end = _trim(text, start + lead.end(), end)
        lead = _LEAD.match(text[start:end])
    return start, end


def _party_in_front(text: str, start: int, floor: int) -> int:
    """Where the party begins whose suffix is written at `start`.

    Walks back a word at a time and stops at the first that is neither
    capitalised nor one of a caption's own small words, which is the sentence
    in front of the name: `to pierce the corporate veil. Service By Air, Inc.`
    stops at `veil`.
    """
    at = start
    for _ in range(_PARTY_WORDS):
        behind = _WORD_BEHIND.search(text, floor, at)
        if behind is None or behind.end() != at:
            break
        at = behind.start()
    return at


def locate_case_name(text: str, citation: CitationBase, locator: Span, floor: int = 0) -> Span | None:
    """The span of the case name this citation is written under, if there is one.

    `None` when the filing writes no name at this citation -- a bare `Id.`, a
    short form with no party beside it, or a table entry whose name the
    converter moved elsewhere.
    """
    start = getattr(citation, "full_span_start", None)
    if start is None or start >= locator.start:
        return None
    # A name cannot begin before the citation in front of it ends. eyecite
    # opens the span of the second of two citations written together back at
    # the first: `Twombly , 550 U.S. 544 (2007) and Ashcroft v. Iqbal , 556
    # U.S. 662` gives the Iqbal citation a window that starts at `Twombly`.
    # Parallel citations are the exception -- `231 Kan. 595, 598, 647 P.2d 320`
    # writes one name for both -- and there the citation in front leaves no
    # letters behind it, so only a window with words of its own is cut back.
    if floor > start and _WORD.search(text[floor : locator.start]):
        start = floor
    if start >= locator.start:
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
    # `Case Law: Sedima, S.P.R.L. v. Imrex Co.` opens with the heading above it,
    # and `1) In Garrett v. Selby` with the number of a list. Either can leave a
    # signal or a carrying word behind it, so the lead is read again after.
    for prefix in (_LABEL, _ENUMERATOR):
        found = prefix.match(text[start:end])
        if found:
            start, end = _trim(text, start + found.end(), end)
            lead = _LEAD.match(text[start:end])
            while lead:
                start, end = _trim(text, start + lead.end(), end)
                lead = _LEAD.match(text[start:end])
    if end <= start or not re.search(r"[A-Za-z]", text[start:end]):
        return None
    opener = _NO_PARTY.search(text[max(0, start - 30) : start])
    if opener:
        start -= len(opener.group())
    elif _SUFFIX_ONLY.match(text[start:end]):
        # The signal in front of the name is read again, because widening back
        # over `Service By Air` also walks back over the `See` in front of it.
        start, end = _strip_lead(text, _party_in_front(text, start, floor), end)
    elif _OPENS_AT_VERSUS.match(text[start:end]):
        # eyecite's span begins at `v.` when the party in front of it did not
        # parse -- after a quotation dash, or where the converter spaced the
        # name. The party is written right there, and one capitalised word back
        # is the whole of what is missing in every instance on this corpus.
        behind = _PARTY_BEHIND.search(text[max(0, start - 40) : start])
        if behind:
            start -= len(behind.group())
    return Span(start=start, end=end)
