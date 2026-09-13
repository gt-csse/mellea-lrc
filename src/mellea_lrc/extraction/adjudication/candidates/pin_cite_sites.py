r"""A page claim the rules are not confident of.

Every citation in this project is measured on two things: whether it was found,
and which page it claims. The second one is where the deterministic reading runs
out, and it runs out in four different ways that are one question to a reader --
**what page does this citation actually claim?**

*   **A page read and cut short.** `extract_pin_cite` matches with
    `strings_only`, so the text it is given stops at the first token that is not
    a string -- and `(quoting` is a stop word, hence a token. `Id. at 71
    (quoting …)` is matched against `' at 71 '`, and `550 U.S. at 570-71
    (quoting X)` parses to 570 with the `-71` in no parse at all. The record
    holds a page and the filing claims more of one.

*   **A page not read where the text plainly states one.** `763 F.2d 1472,
    1478: 'quote'` ends its page at a colon, which eyecite's pattern does not
    accept as a terminator; `874 S.W.2d 656,659 (Tex.1994)` writes no space
    after the comma. The record holds no page and the characters are right
    there.

*   **A page that cannot be one.** The converter drops a hyphen and `749-50`
    arrives as `74950`; it spaces digits apart and `435 U.S. 247, 264` arrives
    as a volume of 24 and pages of 7 and 264. A page of five or more digits, or
    one that falls before the case's own first page, is a reading of damage
    rather than a claim.

*   **A page on a citation eyecite never searched for.** After reading a full
    citation, eyecite looks through the rest of the document for the party name
    followed by a page and builds a citation for what it finds. It will not
    search for a name ending in a period -- a rule meant to keep `Co.` from
    being used as a search term, which also rules out every abbreviated party.
    So `Planned Parenthood Minn., N.D., S.D. at 732` is a case name and a page
    and no citation at all.

**No rule here decides which**, and none of them decides that the reading is
wrong. A page of five digits is nearly always converter damage and is
occasionally a paragraph number in a jurisdiction that runs them that high; the
text after a pin cite is nearly always a new sentence and is occasionally the
rest of the range. The generator says where to look and a reader answers.

The candidate names the citation it is about in `about`, except in the last
shape, where there is no citation yet and the reader's answer creates one.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from eyecite.utils import is_valid_name

from mellea_lrc.core.citations import CitationKind, citation_kind
from mellea_lrc.core.pin_cites import PinCiteKind
from mellea_lrc.core.spans import Span
from mellea_lrc.extraction.adjudication.types import Candidate, CandidateKind

if TYPE_CHECKING:
    from collections.abc import Iterator

    from mellea_lrc.core.record import CitationRecord
    from mellea_lrc.extraction.types import ExtractedDocument

WINDOW = 160
_GENERATOR = "pin_cite_sites"

#: A page of this many digits is not a page. `749-50` losing its hyphen is
#: `74950`; the longest page in `reporters-db` is four digits.
_TOO_MANY_DIGITS = 5
#: eyecite's own bound on how far a pin cite may sit past a first page, reused
#: here so the generator and the resolver disagree about nothing.
_MAX_PAGES = 150

# What the filing writes when the page claim continues past what was read: the
# other end of a range, another page, the footnote on one.
_CONTINUES = re.compile(r"\s*(?:[-–]\s*\*?\d|,\s*\*?\d|&?\s*n{1,2}\.\s*\d|\d)")
# A page claim standing where one belongs and no page was read: after the
# locator, with the connector a filing writes in front of it.
_STATES_A_PAGE = re.compile(r"[\s,]*(?:at\s+)?[*¶]{0,2}\s*\d")
# Where a reference citation's page would be: the name, then a page.
_AFTER_A_NAME = re.compile(r"\s*,?\s*(?:at\s+)?[*¶]{0,2}\s*\d+(?:\s*[-–]\s*\d+)?")


def _first_page(record: CitationRecord) -> int | None:
    page = getattr(record.stated, "page", None)
    return int(page) if page and str(page).isdigit() else None


def _damaged(record: CitationRecord) -> str | None:
    """Why the pages this citation was read with cannot be the ones it claims.

    Only a reporter page is compared against the case's first page. A star page
    and a paragraph number are not pages of that reporter at all: `2006 WL
    1984362, at *1` writes a document number where a first page belongs and a
    Westlaw page after it, and asking whether 1 falls before 1984362 is asking
    the wrong question of both numbers.
    """
    here = _first_page(record)
    for page in record.pin_cite_pages:
        if page.first is None or page.kind is not PinCiteKind.PAGE:
            continue
        if len(str(page.first)) >= _TOO_MANY_DIGITS:
            return f"{page.first} has {len(str(page.first))} digits and no reporter page does"
        # A database citation writes a document number where a first page
        # belongs -- `2024 WL 1455586` -- and a filing that writes its star page
        # without the star (`at 6-7`) then looks like a page before the first
        # one. Nothing here is a page, so nothing is compared.
        if here is None or len(str(here)) >= _TOO_MANY_DIGITS:
            continue
        if page.first < here:
            return f"{page.first} falls before {here}, the page the case begins on"
        if page.first > here + _MAX_PAGES:
            return f"{page.first} is more than {_MAX_PAGES} pages past {here}"
    return None


#: The kinds that claim a page of a case. A statute is not one of them: eyecite
#: reads a bare `§` as a citation, and `§ 1681e(b)` after it is a subsection
#: rather than a page. Nothing in this dataset scores a statute either way.
_CLAIMS_A_PAGE = frozenset(
    {
        CitationKind.FULL_CASE,
        CitationKind.SHORT_CASE,
        CitationKind.ID,
        CitationKind.REFERENCE,
    }
)


def pin_cite_sites(document: ExtractedDocument) -> Iterator[Candidate]:
    """Propose every place a page claim may be more, or other, than was read."""
    text = document.text
    taken = [(record.full_span.start, record.full_span.end) for record in document.citations]
    starts = sorted(record.locator_span.start for record in document.citations)
    for record in document.citations:
        if citation_kind(record.stated) not in _CLAIMS_A_PAGE:
            continue
        pin = record.pin_cite_span
        locator = record.locator_span
        if pin is not None:
            note = _damaged(record)
            if note is not None:
                yield _site(text, pin, record, f"the page read here is damage: {note}")
            elif _reaches_a_number(text, _CONTINUES, pin.end, starts):
                yield _site(text, pin, record, "the page claim continues past what was read")
            continue
        if _reaches_a_number(text, _STATES_A_PAGE, locator.end, starts):
            yield _site(text, locator, record, "no page was read and the text after states one")
    yield from _refused_references(document, taken)


def _reaches_a_number(text: str, pattern: re.Pattern[str], at: int, starts: list[int]) -> bool:
    """Whether what follows is a page, rather than the next citation's volume.

    `411 U.S. 792, 93 S.Ct. 1817` writes a comma and a number after the page and
    the number is a volume: one case, two reporters, one position. What tells
    them apart is not the punctuation but whether a citation was read starting
    there, which the record already knows.
    """
    found = pattern.match(text, at)
    if found is None:
        return False
    return not any(at < start < found.end() for start in starts)


def _site(text: str, span: Span, record: CitationRecord, note: str) -> Candidate:
    return Candidate(
        generator=_GENERATOR,
        kind=CandidateKind.PIN_CITE,
        span=span,
        window=Span(start=max(0, span.start - WINDOW), end=min(len(text), span.end + WINDOW)),
        note=note,
        about=record.citation_id,
    )


def _refused_references(
    document: ExtractedDocument, taken: list[tuple[int, int]]
) -> Iterator[Candidate]:
    """A name and a page, where the name is one eyecite will not search for.

    After reading a full citation, eyecite looks through the rest of the
    document for the party name followed by a page -- `Bell at 546` -- and
    builds a citation for what it finds. It picks the name to search for with
    `is_valid_name`, which throws out anything ending in a period so that `Co.`
    is never used as a search term. That also throws out every abbreviated
    party, so no search is ever made for `Planned Parenthood Minn., N.D., S.D.`
    and the page claim after it is in no citation at all.

    This does the search eyecite did not, and proposes what it finds.
    """
    text = document.text
    for record in document.citations:
        if citation_kind(record.stated) is not CitationKind.FULL_CASE:
            continue
        refused = [
            party
            for party in (
                getattr(record.stated, "plaintiff", None),
                getattr(record.stated, "defendant", None),
            )
            if party and len(party) > 2 and not is_valid_name(party)
        ]
        for party in refused:
            pattern = re.compile(r"\s+".join(re.escape(word) for word in party.split()))
            for found in pattern.finditer(text, record.full_span.end):
                page = _AFTER_A_NAME.match(text, found.end())
                if page is None:
                    continue
                span = Span(start=found.start(), end=page.end())
                if any(a < span.end and span.start < b for a, b in taken):
                    continue
                yield Candidate(
                    generator=_GENERATOR,
                    kind=CandidateKind.PIN_CITE,
                    span=span,
                    window=Span(
                        start=max(0, span.start - WINDOW),
                        end=min(len(text), span.end + WINDOW),
                    ),
                    note=(
                        f"{party!r} ends in a period, so `is_valid_name` refuses it and no "
                        f"reference citation was built on it; the filing writes it again with "
                        f"a page after it"
                    ),
                )
