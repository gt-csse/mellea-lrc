r"""A case named with no locator attached: `Chugach Natives, Inc. v. Doyon, Ltd.`

Every other generator here disagrees with extraction about a span the record
holds. This one reports spans the record does not have. eyecite's tokenizer is
reporter-driven -- it finds a reporter string and builds a citation outward from
it -- so a filing that names a case and gives no volume, reporter or page
produces nothing at all, at any relaxation. Nothing is *missed*; there is no
locator to widen a rule around. A filing full of these earns "nothing to report".

## This isolates a population; it does not judge one

The name says what was found and not what it is worth. A case name with no
locator can be several things, and the difference is not decidable from the
shape:

*   a citation that locates nothing -- document 013 lists five authorities as
    `Akiachak Native Community v. U.S. Department of the Interior (D.D.C. 2016)`
    and similar, and two of those are `validation-v2.0` identity defects that
    until now had no span for anything to check;
*   a textual short form, which Bluebook Rule 10.9 permits once the case has been
    given in full, and document 022 writes repeatedly;
*   a mention that is not a citation at all -- the filing's own caption names its
    own parties in exactly this shape;
*   a name in a table of authorities whose citation column is empty.

Deciding between those is a reading, and rules that encode the reading here
would be Bluebook judgements fitted to 26 filings. So they are reported instead:
:attr:`~mellea_lrc.extraction.adjudication.types.Candidate.note` says whether a
court and year follow, whether the filing cites the case in full somewhere else,
and whether the site sits in a table. What is done with that is the reviewer's,
or the search branch's.

## The one thing that is suppressed, and why it is not a judgement

A name with a locator attached is not a name without one. An extracted citation
following the name, or a docket number -- which identifies a case as surely as a
reporter page does, and is how an unreported decision is cited -- means the
filing did state a locator, and proposing the site would be reporting something
false rather than something arguable. The tolerance runs to the next locator
across more of the name, because eyecite frequently parses no parties at all:
`Rivero v. Bd. of Regents of Univ. of New Mexico , No. CIV 16-0318 JB\SCY,
2019 WL 1085179` comes back with the name exposed and the citation intact.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from mellea_lrc.core.spans import Span
from mellea_lrc.extraction.adjudication.masking import mask_full_spans
from mellea_lrc.extraction.adjudication.types import Candidate, CandidateKind

if TYPE_CHECKING:
    from collections.abc import Iterator

    from mellea_lrc.extraction.types import ExtractedDocument

WINDOW = 200
_GENERATOR = "bare_case_names"

# A party name: capitalised words, the punctuation a company name carries, and
# the lowercase words that sit inside one -- `U.S. Department of the Interior`,
# `Chugach Natives, Inc.`, `Bell Atl. Sys. Leasing Int'l, Inc.`
_TOKEN = r"[A-Z][\w.'’&\-]*"
# `and` is deliberately absent: it joins two case names far more often than it
# sits inside one, and including it merges `Romano v. SLS Residential, Inc. and
# Chime v. Peak Sec. Plus, Inc.` into a single candidate.
_INNER = r"(?:of|the|for|in|on|at|to|by|with|ex|rel\.|de|van|von|del|la|le)"
_PARTY = rf"{_TOKEN}(?:,?\s+(?:{_INNER}\s+){{0,2}}{_TOKEN}){{0,9}}"

# `v.` is the whole signal. `In re` and `Ex parte` are the case-name forms that
# have no `v.` at all. `Matter of` is deliberately absent: `as a matter of law`
# is far more common in a brief than `Matter of Smith`.
_ADVERSARIAL = re.compile(rf"\b(?P<name>{_PARTY}\s+(?:v\.|vs\.|v\b)\s+{_PARTY})")
_ONE_SIDED = re.compile(rf"\b(?P<name>(?:In\s+re|Ex\s+parte)\s+{_PARTY})")

_DOCKET = re.compile(
    r"(?:Case\s+|Civ(?:il)?\s+Action\s+)?(?:No\.?|Nos\.?|Dkt\.?|D\.I\.)\s*[\[(]?\s*\d",
    re.IGNORECASE,
)

# A court-and-year parenthetical directly after the name: not a locator, but the
# thing that makes a bare name unambiguously meant as a citation.
_DATE_PARENTHETICAL = re.compile(r"\s*[(\[][^)\]]{0,44}?(?:1[89]|20)\d{2}\s*[)\]]")

# How a table of authorities is rendered once the PDF has been converted: cell
# pipes, and the leaders running to the filing's own page numbers.
_TABLE = re.compile(r"[|]|…{2,}|\.{4,}")

# How far after a name a locator may begin and still belong to it, and what may
# stand in between: more of the name, and nothing else.
_ITS_OWN_LOCATOR = 120
_NAME_TAIL = re.compile(
    r"(?:[\s,.;:'’&()\[\]\\/-]|\d|[A-Z][\w.'’&-]*|(?:of|the|and|for|in|on|at|to|by|with|ex|rel)\b)*"
)

# Where a name is looked for when reporting whether the filing cites it properly
# somewhere else: the run-up to each extracted citation, which is where a case
# name sits whether or not eyecite parsed it into party fields.
_NAME_BEFORE_CITATION = 130

# Words that identify nobody, so matching on them would say nothing.
_NOT_A_NAME = frozenset(
    {
        "inc",
        "llc",
        "ltd",
        "corp",
        "co",
        "lp",
        "pllc",
        "plc",
        "na",
        "assn",
        "association",
        "company",
        "corporation",
        "incorporated",
        "limited",
        "the",
        "of",
        "and",
        "for",
        "et",
        "al",
        "state",
        "states",
        "united",
    }
)


def _words(*values: str | None) -> set[str]:
    """The words in a case name that could identify a party."""
    found: set[str] = set()
    for value in values:
        if not value:
            continue
        for word in re.findall(r"[A-Za-z][\w'’]*", value.lower()):
            if len(word) >= 3 and word not in _NOT_A_NAME:
                found.add(word)
    return found


def _cited_names(document: ExtractedDocument) -> list[set[str]]:
    """One word set per extracted citation, from its parties and the text before it.

    The text as well as the parsed fields, because a case name in a table of
    authorities cell or in front of a citation eyecite read only partially is
    still the filing citing that case with a locator.
    """
    text = document.text
    names: list[set[str]] = []
    for item in document.citations:
        citation = item.citation
        start = item.full_span.start
        words = _words(
            getattr(citation, "plaintiff", None),
            getattr(citation, "defendant", None),
            getattr(citation, "antecedent", None),
            text[max(0, start - _NAME_BEFORE_CITATION) : start],
        )
        if words:
            names.append(words)
    return names


def _cited_in_full(name: str, cited: list[set[str]]) -> bool:
    """Whether some extracted citation names the case this name names.

    Both sides must match for an adversarial name, because one is not enough:
    document 022 cites eleven cases whose plaintiff is `Doe`. One word is enough
    for an `In re` name, which has only one party to match on.
    """
    parts = re.split(r"\s+(?:v\.|vs\.|v)\s+", name, maxsplit=1)
    if len(parts) != 2:
        words = _words(name)
        return bool(words) and any(words & cited_words for cited_words in cited)
    left, right = _words(parts[0]), _words(parts[1])
    if not left or not right:
        return False
    return any((left & words) and (right & words) for words in cited)


def _is_name_tail(gap: str) -> bool:
    """Whether everything between a name and a locator is more of the name."""
    return _NAME_TAIL.fullmatch(gap) is not None


def _has_a_locator(masked: str, end: int, starts: list[int]) -> bool:
    """Whether a citation or a docket number attaches to the name ending here."""
    if any(end <= start < end + _ITS_OWN_LOCATOR and _is_name_tail(masked[end:start]) for start in starts):
        return True
    return any(
        _is_name_tail(masked[end : match.start()])
        for match in _DOCKET.finditer(masked, end, end + _ITS_OWN_LOCATOR)
    )


def _line(text: str, position: int) -> str:
    start = text.rfind("\n", 0, position) + 1
    end = text.find("\n", position)
    return text[start : end if end != -1 else len(text)]


def _note(*, dated: bool, in_full: bool, in_table: bool) -> str:
    """What was observed about this site, without saying what it amounts to."""
    parts = ["A case name with no volume, reporter, page or docket number attached to it."]
    parts.append("A court and year follow it." if dated else "No court or year follows it.")
    if in_full:
        parts.append("The filing cites this case with a locator elsewhere.")
    if in_table:
        parts.append("The site is a row of a table.")
    return " ".join(parts)


def bare_case_names(document: ExtractedDocument) -> Iterator[Candidate]:
    """Propose every case name the filing states with no locator attached."""
    text = document.text
    masked = mask_full_spans(document)
    cited = _cited_names(document)
    starts = sorted(item.full_span.start for item in document.citations)

    found: list[tuple[int, int]] = []
    for pattern in (_ADVERSARIAL, _ONE_SIDED):
        found.extend(match.span("name") for match in pattern.finditer(masked))

    taken: list[tuple[int, int]] = []
    for start, end in sorted(found):
        if any(not (end <= other or start >= other_end) for other, other_end in taken):
            continue
        if _has_a_locator(masked, end, starts):
            continue
        taken.append((start, end))
        dated = _DATE_PARENTHETICAL.match(masked, end)
        stop = dated.end() if dated else end
        yield Candidate(
            generator=_GENERATOR,
            kind=CandidateKind.BARE_CASE_NAME,
            span=Span(start=start, end=stop),
            window=Span(start=max(0, start - WINDOW), end=min(len(text), stop + WINDOW)),
            note=_note(
                dated=dated is not None,
                in_full=_cited_in_full(text[start:end], cited),
                in_table=bool(_TABLE.search(_line(masked, start))),
            ),
        )
