r"""A case named with no locator: `Chugach Natives, Inc. v. Doyon, Ltd. (D. Alaska 1984)`.

Bluebook Rule 10.2 gives one shape for a full case citation -- name, volume,
reporter, first page, then the court and year. A filing that writes the name and
the parenthetical and nothing between them has cited a case in a form that
locates nothing, and eyecite extracts nothing at all there: its tokenizer is
reporter-driven, so with no reporter there is no token, at any relaxation. The
citation is not *missed*, it is invisible, and no widening of the reporter rules
reaches it because there is no locator to widen.

That matters more than the count suggests. Every other defect this project finds
is a defect in something the record holds; this one leaves no record, so a filing
full of these earns "nothing to report". Document 013 lists five authorities this
way under a heading of "Relevant Legal Precedents and Authorities", and two of
them are in `validation-v2.0` as identity defects that the pipeline had no span
for.

## Why it is a defect of identity and not of form

The omission is not cosmetic. A reader given `Akiachak Native Community v. U.S.
Department of the Interior (D.D.C. 2016)` cannot reach a decision, and the one
identifying fact supplied is false: that case is the D.C. Circuit's at
`827 F.3d 100`, while `D.D.C.` names a different court whose opinion in the same
litigation is a different one. `Chugach Natives, Inc. v. Doyon, Ltd.` is Ninth
Circuit and 1978, not District of Alaska and 1984. A name with an invented
court-and-year is what a citation looks like when nobody consulted a reporter,
which is why the shape is worth finding rather than tolerating.

## What is not this

**A textual short form.** `Doe v. Skyline denied anonymity because ...` is proper
where the case was given in full earlier (Rule 10.9), and document 022 does it
repeatedly. Those are suppressed by checking whether some extracted citation in
the same document names both parties.

**A table of authorities.** A table lists names against the filing's own page
numbers, and its reporter citations sit in another cell. Rows are suppressed by
the pipes and dot leaders a table is rendered with.

**The filing's own caption.** Not suppressed, and a reviewer will see them: the
parties of the case being litigated are not an authority, but they are written in
the same shape, and no rule here can tell them apart from a cited case that the
filing happens never to cite in full.

This proposes; it does not decide. Whether a name-only reference was meant as a
citation at all is exactly the judgement the adjudication layer exists to ask for.
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
_GENERATOR = "nonconforming_citations"

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
# have no `v.` at all, and a filing citing one by name alone is the same defect.
_ADVERSARIAL = re.compile(rf"\b(?P<name>{_PARTY}\s+(?:v\.|vs\.|v\b)\s+{_PARTY})")
# `Matter of` is deliberately absent: `as a matter of law` is far more common in
# a brief than `Matter of Smith`, and no cheap rule separates them.
_ONE_SIDED = re.compile(rf"\b(?P<name>(?:In\s+re|Ex\s+parte)\s+{_PARTY})")

# A docket number is a locator. Bluebook Rule 10.8.1 cites an unreported case by
# docket number, court and date, and document 015 does it correctly eleven times:
# `In re Muscletech Research and Dev. Inc., No. 06-01147 (JMP) (Bankr. S.D.N.Y.
# Jan. 18, 2006)`. Nothing is missing from those.
_DOCKET = re.compile(
    r"(?:Case\s+|Civ(?:il)?\s+Action\s+)?(?:No\.?|Nos\.?|Dkt\.?|D\.I\.)\s*[\[(]?\s*\d",
    re.IGNORECASE,
)

# How far after a name a locator may begin and still belong to it, and what may
# stand in between: more of the name, and nothing else. Two things need this.
# `Rivero v. Bd. of Regents , No. CIV 16-0318 JB\SCY, 2019 WL 1085179` is one
# citation whose parties eyecite did not parse, and `In re Muscletech Research
# and Dev. Inc., No. 06-01147` runs past where the pattern above stops. In both
# the gap is the rest of a case name, which prose would not be.
_ITS_OWN_CITATION = 120
_NAME_TAIL = re.compile(
    r"(?:[\s,.;:'’&()\[\]\\/-]|\d|[A-Z][\w.'’&-]*"
    r"|(?:of|the|and|for|in|on|at|to|by|with|ex|rel)\b)*"
)

# Where a name is looked for when deciding whether the filing cites it properly
# somewhere else: the run-up to each extracted citation, which is where a case
# name sits whether or not eyecite parsed it into party fields.
_NAME_BEFORE_CITATION = 130

# A court-and-year parenthetical directly after the name. Not required -- the
# filing may give nothing at all -- but it is what makes a bare name unambiguously
# a citation rather than prose.
_DATE_PARENTHETICAL = re.compile(r"\s*[(\[][^)\]]{0,44}?(?:1[89]|20)\d{2}\s*[)\]]")

# How a table of authorities is rendered once the PDF has been converted: cell
# pipes, and the leaders running to the filing's own page numbers.
_TABLE = re.compile(r"[|]|…{2,}|\.{4,}")

# Words that identify nobody, so matching on them would suppress real candidates.
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
    authorities cell or in front of a citation eyecite read partially is still
    the filing citing that case properly -- and a later `Doe v. Skyline` is then
    a textual short form under Rule 10.9 rather than a case cited from nowhere.
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


def _already_cited(name: str, cited: list[set[str]]) -> bool:
    """Whether some extracted citation names the case this reference names.

    Both sides must match for an adversarial name, because one is not enough:
    document 022 cites eleven cases whose plaintiff is `Doe`, and matching on
    that alone would suppress every one of them. One word is enough for an
    `In re` name, which has only one party to match on.
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


def _belongs_to_a_citation(masked: str, end: int, starts: list[int]) -> bool:
    """Whether an extracted citation follows this name with only name in between."""
    return any(
        end <= start < end + _ITS_OWN_CITATION and _is_name_tail(masked[end:start]) for start in starts
    )


def _cited_by_docket(masked: str, end: int) -> bool:
    """Whether a docket number follows this name, which Rule 10.8.1 permits."""
    return any(
        _is_name_tail(masked[end : match.start()])
        for match in _DOCKET.finditer(masked, end, end + _ITS_OWN_CITATION)
    )


def _citation_starts(document: ExtractedDocument) -> list[int]:
    return sorted(item.full_span.start for item in document.citations)


def _line(text: str, position: int) -> str:
    start = text.rfind("\n", 0, position) + 1
    end = text.find("\n", position)
    return text[start : end if end != -1 else len(text)]


def nonconforming_citations(document: ExtractedDocument) -> Iterator[Candidate]:
    """Propose case names the filing states with no locator of any kind."""
    text = document.text
    masked = mask_full_spans(document)
    cited = _cited_names(document)
    starts = _citation_starts(document)

    found: list[tuple[int, int]] = []
    for pattern in (_ADVERSARIAL, _ONE_SIDED):
        found.extend(match.span("name") for match in pattern.finditer(masked))

    taken: list[tuple[int, int]] = []
    for start, end in sorted(found):
        if any(not (end <= other or start >= other_end) for other, other_end in taken):
            continue
        if _TABLE.search(_line(masked, start)):
            continue
        if _cited_by_docket(masked, end) or _belongs_to_a_citation(masked, end, starts):
            continue
        if _already_cited(text[start:end], cited):
            continue
        taken.append((start, end))
        dated = _DATE_PARENTHETICAL.match(masked, end)
        stop = dated.end() if dated else end
        yield Candidate(
            generator=_GENERATOR,
            kind=CandidateKind.NONCONFORMING,
            span=Span(start=start, end=stop),
            window=Span(start=max(0, start - WINDOW), end=min(len(text), stop + WINDOW)),
            note=(
                "A case name followed by a court and year, with no volume, reporter or page anywhere in it."
                if dated
                else "A case name with no locator and no court or year beside it."
            ),
        )
