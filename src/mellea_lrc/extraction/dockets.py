r"""Docket numbers, read by eyecite as a locator in their own right.

A reporter locator and a docket number do the same job. Each is the string that
identifies which case a filing is talking about, and eyecite has a shape for one
and no shape at all for the other::

    Calderon v. GEICO Gen. Ins. Co., No. 1:19-CV-362 (M.D.N.C. Jan. 26, 2021)
                                     ^ the locator, and eyecite emits nothing

That silence is not a small gap. A case too recent or too minor to have reached
a reporter is cited by docket and by nothing else, which is exactly the
population where a fabricated citation is hardest to catch: there is no
reporter page to check it against. On false-citation-bench the docket is the
only identifier for twelve occurrences across five filings, and one of them --
the indictment in document 016 -- is the head of a chain of fifteen `Id. ¶ N`
references that had nowhere to attach.

So rather than reading dockets in a pass of our own, this tells eyecite to keep
going: a :class:`~eyecite.models.TokenExtractor` registered with the tokenizer,
emitting an ordinary ``CitationToken``. Everything downstream then happens for
free and happens the same way it does for a reporter citation -- eyecite finds
the case name in front of it, the pin cite and year behind it, groups repeat
occurrences under one resource, and points `Id.` at it.

Two things make a docket different from a reporter locator, and both are
handled here rather than downstream.

**The court is half of the identifier.** ``1:19-cv-362`` exists in every
district; only ``1:19-cv-362`` *in the Middle District of North Carolina* names
a case. So the court is resolved against ``courts-db`` -- the same database
CourtListener uses -- and a docket number with no court written alongside it is
declined rather than guessed at. That is the same rule the model-assisted path
follows, where the model is offered a closed set of courts found in the text so
that it must pick or decline; a deterministic extractor should be at least as
strict.

**A filing states its own docket number, everywhere.** It stands in the caption
and in every ECF page stamp -- document 020 carries twenty identical ``Case
2:25-cv-01295-GMS Document 1 Filed 04/18/25`` lines, page furniture that
preprocessing should have dropped and did not. Those are not citations. What
separates them from a citation is not the number but what follows it: a cited
docket is followed by its court, in parentheses, on the same block of text::

    No. 1:25-cr-00312-RPK (E.D.N.Y. filed Oct. 8, 2025)     a citation
    Case 2:25-cv-01295-GMS  Document 1  Filed 04/18/25      a page stamp

Requiring that is what makes the extractor usable at all. Proximity alone is
not enough: document 022 stamps ``Case 2:25-cv-01295-GMS`` forty characters
after another case's ``(N.D. Cal. May 13, 2011)``, and document 006 has its own
caption number sitting a blank line before an unrelated ``(10th Cir. 1994)``.
Both are excluded by insisting the court parenthesis follow the number without
a paragraph break between them -- the same block-boundary rule
:mod:`~mellea_lrc.extraction.relaxation` applies to a broken reporter citation,
for the same reason: what lies beyond a blank line belongs to something else.

Measured on the 26 documents of false-citation-bench, this reads twelve docket
citations in five documents, finds all eleven the model-assisted hunt found and
one it missed, and picks up none of the twenty ECF stamps.

What it does not read is stated rather than hidden: the federal
``office:year-type-sequence`` shape only. State docket numbers, appellate
``No. 23-1234`` and administrative numbers have no shape this strict, and a
pattern loose enough to catch them also catches statutory subsections and phone
numbers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import cache, lru_cache

import ahocorasick
from courts_db import courts
from eyecite.models import CitationToken, Edition, Reporter, TokenExtractor
from eyecite.tokenizers import Tokenizer, default_tokenizer

# The case-type codes a federal docket number carries. Spelled out rather than
# written as `[a-z]{2,4}` so that the pattern cannot drift onto an arbitrary
# `12:30-am-1234`.
CASE_TYPES = ("cv", "cr", "bk", "md", "mj", "mc", "ap", "civ")

# What may sit between the parts of a docket number. A hyphen, and up to two
# spaces -- `4:25-cv- 00175` is a real citation in document 022, printed with a
# space inside the number -- but never a line ending: a number broken across
# lines is not distinguishable from two numbers.
_JOIN = r"-?[^\S\r\n]{0,2}"

# `No.`, `Case No.`, `Civil Action No:`, `CaseNo.`, and the bare `Case` of an
# ECF stamp. Kept inside the match so that the case name in front of a citation
# ends where it should -- without it eyecite reads `GEICO Gen. Ins. Co., No.`
# as the defendant -- and so that spans line up with how the bench records a
# docket occurrence.
_SIGNAL = (
    r"(?:\b(?:Case|Civil[^\S\r\n]+Action|Civ\.?[^\S\r\n]*A\.?|Docket)?[^\S\r\n]*"
    r"No\.?[^\S\r\n]*:?[^\S\r\n]*|\bCase[^\S\r\n]+)?"
)

DOCKET_NUMBER = (
    rf"{_SIGNAL}"
    r"\b(?P<office>\d{1,2}):(?P<year>\d{2})"
    rf"{_JOIN}(?P<case_type>" + "|".join(CASE_TYPES) + rf"){_JOIN}"
    r"(?P<sequence>\d{3,6})"
    r"(?P<suffix>(?:-[A-Za-z]{2,4})+)?\b"
)
"""The shape of a federal docket number, with the signal that introduces it."""

# How far past the number the court may be written, and how much of a gap is
# still the same citation. One line ending is a citation broken by the page;
# two is a different thing on the page.
_COURT_WINDOW = 70
_PARAGRAPH_BREAK = re.compile(r"\r?\n[^\S\r\n]*\r?\n")
_BRACKETED = re.compile(r"[(\[]([^)\]\r\n]*)[)\]]")

# courts-db stores one-letter and two-letter citation strings that would match
# a judge's initials; four characters is the shortest real court abbreviation.
_MIN_COURT_STRING = 4
_NOT_ALPHANUMERIC = re.compile(r"[^0-9a-z]")

# How many words into a parenthetical a court name may run: `Bankr. S.D.N.Y.`
# is two, `E.D. Pa.` is two, and nothing real is longer than four.
_MAX_COURT_WORDS = 4
_WORD = re.compile(r"\S+")

# How far either side of a docket number a court string may be written and
# still be offered as a candidate for it.
_COURT_SEARCH_BEFORE = 90
_COURT_SEARCH_AFTER = 140

# The group that marks a citation token as a docket rather than a reporter.
DOCKET_GROUP = "docket"


@dataclass(frozen=True, slots=True)
class CourtCandidate:
    """One court string found in the text, resolved against courts-db."""

    span_start: int
    span_end: int
    text: str
    court_id: str
    court_name: str


def _normalize(value: str) -> str:
    """Compare court strings ignoring case and trailing punctuation.

    courts-db is not internally consistent about the final period -- the Eastern
    District of New York is stored as ``E.D.N.Y`` while the Southern District is
    ``S.D.N.Y.`` -- so an exact match would silently miss whole courts.
    """
    return value.casefold().rstrip(". ")


def _tight(value: str) -> str:
    """Compare court strings ignoring everything but their letters and digits.

    A converter that drops the space out of ``D. Ariz.`` has not written a
    different court, but it has written a string that matches ``Ariz.`` -- the
    Arizona Supreme Court -- and nothing else. Document 022 does exactly that.
    Answering `ariz` there would be inventing a court out of a typo, so the
    reading that ignores the spacing has to be available.
    """
    return _NOT_ALPHANUMERIC.sub("", value.casefold())


def _build_court_index() -> tuple[
    ahocorasick.Automaton, dict[str, tuple[str, str]], dict[str, tuple[str, str]]
]:
    """Index every court citation string courts-db knows, two ways."""
    lookup: dict[str, tuple[str, str]] = {}
    tight: dict[str, tuple[str, str]] = {}
    for court in courts:
        citation_string = court.get("citation_string")
        if not citation_string or len(citation_string) < _MIN_COURT_STRING:
            continue
        entry = (court["id"], court["name"])
        lookup.setdefault(_normalize(citation_string), entry)
        tight.setdefault(_tight(citation_string), entry)
    automaton = ahocorasick.Automaton()
    for normalized in lookup:
        automaton.add_word(normalized, normalized)
    automaton.make_automaton()
    return automaton, lookup, tight


_COURT_AUTOMATON, _COURT_LOOKUP, _COURT_TIGHT = _build_court_index()


def courts_in(text: str, start: int, end: int) -> tuple[CourtCandidate, ...]:
    """Every courts-db citation string written in ``text[start:end]``."""
    left = max(0, start)
    region = text[left:end]
    normalized = region.casefold()
    found: dict[tuple[int, int], CourtCandidate] = {}
    for finish, matched in _COURT_AUTOMATON.iter(normalized):
        begin = finish - len(matched) + 1
        before = normalized[begin - 1] if begin else " "
        after = normalized[finish + 1] if finish + 1 < len(normalized) else " "
        if before.isalnum() or after.isalnum():
            continue
        court_id, court_name = _COURT_LOOKUP[matched]
        found[(left + begin, left + finish + 1)] = CourtCandidate(
            span_start=left + begin,
            span_end=left + finish + 1,
            text=text[left + begin : left + finish + 1],
            court_id=court_id,
            court_name=court_name,
        )
    # Court abbreviations nest: "N.C" sits inside "D.N.C" inside "M.D.N.C", and
    # each is a real courts-db entry. Only the longest reading is the court
    # actually written, so drop any candidate contained in another.
    maximal = [
        candidate
        for candidate in found.values()
        if not any(
            other.span_start <= candidate.span_start
            and candidate.span_end <= other.span_end
            and (other.span_end - other.span_start) > (candidate.span_end - candidate.span_start)
            for other in found.values()
        )
    ]
    return tuple(sorted(maximal, key=lambda candidate: candidate.span_start))


def courts_near(text: str, start: int, end: int) -> tuple[CourtCandidate, ...]:
    """Return court strings written close enough to identify this docket."""
    return courts_in(text, start - _COURT_SEARCH_BEFORE, end + _COURT_SEARCH_AFTER)


def court_for_docket(text: str, end: int) -> CourtCandidate | None:
    """The court written with the docket number that ends at ``end``.

    A cited docket carries its court in the parenthesis that follows it, in the
    same block of text. Nothing else counts: a court string merely nearby
    belongs to whatever citation put it there, which on a page of ECF stamps is
    never this one.
    """
    for bracket in _BRACKETED.finditer(text, end, end + _COURT_WINDOW):
        if _PARAGRAPH_BREAK.search(text, end, bracket.start()):
            return None
        court = _court_opening(text, bracket.start(1), bracket.end(1))
        if court is not None:
            return court
    return None


def _court_opening(text: str, start: int, end: int) -> CourtCandidate | None:
    """The court a parenthetical opens with, if it opens with one.

    A citation parenthetical begins with the court and then gives the date --
    ``(E.D.N.Y. filed Oct. 8, 2025)`` -- so the court is read from the front
    rather than searched for anywhere inside. That is stricter than scanning,
    and the strictness is the point: a parenthetical that merely mentions a
    court somewhere in the middle is quoting another citation, not naming this
    docket's court.
    """
    words = list(_WORD.finditer(text, start, end))
    for count in range(min(_MAX_COURT_WORDS, len(words)), 0, -1):
        opening = text[words[0].start() : words[count - 1].end()]
        entry = _COURT_TIGHT.get(_tight(opening))
        if entry is None or not _is_written_as_a_court(opening):
            continue
        court_id, court_name = entry
        return CourtCandidate(
            span_start=words[0].start(),
            span_end=words[count - 1].end(),
            text=opening,
            court_id=court_id,
            court_name=court_name,
        )
    return None


def _is_written_as_a_court(opening: str) -> bool:
    """Whether this reads as a court abbreviation rather than as initials.

    Ignoring the periods is what lets ``D.Ariz.`` be read; the cost is that it
    also lets ``(SC)`` be read as South Carolina, and the parenthesis after a
    caption's docket number holds the assigned judge's initials -- ``(JMW)``,
    ``(RPK)``. A court is written as an abbreviation, with the periods, or as a
    whole word; initials are neither.
    """
    return "." in opening or (opening.isalpha() and len(opening) >= _MIN_COURT_STRING)


@cache
def _edition(court_id: str, court_name: str, case_type: str) -> Edition:
    """The docket series this citation belongs to: one court's cases of one type.

    eyecite identifies a case by volume, reporter and page, and it reads the
    reporter through the edition it resolved to. Making the court part of the
    edition is what keeps two districts' ``1:19-cv-362`` from being read as one
    case, and what makes ``No. 1:19-CV-362`` and ``no. 1:19-cv-362`` read as the
    same one.
    """
    return Edition(
        reporter=Reporter(
            short_name=court_id,
            name=court_name,
            # Not one of reporters-db's cite types. `source` must be
            # "reporters", because that is what tells eyecite to build a
            # FullCaseCitation rather than a statute or a journal article.
            cite_type="docket",
            source="reporters",
        ),
        short_name=f"{court_id} {case_type}",
        start=None,
        end=None,
    )


def docket_token(match: re.Match[str], extra: dict, offset: int = 0) -> CitationToken:
    """Build the citation token for one docket number.

    The three parts of a docket number stand in for the three parts of a
    locator, which is the whole of the idea: the office is the volume, the
    court's docket series is the reporter, and the year and sequence together
    are the page. eyecite's own machinery then treats it as it treats any other
    case citation.

    The page deliberately keeps its hyphen. eyecite rejects an `Id.` whose pin
    cite cannot be a page within 150 pages of a numeric one, which is a sound
    rule for a reporter and a meaningless one for a docket -- and it is what
    would throw away the fifteen `Id. ¶ N` references that make this worth
    doing.
    """
    del extra
    court = court_for_docket(match.string, match.end())
    if court is None:  # pragma: no cover - get_matches has already declined it
        msg = "a docket number is only a citation when its court is written with it"
        raise ValueError(msg)
    groups = match.groupdict()
    case_type = groups["case_type"].lower()
    return CitationToken(
        match.group(0),
        match.start() + offset,
        match.end() + offset,
        groups={
            "volume": groups["office"],
            "reporter": f"{court.court_id} {case_type}",
            "page": f"{groups['year']}-{groups['sequence']}",
            DOCKET_GROUP: match.string[match.start("office") : match.end()],
            "court": court.court_id,
            "court_name": court.court_name,
            "court_text": court.text,
        },
        exact_editions=(_edition(court.court_id, court.court_name, case_type),),
    )


class _DocketExtractor(TokenExtractor):
    """A token extractor that declines a docket number with no court.

    ``TokenExtractor`` hands its constructor the match and nothing else, and
    whether a docket number is a citation is a fact about the text after it.
    ``get_matches`` is where eyecite gives an extractor the whole document, so
    it is where the question gets asked.
    """

    def get_matches(self, text: str) -> list[re.Match[str]]:
        """Return only the docket numbers whose court is written with them."""
        return [
            match for match in super().get_matches(text) if court_for_docket(text, match.end()) is not None
        ]


@lru_cache(maxsize=1)
def docket_extractors() -> tuple[TokenExtractor, ...]:
    """The extractors that read docket numbers, to register with a tokenizer.

    ``strings`` is empty, which puts this among the extractors the ahocorasick
    prefilter always runs. There is no literal a docket number must contain --
    the case-type code is already in the pattern -- and one more regex per
    document is not a cost worth a prefilter.
    """
    return (
        _DocketExtractor(
            regex=DOCKET_NUMBER,
            constructor=docket_token,
            flags=re.IGNORECASE,
            strings=[],
        ),
    )


@dataclass
class _DocketAwareTokenizer(Tokenizer):
    """Whatever a tokenizer already reads, plus docket numbers.

    Composed rather than substituted because the two questions are unrelated.
    ``Relaxation`` decides how much whitespace damage a *reporter* pattern will
    tolerate, and it has nothing to say about a docket number -- so a docket is
    read the same way at every level, and each level's own prefilter is left
    exactly as it was.
    """

    base: Tokenizer = field(default_factory=lambda: default_tokenizer)

    def get_extractors(self, text: str) -> list[TokenExtractor]:
        """Run the base tokenizer's extractors, and then ours."""
        return [*self.base.get_extractors(text), *docket_extractors()]


def with_dockets(tokenizer: Tokenizer) -> Tokenizer:
    """Return a tokenizer that also reads docket numbers."""
    return _DocketAwareTokenizer(base=tokenizer)
