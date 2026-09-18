r"""Find docket-shaped locator candidates before resolving their context.

The stable extraction profile registers these patterns with eyecite's tokenizer,
so a docket enters the same canonical citation stream as a reporter locator.
Court and date are later readers: a docket is retained with court=None when no
court is written beside it, and validation can check that candidate before it
tries to resolve the docket to a case root.

The district shape is self-describing (office, year, case type and sequence),
while bankruptcy numbers need an introducing signal because a year and sequence
alone also occur as page ranges. The patterns still admit filing captions,
ECF stamps and some bar-number lookalikes. They remain visible to raw locator
evaluation; an independent audit after colocation decides admission using an
explicit court or a colocated reporter.

A court is attached only when its own parenthetical follows the docket in the
same text block. Proximity across a paragraph break is insufficient. A docket
without a resolved court gets an internal per-occurrence eyecite edition, so
unknown jurisdictions cannot be merged before a validation check establishes
their identity.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import cache, lru_cache

import ahocorasick
from courts_db import courts
from eyecite.models import CitationToken, Edition, Reporter, TokenExtractor
from eyecite.tokenizers import Tokenizer, default_tokenizer

from mellea_lrc.extraction.reading.courts import resolve_court

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
_REQUIRED_SIGNAL = (
    r"(?:\b(?:Case|Civil[^\S\r\n]+Action|Civ\.?[^\S\r\n]*A\.?|Docket)?[^\S\r\n]*"
    r"No\.?[^\S\r\n]*:?[^\S\r\n]*|\bCase[^\S\r\n]+)"
)
_SIGNAL = rf"{_REQUIRED_SIGNAL}?"

# The district shape: an optional office, then year, case type and sequence.
# Self-identifying enough that the signal in front of it is optional.
#
# The office is optional because most courts do not write one. `No. 22-cv-1231`
# (W.D. Wash.), `No. 24-cv-8760` (S.D.N.Y.) and `No. 25-CV-2463` (D.D.C.) are
# the ordinary form, and requiring `1:` in front cost eleven citations on this
# corpus -- and cost them twice over, because a docket number the extractor
# does not take is one the *next* citation's case-name search runs over:
# `Doe v. Amazon.com, Inc. , No. 22-cv-1231, 2023 WL 3568691` came back with the
# defendant as `Amazon.com, Inc. , No. 22-cv-1231`.
_DISTRICT = (
    r"\b(?:(?P<office>\d{1,2}):)?(?P<year>\d{2})"
    rf"{_JOIN}(?P<case_type>" + "|".join(CASE_TYPES) + rf"){_JOIN}"
    r"(?P<sequence>\d{3,6})"
    r"(?P<suffix>(?:-[A-Za-z]{2,4})+)?\b"
)

# The bankruptcy shape: a year and a sequence, and nothing else. `No. 06-01147
# (JMP) (Bankr. S.D.N.Y. Jan. 18, 2006)` is how a bankruptcy court numbers a
# case, and documents 015 and 016 cite eighteen cases that way -- correctly,
# under Bluebook Rule 10.8.1, and every one of them was read as no citation at
# all.
#
# Nothing in the number itself says it is one. `1124201` is an attorney's bar
# number in document 005's signature block and `035547/2021` a state index
# number in document 009, and both are this shape. The signal is required here
# rather than optional, which reduces but does not eliminate collisions. The
# filing's own `Case No. 26-10769` and `No. 1124201` are intentionally kept as
# candidates; locator-layer evaluation measures their cost.
_BANKRUPTCY = rf"\b(?P<year>\d{{2}}){_JOIN}(?P<sequence>\d{{4,5}})(?P<suffix>(?:-[A-Za-z]{{2,4}})+)?\b"

# The group that marks a citation token as a docket rather than a reporter.
DOCKET_GROUP = "docket"

DOCKET_NUMBER = rf"{_SIGNAL}{_DISTRICT}"
"""A district docket number, with the optional signal that introduces it."""

BANKRUPTCY_DOCKET_NUMBER = rf"{_REQUIRED_SIGNAL}{_BANKRUPTCY}"
"""A bankruptcy docket number, which is only one where a signal introduces it."""

# A district docket's conventional fields are useful when they are present,
# but they are not a complete grammar for how filings cite a case.  Courts and
# counsel also write identifiers such as ``CIV 11-0107 JB/KBM`` and
# ``13CV04115WHODMR``.  Those forms have no reliable internal structure that
# distinguishes a docket from other text.  Their *citation position* does:
# they follow an introducing signal and are immediately followed by a database
# locator that eyecite also reads.  Courts commonly place one or more judge or
# chamber parentheticals between the number and that locator; they describe the
# case but are not part of its docket identifier.
#
# This deliberately does not make a bare ``No.`` phrase a docket.  The comma
# and database locator are part of the rule, so captions, ECF stamps, bar
# numbers, and prose still remain outside this broad path.  The identifier is
# kept as written, including ordinary spacing, slashes, and backslashes.  Its
# later admission and court resolution are independent passes.
POSITIONAL_DOCKET_NUMBER = rf"""
{_REQUIRED_SIGNAL}
(?P<{DOCKET_GROUP}>
    (?=[^,\r\n]{{0,80}}\d)
    [A-Za-z0-9]
    (?:[A-Za-z0-9:./\\-]|[^\S\r\n]+(?=[A-Za-z0-9]))*
)
(?=
    [^\S\r\n]*(?:\([^()\r\n]*\)[^\S\r\n]*)*
    ,[^\S\r\n]*\d{{4}}
    (?:[^\S\r\n]+[A-Za-z][A-Za-z.]*)*
    [^\S\r\n]+(?:WL|LEXIS)[^\S\r\n]+\d+\b
)
"""
"""A signaled identifier immediately before an eyecite-readable database locator."""

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
_COURT_NAMES = {str(court["id"]): court["name"] for court in courts}


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


def court_for_docket(text: str, end: int, *, stop: int | None = None) -> CourtCandidate | None:
    """The court written with the docket number that ends at ``end``.

    A cited docket carries its court in the parenthesis that follows it, in the
    same block of text. Nothing else counts: a court string merely nearby
    belongs to whatever citation put it there, which on a page of ECF stamps is
    never this one.
    """
    limit = min(end + _COURT_WINDOW, stop if stop is not None else len(text))
    for bracket in _BRACKETED.finditer(text, end, limit):
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
        if not _is_written_as_a_court(opening):
            continue
        entry = _COURT_TIGHT.get(_tight(opening))
        if entry is None:
            # courts-db spells some courts out where a filing abbreviates:
            # `Bankr. S.D. Florida` is stored and `Bankr. S.D. Fla.` is written.
            # `resolve_court` reads the abbreviation; it decides nothing this
            # index would have decided differently.
            resolved = resolve_court(opening)
            entry = (resolved, _COURT_NAMES[resolved]) if resolved else None
        if entry is None:
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
    """An internal edition used to keep eyecite from merging unknown dockets.

    Before court resolution, each occurrence gets a unique edition, so equal
    numbers in unknown jurisdictions cannot share an eyecite resource. Once a
    court is known, the later root stage groups equal docket numbers within it.
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

    The docket's office, year and sequence stand in for eyecite's volume and
    page fields, preserving the locator as a normal full-case token. The
    reporter field is an internal per-occurrence edition until court context is
    resolved later.

    The page deliberately keeps its hyphen. eyecite rejects an `Id.` whose pin
    cite cannot be a page within 150 pages of a numeric one, which is a sound
    rule for a reporter and a meaningless one for a docket -- and it is what
    would throw away the fifteen `Id. ¶ N` references that make this worth
    doing.
    """
    del extra
    groups = match.groupdict()
    positional_docket = groups.get(DOCKET_GROUP)
    if positional_docket is not None:
        # The positional form deliberately accepts identifiers whose internal
        # fields are unknown.  Eyecite still requires a volume and page to
        # construct a full-case token, so use an occurrence-local numeric
        # stand-in.  The actual locator is carried separately in DOCKET_GROUP.
        case_type = "docket"
        office = str(match.start() + offset + 1)
        page = office
        docket_number = positional_docket
    else:
        # A bankruptcy number carries neither an office nor a case type: it is
        # a year and a sequence. The year stands in for the office so eyecite
        # has a volume; the type is filled in after the court is resolved.
        case_type = (groups.get("case_type") or "bk").lower()
        office = groups.get("office") or groups["year"]
        begins = "office" if groups.get("office") else "year"
        page = f"{groups['year']}-{groups['sequence']}"
        docket_number = match.string[match.start(begins) : match.end()]
    # Court lookup runs after locator detection. Until it does, give each
    # courtless occurrence an internal unique edition so eyecite cannot merge
    # equal docket strings from unknown jurisdictions.
    court_id = f"unresolved-{match.start() + offset}"
    court_name = "Unresolved docket court"
    return CitationToken(
        match.group(0),
        match.start() + offset,
        match.end() + offset,
        groups={
            "volume": office,
            "reporter": f"{court_id} {case_type}",
            "page": page,
            DOCKET_GROUP: docket_number,
            "court": None,
            "court_name": None,
            "court_text": None,
        },
        exact_editions=(_edition(court_id, court_name, case_type),),
    )


class _DocketExtractor(TokenExtractor):
    """A token extractor that finds docket-shaped locators before context.

    The tokenizer recognizes the locator's shape. Whether that locator belongs
    to a cited case is left to locator evaluation and later validation.
    """

    def get_matches(self, text: str) -> list[re.Match[str]]:
        """Return syntactically valid docket numbers for later context resolution."""
        return super().get_matches(text)


class _PositionalDocketExtractor(_DocketExtractor):
    """Read broad docket forms only when the conventional grammars do not."""

    def get_matches(self, text: str) -> list[re.Match[str]]:
        """Avoid emitting a second token for a conventionally shaped docket."""
        known = tuple(
            re.compile(pattern, re.IGNORECASE)
            for pattern in (DOCKET_NUMBER, BANKRUPTCY_DOCKET_NUMBER)
        )
        return [
            match
            for match in super().get_matches(text)
            if not any(pattern.fullmatch(match.group(0)) for pattern in known)
        ]


@lru_cache(maxsize=1)
def docket_extractors() -> tuple[TokenExtractor, ...]:
    """The extractors that read docket numbers, to register with a tokenizer.

    ``strings`` is empty, which puts this among the extractors the ahocorasick
    prefilter always runs. There is no literal a docket number must contain --
    the case-type code is already in the pattern -- and one more regex per
    document is not a cost worth a prefilter.
    """
    conventional = tuple(
        _DocketExtractor(
            regex=pattern,
            constructor=docket_token,
            flags=re.IGNORECASE,
            strings=[],
        )
        # Two extractors rather than one alternation: eyecite compiles these
        # with the standard library's `re`, which will not let two branches of
        # a pattern name the same group, and both shapes have a year and a
        # sequence.
        for pattern in (DOCKET_NUMBER, BANKRUPTCY_DOCKET_NUMBER)
    )
    positional = _PositionalDocketExtractor(
        regex=POSITIONAL_DOCKET_NUMBER,
        constructor=docket_token,
        flags=re.IGNORECASE | re.VERBOSE,
        strings=[],
    )
    return (*conventional, positional)


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
