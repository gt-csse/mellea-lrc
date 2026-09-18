r"""Read generic docket locators before later root-stage audits.

A docket number is an opaque court-local identifier, not a reporter and not a
federal case-number grammar.  The stable tokenizer therefore reads one broad,
signaled identifier envelope and preserves its exact span.  It can include
filing captions and ECF page stamps; that is intentional.  Colocation is built
next, then the independent docket audit admits a locator only when an explicit
court or colocated reporter supports it.  Metrics score those audit-admitted
docket locators, never raw candidates.

A future site hunter may propose docket-shaped text that lacks an introducing
signal.  It is deliberately unavailable until it has an independent candidate
and review contract; it must not silently create roots through a second,
duplicate docket grammar.

Court and date readers run only after colocation and audit.  An admitted docket
without a written court remains courtless, so validation can first check its
colocated reporter before attempting a docket lookup.
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

# The group that marks a citation token as a docket rather than a reporter.
DOCKET_GROUP = "docket"

# A docket identifier is an opaque, court-specific string. It is not a federal
# case-number grammar: filings legitimately write compact identifiers such as
# ``13CV04115WHODMR``, chamber suffixes, slashes, backslashes, and forms this
# project has not seen yet.  The stable reader asks only for an introducing
# signal, a digit, and an identifier envelope.  Court and reporter context are
# deliberately *not* part of this rule: the audit after colocation decides
# whether this raw locator is actually citing a case.
#
# An envelope has one compact first token and up to two compact continuation
# tokens.  A continuation must carry a digit, separator punctuation, or an
# all-caps designator.  That retains ``CIV 11-0107 JB/KBM`` and ``CIV. A.
# 08-222-KD-B`` while stopping before ECF furniture such as ``Document 1``.
# The rule neither encodes a court's case-number convention nor repairs text.
_REQUIRED_SIGNAL = (
    r"(?:"
    r"\b(?:Case|Civil[^\S\r\n]+Action|Civ\.?[^\S\r\n]*A\.?|Docket)"
    r"[^\S\r\n]*No(?=[.^\s:]|$)[^\S\r\n]*\.?[^\S\r\n]*:?[^\S\r\n]*"
    r"|(?<!\.\s)(?<!\.)\bNo(?=[.^\s:]|$)[^\S\r\n]*\.?[^\S\r\n]*:?[^\S\r\n]*"
    r"|\bCase[^\S\r\n]+"
    r")"
)

_COMPACT_DOCKET_TOKEN = r"[A-Za-z0-9](?:[A-Za-z0-9:./\\-]*[A-Za-z0-9])?\.?"
# A number this short has no identifying weight by itself.  A retained raw
# candidate contains either a four-digit run or a structural separator.  That
# rejects ``D.I. No. 17`` while admitting court-local forms such as ``21-11854``
# and ``1:25-cv-00312`` without naming their individual case-type grammars.
_IDENTIFIER_EVIDENCE = r"(?=[^,;()\r\n]{0,40}(?:\d{4}|[:/\\-]))"
# Filings may omit ``No.`` in a table of authorities.  A signal-free identifier
# is read only where a following database citation supplies local citation
# context.  The date/database pattern belongs to the context, not a court's
# docket syntax, so this remains one general reader rather than a second list
# of court-specific grammars.
_DATABASE_CITATION = (
    r"\d{4}(?:[^\S\r\n]+[A-Za-z][A-Za-z.]*)*"
    r"[^\S\r\n]+(?:WL|LEXIS)[^\S\r\n]+\d+\b"
)
_SIGNAL_FREE_DOCKET = r"(?:\d{2}-(?:[A-Za-z]{1,6}-)?\d{3,}|\d{1,2}:\d{2}-[A-Za-z]{2,6})"
_SIGNAL_FREE_CONTEXT = rf"(?={_SIGNAL_FREE_DOCKET}(?:,[^\S\r\n]*|[^\S\r\n]+){_DATABASE_CITATION})"
_DOCKET_PREFIX = rf"(?:{_REQUIRED_SIGNAL}|{_SIGNAL_FREE_CONTEXT}|(?=\d{{1,2}}:\d{{2}}-[A-Za-z]{{2,6}}))"
_COMPACT_CONTINUATION = (
    r"(?!" + _DATABASE_CITATION + r")"
    r"(?=[A-Za-z0-9:./\\-]*(?:\d|[:./\\-])|(?-i:[A-Z.]{1,6})\b)"
    + _COMPACT_DOCKET_TOKEN
)
_CONTINUATION_JOIN = r"(?:[^\S\r\n]+|-[^\S\r\n]+)"

DOCKET_NUMBER = rf"""
{_DOCKET_PREFIX}
(?P<{DOCKET_GROUP}>
    {_IDENTIFIER_EVIDENCE}
    (?=
        (?:[A-Za-z0-9:./\\-]*\d)
      |
        (?-i:[A-Z.]{{1,6}}){_CONTINUATION_JOIN}
        (?:
            [A-Za-z0-9:./\\-]*\d
          | (?-i:[A-Z.]{{1,6}}){_CONTINUATION_JOIN}[A-Za-z0-9:./\\-]*\d
        )
    )
    {_COMPACT_DOCKET_TOKEN}
    (?:{_CONTINUATION_JOIN}{_COMPACT_CONTINUATION}){{0,2}}
)
(?![A-Za-z0-9:./\\-])
"""
"""A signaled opaque docket identifier, before court/context audit."""

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
    """Build one opaque docket locator as a full-case citation token.

    Eyecite requires numeric volume and page fields even though a docket has
    neither.  Per-occurrence stand-ins satisfy that transport requirement; the
    text in ``DOCKET_GROUP`` remains the only actual identifier.  Court lookup
    happens after locator detection, so the internal edition is unique until a
    later reader supplies a court.
    """
    del extra
    docket_number = match.group(DOCKET_GROUP)
    token_position = str(match.start() + offset + 1)
    court_id = f"unresolved-{match.start() + offset}"
    court_name = "Unresolved docket court"
    return CitationToken(
        match.group(0),
        match.start() + offset,
        match.end() + offset,
        groups={
            "volume": token_position,
            "reporter": f"{court_id} docket",
            "page": token_position,
            DOCKET_GROUP: docket_number,
            "court": None,
            "court_name": None,
            "court_text": None,
        },
        exact_editions=(_edition(court_id, court_name, "docket"),),
    )


@lru_cache(maxsize=1)
def docket_extractors() -> tuple[TokenExtractor, ...]:
    """Return the one contextual docket reader registered with eyecite.

    There is no literal every docket contains, so the tokenizer must run this
    regular expression directly.  The reader intentionally has one grammar:
    docket syntax belongs to courts, while citation context is the general fact
    extraction can establish without knowing a court's local numbering scheme.
    """
    return (
        TokenExtractor(
            regex=DOCKET_NUMBER,
            constructor=docket_token,
            flags=re.IGNORECASE | re.VERBOSE,
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
