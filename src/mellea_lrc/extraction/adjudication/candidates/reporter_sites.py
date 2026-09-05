r"""Find positions that look like citation locators but produced no citation.

eyecite reports a citation only when it can also parse one, so a locator it
cannot parse leaves no trace at all. This module recovers those positions
without parsing anything: it masks what was already extracted, then scans the
remainder for a reporter in locator-like company.

The result is a *candidate*, not a citation. Deciding whether a candidate is
real -- and recovering its fields when it is -- is left to a downstream judge.

**Two stages, because "a reporter is there" has two answers.**

:attr:`SiteStage.STRICT`
    The reporter is written as eyecite's gazetteer spells it. This is the whole
    of what an exact matcher can do, and on clean text it is almost all of the
    yield.

:attr:`SiteStage.FUZZY`
    Something with a number on either side reduces to a reporter once
    punctuation is removed. ``550 US 544`` and ``556 U,S, 662`` are invisible to
    the strict stage -- the damage is *inside* the reporter, and no gazetteer
    string matches -- but a number, a short letter run and a number is a shape,
    and the letters are ``U.S.`` once the periods stop mattering.

    The number on either side is what makes this affordable. Without it the
    pattern matches most of a page of prose; with it, the stage proposes tens of
    sites per corpus rather than thousands.

**Optical damage and capitalisation apply to both stages**, because they are
properties of the text and not of a matching strategy. Capitalisation is folded
away by scanning a lower-cased copy; optical damage is folded away by searching
for each reporter's damaged spellings as well as its own
(:mod:`~mellea_lrc.extraction.adjudication.ocr`). Both copies preserve length, so
every offset reported here indexes the original document.

Folding capitals in subsumes the former ``uppercase_reporters`` generator, which
proposed the same spans for the same reason and reached them by a separate
route; a span proposed twice is reviewed twice and reported twice.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import TYPE_CHECKING

import ahocorasick
from eyecite.tokenizers import EDITIONS_LOOKUP, EXTRACTORS

from mellea_lrc.extraction.adjudication import ocr
from mellea_lrc.extraction.adjudication.masking import mask_full_spans, mask_locator_spans

if TYPE_CHECKING:
    from mellea_lrc.extraction.types import ExtractedDocument

_MIN_GAZETTEER_LENGTH = 2
_DIGIT_WINDOW = 12
_CONTEXT = 170
_DIGIT = re.compile(r"\d")
# A section sign between the reporter and the number means the number is a
# statute section, not a page: `28 U.S.C. § 1927` has the volume-and-page shape
# and is not a case. The mark is never skipped over as punctuation -- it is the
# evidence that this site is out of scope, and a site sent to a judge as a
# suspected case locator is a statute offered up for a case-law verdict.
_SECTION = re.compile(r"§")
_MAX_NESTED_LOOKBACK = 6

# The fuzzy net: a number, a short letter run, a number. Interior digits are
# allowed only in front of a letter, which is how a series is written -- `F.2d`,
# `Cal. App. 4th` -- and which stops the run from swallowing the page.
#
# It is matched at every number in the text rather than scanned with
# ``finditer``, because the readings overlap and only one of them is the
# citation. In "2. See Ashcroft v. Iqbal, 556 U,S, 662" a scan takes the first
# match it can -- volume "2", letters "See Ashcroft v. Iqbal", page "556" --
# and resumes past the end of it, so the locator is never offered at all.
# Anchoring at each number tries "556" too, and the gazetteer throws the other
# reading away.
_FUZZY_SITE = re.compile(
    r"(\d{1,4})"
    r"([^0-9A-Za-z]{0,4})"
    r"([A-Za-z](?:[A-Za-z.,'‘’ \-&]|\d(?=[A-Za-z])){0,22}[A-Za-z.])"
    r"([^0-9A-Za-z]{0,4})"
    r"(\d{1,6})(?!\d)"
)
_NUMBER_START = re.compile(r"(?<!\d)\d")
_MIN_FUZZY_KEY = 2
_SIMILARITY = 0.9
_MIN_SIMILAR_KEY = 8
"""A near match is only allowed on a long key, and the bar is high.

Approximate matching is the one part of this module that guesses, and short
keys make it guess constantly: at four characters `Case` is within one edit of
`Chase`, `Page` of `Paige`, and `after` of `A.F.T.R.`, all of which it duly
proposed. Nothing shorter than eight characters is close to anything by
accident, and the near matches that survive -- `Fordham Intl. L.J.` for
`Fordham Int'l L.J.` -- are the reporters this was for.
"""

_MIN_UNPUNCTUATED_KEY = 4
"""Below this length, a site must be written like an abbreviation.

The gazetteer holds `At.`, `Or`, `No.` and `Ms.`, which are also ordinary
English words, and folding capitals away makes every occurrence of `at` in
running prose a reporter with a number on either side -- `Doc. 174 at 5` reads
as volume 174, reporter `At.`, page 5. Requiring a full stop, or every letter in
capitals, keeps `U.S.`, `US`, `U,S,` and `F.2d` and drops the prose. It is only
applied to short keys because that is where the collisions are: no ordinary word
reduces to `calrptr3d`.
"""


class SiteStage(str, Enum):
    """How the reporter at a site was recognised."""

    STRICT = "strict"
    """Written as the gazetteer spells it, up to capitalisation and one
    optically confused character."""

    FUZZY = "fuzzy"
    """Reduces to a gazetteer spelling once punctuation is discarded, and has a
    number on either side."""


@dataclass(frozen=True, slots=True)
class SuspectedLocator:
    """One position where a reporter appears but no citation was produced."""

    span_start: int
    span_end: int
    reporter: str
    """The characters actually written at the site, damage and all."""

    window: str
    stage: SiteStage = SiteStage.STRICT
    matched_reporter: str = ""
    """The gazetteer spelling the site was recognised as, when it differs.

    Empty when the site is written exactly as the gazetteer spells it. A
    reviewer is told about *this* reporter, not the damaged characters, because
    a description built from the damage says only that the database has never
    heard of it.
    """

    note: str = ""

    @property
    def canonical_reporter(self) -> str:
        """The reporter to describe to a reader."""
        return self.matched_reporter or self.reporter


def _spellings() -> set[str]:
    return {s for e in EXTRACTORS for s in e.strings if len(s) >= _MIN_GAZETTEER_LENGTH}


@lru_cache(maxsize=1)
def _strict_automaton() -> tuple[ahocorasick.Automaton, dict[str, str]]:
    """A matcher over every reporter spelling, lower-cased and optically damaged.

    The payload maps each searched form back to the gazetteer spelling it stands
    for, so a hit on ``s0. 2d`` can be reported as ``So. 2d``.
    """
    canonical: dict[str, str] = {}
    for spelling in sorted(_spellings()):
        for variant in ocr.variants(spelling):
            canonical.setdefault(variant, spelling)
    automaton = ahocorasick.Automaton()
    for variant in canonical:
        automaton.add_word(variant, variant)
    automaton.make_automaton()
    return automaton, canonical


@lru_cache(maxsize=1)
def _loose_gazetteer() -> dict[str, str]:
    """Every reporter keyed by its letters and digits alone.

    Editions rather than the reporters index, because that is the level at which
    a citation names one. Where two spellings share a key the shorter wins: it
    is the abbreviation, and the longer is usually the same reporter with a
    trailing comma or an expanded series.
    """
    keys: dict[str, str] = {}
    for spelling in sorted(_spellings(), key=lambda s: (len(s), s)):
        reduced = ocr.key(spelling)
        if len(reduced) >= _MIN_FUZZY_KEY:
            keys.setdefault(reduced, spelling)
    return keys


def _maximal_hits(text: str) -> list[tuple[int, int, str]]:
    """Return gazetteer hits, dropping any contained within a longer one.

    ``F.`` matches inside ``F. Supp. 3d``; only the longest reading at a
    position is a plausible reporter.
    """
    automaton, _ = _strict_automaton()
    hits = [(end - len(s) + 1, end + 1, s) for end, s in automaton.iter(text)]
    hits.sort(key=lambda hit: (hit[0], -(hit[1] - hit[0])))
    kept: list[tuple[int, int, str]] = []
    for hit in hits:
        contained = any(
            other[0] <= hit[0] and hit[1] <= other[1] and (other[1] - other[0]) > (hit[1] - hit[0])
            for other in kept[-_MAX_NESTED_LOOKBACK:]
        )
        if not contained:
            kept.append(hit)
    return kept


# Number-string-number shapes that are explicably not a locator. The fuzzy net
# is broad enough that these are most of what it catches, and each site it
# proposes costs a reader a call, so they are dropped rather than labelled.
_YEAR = re.compile(r"^(1[6-9]|20)\d\d$")
_SERIES_AHEAD = re.compile(r"^(d|th|st|nd|rd)\b")
_PIN_CITE_AHEAD = re.compile(r"^\s*(at|§)\b")
_MONTHS = frozenset(
    "january february march april may june july august september october november december "
    "jan feb mar apr jun jul aug sept sep oct nov dec".split()
)
_STATUTE_KEYS = frozenset({"u5c", "u5ca", "cfr", "u5c5"})


def _explained_away(text: str, match: re.Match[str], letters: str) -> str:
    """Why this number-string-number is not a citation, when it plainly is not."""
    start, end = match.start(), match.end()
    before = text[max(0, start - 2) : start]
    after = text[end : end + 2]
    if letters in _MONTHS:
        return "a date"
    if _YEAR.match(match.group(5)):
        if "(" in before or ")" in after or "(" in match.group(2):
            return "a court parenthetical"
        return "ends in a year"
    if "at" in match.group(4).lower() or text[max(0, start - 4) : start].lower().endswith("at "):
        return "a short-form pin cite"
    if ocr.key(letters) in _STATUTE_KEYS or "§" in text[end : end + 4]:
        return "a statute or regulation"
    # `529 F.3d at 935` closes early at the series digit, reporting a page of 3.
    # Anything immediately followed by a series letter is that, not a locator.
    if _SERIES_AHEAD.match(text[end : end + 3]):
        return "cut short inside the reporter"
    if _PIN_CITE_AHEAD.match(text[end : end + 6]):
        return "a short-form pin cite"
    if "@" in text[max(0, start - 30) : end + 30] or "Suite" in text[max(0, start - 40) : end + 10]:
        return "an address or contact block"
    return ""


def _abbreviation(written: str) -> bool:
    """Whether the written form looks like an abbreviation rather than a word."""
    letters = [character for character in written if character.isalpha()]
    return "." in written or (bool(letters) and all(character.isupper() for character in letters))


def _capitalised(written: str, spelling: str) -> bool:
    """Whether a site that is not written as the gazetteer spells it is credible.

    Folding capitals away is what reaches ``33 F.4TH 693`` and ``416 U.s. 232``,
    and it is also what makes ``[Doc. 40, p. 8]`` a site: ``p.`` differs from
    the Pacific Reporter's ``P.`` in nothing but case, and a filing citing its
    own exhibits by page produced thirty of them. The same fold turns every
    sentence-initial ``Citing`` into the gazetteer's lower-case ``citing``.

    Both are ruled out by one observation: a reporter abbreviation is
    capitalised. So a site written differently from the spelling it matched must
    carry a capital, and the spelling it matched must carry one too -- otherwise
    the difference is ordinary sentence capitalisation of an ordinary word,
    which is not damage.

    The cost is a locator whose reporter is written entirely in lower case
    (``29 ny3d 425`` appears once across both corpora). That is one site against
    thirty-five, and the fuzzy stage does not reach it either.
    """
    if written == spelling:
        return True
    return any(c.isupper() for c in written) and any(c.isupper() for c in spelling)


def _resembles(reduced: str) -> tuple[str, str] | None:
    """The gazetteer spelling a reduced key names, and how it was reached."""
    gazetteer = _loose_gazetteer()
    exact = gazetteer.get(reduced)
    if exact is not None:
        return exact, "punctuation"
    if len(reduced) < _MIN_SIMILAR_KEY:
        return None
    close = difflib.get_close_matches(reduced, gazetteer, n=1, cutoff=_SIMILARITY)
    if not close:
        return None
    return gazetteer[close[0]], "similar"


def _strict_sites(document: ExtractedDocument, masked: str, reading: str) -> list[SuspectedLocator]:
    """Reporter spellings the gazetteer holds, standing in locator-like company.

    A position qualifies when the reporter string is not embedded in a word and
    has a digit within a short window on both sides -- the volume-and-page
    shape. Both tests are deliberately cheap: this is a recall-oriented filter
    whose output a judge is expected to reject freely.
    """
    _, canonical = _strict_automaton()
    lowered = ocr.lower(masked)
    sites: list[SuspectedLocator] = []
    for start, end, variant in _maximal_hits(lowered):
        before = lowered[start - 1] if start else " "
        after = lowered[end] if end < len(lowered) else " "
        if before.isalnum() or after.isalpha():
            continue
        has_volume = _DIGIT.search(lowered[max(0, start - _DIGIT_WINDOW) : start])
        ahead = lowered[end : end + _DIGIT_WINDOW]
        has_page = _DIGIT.search(ahead)
        if not (has_volume and has_page):
            continue
        # The section sign is read from the original text, not the masked copy.
        # eyecite emits a token of its own for a bare `§`, so masking blanks the
        # one character that says this number is a section rather than a page --
        # and the site would then be offered to a judge as a case locator.
        written = document.text[end : end + _DIGIT_WINDOW]
        if _SECTION.search(written[: has_page.start()]):
            continue
        spelling = canonical[variant]
        as_written = document.text[start:end]
        if len(ocr.key(as_written)) < _MIN_UNPUNCTUATED_KEY and not _abbreviation(as_written):
            continue
        if not _capitalised(as_written, spelling):
            continue
        sites.append(
            SuspectedLocator(
                span_start=start,
                span_end=end,
                reporter=as_written,
                # From the masked copy, not the original. The site itself is
                # never masked -- it produced no citation, which is why it is a
                # candidate -- so what a reviewer reads at this position is
                # exactly what is written there. What is hidden is every
                # *other* citation in the window, which has already been read
                # and is not the question being asked. Left visible, a reviewer
                # quotes one of them and returns a locator the record already
                # holds.
                window=reading[max(0, start - _CONTEXT) : end + _CONTEXT // 2],
                stage=SiteStage.STRICT,
                matched_reporter="" if as_written == spelling else spelling,
                note="" if as_written == spelling else f"written {as_written!r} for {spelling!r}",
            )
        )
    return sites


def _fuzzy_sites(document: ExtractedDocument, masked: str, reading: str) -> list[SuspectedLocator]:
    """Number-letters-number runs whose letters reduce to a reporter."""
    sites: list[SuspectedLocator] = []
    for anchor in _NUMBER_START.finditer(masked):
        match = _FUZZY_SITE.match(masked, anchor.start())
        if match is None:
            continue
        letters = match.group(3)
        reduced = ocr.key(letters)
        if len(reduced) < _MIN_FUZZY_KEY:
            continue
        if len(reduced) < _MIN_UNPUNCTUATED_KEY and not _abbreviation(letters):
            continue
        resembles = _resembles(reduced)
        if resembles is None:
            continue
        spelling, how = resembles
        if not _capitalised(letters, spelling):
            continue
        if _explained_away(masked, match, ocr.lower(letters).strip(" .,'‘’-&")):
            continue
        start, end = match.start(3), match.end(3)
        if _SECTION.search(document.text[end : match.end()]):
            continue
        sites.append(
            SuspectedLocator(
                span_start=start,
                span_end=end,
                reporter=letters,
                window=reading[max(0, match.start() - _CONTEXT) : match.end() + _CONTEXT // 2],
                stage=SiteStage.FUZZY,
                matched_reporter=spelling,
                note=f"{letters!r} reads as {spelling!r} ({how}), with a number on either side",
            )
        )
    return sites


def suspected_locators(document: ExtractedDocument) -> tuple[SuspectedLocator, ...]:
    """Report reporter occurrences that look like locators but were not parsed.

    Strict sites first, then fuzzy ones that do not overlap a strict site: a
    position the gazetteer reaches exactly needs no guess about what it says.
    """
    masked = mask_full_spans(document)
    # Two masks, for two different jobs. Hits are found in the full-span copy,
    # so a reporter inside a citation already read is not flagged again. The
    # window a reviewer reads blanks only the **locators**, because a locator
    # quote needs a volume, a reporter and a page, and with those characters
    # gone a neighbour cannot be quoted -- while its party names and the prose
    # around it stay, and, more importantly, so does any citation that was never
    # read. Masking full spans here would hide those: an over-reaching full span
    # covers text that is not a citation at all, and at `Relaxation.NONE` one
    # citation's span can cover the entire sentence.
    reading = mask_locator_spans(document)
    sites = _strict_sites(document, masked, reading)
    taken = [(site.span_start, site.span_end) for site in sites]
    for site in _fuzzy_sites(document, masked, reading):
        if any(start < site.span_end and site.span_start < end for start, end in taken):
            continue
        sites.append(site)
    return tuple(sorted(sites, key=lambda site: site.span_start))


def known_reporter(spelling: str) -> bool:
    """Whether eyecite's reporter database holds this spelling."""
    return spelling in EDITIONS_LOOKUP
