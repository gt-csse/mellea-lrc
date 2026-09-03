"""Find positions that look like citation locators but produced no citation.

eyecite reports a citation only when it can also parse one, so a locator it
cannot parse leaves no trace at all. This module recovers those positions
without parsing anything: it masks what was already extracted, then scans the
remainder for a reporter string from eyecite's own gazetteer sitting in
locator-like company.

The result is a *candidate*, not a citation. Deciding whether a candidate is
real -- and recovering its fields when it is -- is left to a downstream judge.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import ahocorasick
from eyecite.tokenizers import EXTRACTORS

from mellea_lrc.adjudication.masking import mask_full_spans

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


@dataclass(frozen=True, slots=True)
class SuspectedLocator:
    """One position where a reporter appears but no citation was produced."""

    span_start: int
    span_end: int
    reporter: str
    window: str


def _gazetteer() -> ahocorasick.Automaton:
    """Build a matcher over every reporter string eyecite knows."""
    automaton = ahocorasick.Automaton()
    for string in {s for e in EXTRACTORS for s in e.strings if len(s) >= _MIN_GAZETTEER_LENGTH}:
        automaton.add_word(string, string)
    automaton.make_automaton()
    return automaton


_AUTOMATON = _gazetteer()


def _maximal_hits(text: str) -> list[tuple[int, int, str]]:
    """Return gazetteer hits, dropping any contained within a longer one.

    ``F.`` matches inside ``F. Supp. 3d``; only the longest reading at a
    position is a plausible reporter.
    """
    hits = [(end - len(s) + 1, end + 1, s) for end, s in _AUTOMATON.iter(text)]
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


def suspected_locators(document: ExtractedDocument) -> tuple[SuspectedLocator, ...]:
    """Report reporter occurrences that look like locators but were not parsed.

    A position qualifies when the reporter string is not embedded in a word and
    has a digit within a short window on both sides -- the volume-and-page
    shape. Both tests are deliberately cheap: this is a recall-oriented filter
    whose output a judge is expected to reject freely.
    """
    masked = mask_full_spans(document)
    sites: list[SuspectedLocator] = []
    for start, end, reporter in _maximal_hits(masked):
        before = masked[start - 1] if start else " "
        after = masked[end] if end < len(masked) else " "
        if before.isalnum() or after.isalpha():
            continue
        has_volume = _DIGIT.search(masked[max(0, start - _DIGIT_WINDOW) : start])
        ahead = masked[end : end + _DIGIT_WINDOW]
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
        sites.append(
            SuspectedLocator(
                span_start=start,
                span_end=end,
                reporter=reporter,
                # Context comes from the original text, not the masked copy: a
                # judge should read what is actually written at this position.
                window=document.text[max(0, start - _CONTEXT) : end + _CONTEXT // 2],
            )
        )
    return tuple(sites)
