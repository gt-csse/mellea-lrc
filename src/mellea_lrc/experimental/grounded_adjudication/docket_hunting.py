"""Find docket-number citations and the court strings that identify them.

A docket number is not a locator. ``1:19-cv-362`` names a case only alongside
its court -- the same number exists in many districts -- so the identifying
pair is the docket and the court, and the two sit in different places in the
text::

    Calderon v. GEICO Gen. Ins. Co., No. 1:19-CV-362 (M.D.N.C. Jan. 26, 2021)
                                     ^docket          ^court

This module reports both, independently of any reporter locator at the same
position. A citation carrying both identifiers points at two different
databases -- RECAP for the docket, a reporter corpus for the locator -- which
carry different information. Deciding that the two denote one case is a later
service, and folding it in here would mean discarding one of them.

Court strings are recognised from ``courts-db``, the same database
CourtListener uses, rather than a hand-written pattern. That yields the court's
identifier and full name, which downstream can use as a cue instead of guessing
what a given abbreviation means.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import ahocorasick
from courts_db import courts

if TYPE_CHECKING:
    from mellea_lrc.extraction.types import ExtractedDocument

# "No. 1:19-CV-362", "Case No. 3:23-cv-06558", "Civil Action No. 2:25-cv-00804".
# The office/party suffix ("-RPK", "-PAB-SBP") is optional, and the separator
# after the year is allowed to be missing entirely: PDF extraction drops it, as
# in "No. 1:25cv-05745-RPK".
_DOCKET = re.compile(
    r"\b(?:No|Case No|Civil Action No|Civ\.? A\.? No|Docket No)\.?\s*"
    r"\d{1,2}[:\-]\d{2}[-\s]?[a-zA-Z]{2,4}[-\s]?\d{2,6}(?:-[A-Za-z]{2,4})*",
    re.I,
)

_COURT_SEARCH_BEFORE = 90
_COURT_SEARCH_AFTER = 140
_CONTEXT = 170
_MIN_COURT_STRING = 4


@dataclass(frozen=True, slots=True)
class CourtCandidate:
    """One court string found near a docket number, resolved against courts-db."""

    span_start: int
    span_end: int
    text: str
    court_id: str
    court_name: str


@dataclass(frozen=True, slots=True)
class SuspectedDocket:
    """One docket-shaped string, with any court strings written near it."""

    span_start: int
    span_end: int
    docket_text: str
    courts: tuple[CourtCandidate, ...]
    window: str


def _normalize(value: str) -> str:
    """Compare court strings ignoring case and trailing punctuation.

    courts-db is not internally consistent about the final period -- the Eastern
    District of New York is stored as ``E.D.N.Y`` while the Southern District is
    ``S.D.N.Y.`` -- so an exact match would silently miss whole courts.
    """
    return value.casefold().rstrip(". ")


def _build_court_index() -> tuple[ahocorasick.Automaton, dict[str, tuple[str, str]]]:
    """Index every court citation string courts-db knows."""
    lookup: dict[str, tuple[str, str]] = {}
    for court in courts:
        citation_string = court.get("citation_string")
        if not citation_string or len(citation_string) < _MIN_COURT_STRING:
            continue
        lookup.setdefault(_normalize(citation_string), (court["id"], court["name"]))
    automaton = ahocorasick.Automaton()
    for normalized in lookup:
        automaton.add_word(normalized, normalized)
    automaton.make_automaton()
    return automaton, lookup


_COURT_AUTOMATON, _COURT_LOOKUP = _build_court_index()


def _courts_near(text: str, start: int, end: int) -> tuple[CourtCandidate, ...]:
    """Return court strings written close enough to identify this docket."""
    left = max(0, start - _COURT_SEARCH_BEFORE)
    region = text[left : end + _COURT_SEARCH_AFTER]
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


def suspected_dockets(document: ExtractedDocument) -> tuple[SuspectedDocket, ...]:
    """Report every docket-shaped string, with the courts written near it."""
    text = document.text
    sites: list[SuspectedDocket] = []
    for match in _DOCKET.finditer(text):
        start, end = match.span()
        sites.append(
            SuspectedDocket(
                span_start=start,
                span_end=end,
                docket_text=match.group(0),
                courts=_courts_near(text, start, end),
                window=text[max(0, start - _CONTEXT) : end + _CONTEXT],
            )
        )
    return tuple(sites)


def docket_context(site: SuspectedDocket) -> str:
    """Describe the courts found near a docket, for use as a prompt cue.

    Naming the candidates and what they resolve to spares the model from
    inferring that ``M.D.N.C.`` means the Middle District of North Carolina,
    and keeps it from inventing a court that is not written down.
    """
    if not site.courts:
        return (
            "No court string was found near this docket number. A docket number "
            "identifies a case only together with its court, so if no court is "
            "written in the window, report the court as null."
        )
    described = "; ".join(f'"{court.text}" is {court.court_name}' for court in site.courts)
    return (
        f"Court strings written near this docket number: {described}. "
        f"Use one of these exactly as written, or null if none of them is the "
        f"court of the case this docket number belongs to."
    )
