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
what a given abbreviation means. The index lives in
:mod:`mellea_lrc.extraction.reading.dockets`, which reads dockets deterministically;
this module and that one must agree about what a court is, or a site the
extractor declined would be offered to a model with a different set of
candidates.

The two do not share a docket *shape*, and deliberately. This one insists on
the `No.` that introduces a docket number, because a site here costs a model
call and a filing's own number appears in every ECF page stamp. The extractor
can afford the looser shape because it decides for itself, from the court
written alongside, whether what it found is a citation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from mellea_lrc.extraction.reading.dockets import CourtCandidate, courts_near

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

_CONTEXT = 170


@dataclass(frozen=True, slots=True)
class SuspectedDocket:
    """One docket-shaped string, with any court strings written near it."""

    span_start: int
    span_end: int
    docket_text: str
    courts: tuple[CourtCandidate, ...]
    window: str


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
                courts=courts_near(text, start, end),
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
