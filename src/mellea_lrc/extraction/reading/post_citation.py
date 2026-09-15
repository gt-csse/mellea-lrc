r"""Stop a citation reading the court and date of the citation after it.

When eyecite has read a citation's locator it scans forward for a pin cite,
court, date and parenthetical. That scan is not stopped by the next citation:
``add_post_citation`` calls ``match_on_tokens`` without ``strings_only``, so it
runs to a paragraph break or 300 characters, and the pattern's own leftover
group is written ``[^(;]*`` -- unbounded until the next bracket or semicolon.

A citation with no parenthetical of its own therefore walks forward and helps
itself to whatever the *next* citation says. In document 009,
``Koulkina, 2009 WL 2103627, at *3.`` stands two sentences before
``Spector v. Torenberg, 852 F. Supp. 201, 205 (S.D.N.Y. 1994)`` and comes back
carrying 1994 and ``nysd``. Twenty citations on ``false-citation-bench`` carry a
year belonging to another case: Iqbal reads 2001, Twombly reads 2001 in one
place and 2009 in another, ``137 S. Ct. 1285`` reads 1978.

That is worse than a blank. A missing year costs a check; a wrong one buys a
confident verdict about the wrong case.

## Why the boundary is co-location and not simply "the next citation"

A parallel citation is one decision printed in several reporters, written as a
comma-separated run with a single date at the end::

    St. Amant v. Thompson, 390 U.S. 727, 731, 88 S.Ct. 1323, 20 L.Ed.2d 262 (1968)

``390 U.S. 727`` *has* to read across the other two to find 1968, and 30
citations here do exactly that. So the scan stops at the next citation **that is
not co-located with this one** -- co-located meaning another identifier for the
same place in the text, which :mod:`mellea_lrc.extraction.structure.colocation` reports.

Measured over the corpus, that boundary gives::

    year     518 -> 511      7 lost, 23 corrected
    court    423 -> 422      1 lost

and every one of the eight losses was already wrong: each was taken from a later
citation, and in five of them the citation's own volume states the real year.
No correct value is lost.

## What this does not do

Only case citations. A statute and a journal article are read with different
post-citation patterns, and neither was measured here.

The date is re-read but never invented: this runs eyecite's own pattern over a
shorter window, so it can only find what eyecite would have found had it stopped
in the right place.
"""

from __future__ import annotations

from dataclasses import replace
from functools import cache, lru_cache
from typing import TYPE_CHECKING

import eyecite.regexes
import regex as re  # eyecite matches with this; its patterns repeat group names
from eyecite.helpers import MAX_MATCH_CHARS
from reporters_db import REPORTERS

from mellea_lrc.core.citations import CitationDate, FullCaseCitation
from mellea_lrc.core.spans import Span
from mellea_lrc.extraction.reading.courts import court_from_reporter, resolve_court
from mellea_lrc.extraction.reading.pin_cites import relax

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mellea_lrc.extraction.types import CitationRecord

# eyecite's own post-citation pattern, widened the way this project widens pin
# cites, and anchored the way `match_on_tokens` anchors it.
_POST_CITATION = re.compile(rf"^(?:{relax(eyecite.regexes.POST_FULL_CITATION_REGEX)})", re.X)


def _boundary(
    item: CitationRecord,
    citations: Sequence[CitationRecord],
    length: int,
) -> int:
    """Where this citation's forward scan should stop.

    The next citation that is not co-located with it, or eyecite's own 300
    character limit, whichever comes first.
    """
    starts = [
        other.locator_span.start
        for other in citations
        if other is not item
        and other.locator_span.start >= item.locator_span.end
        and not (item.colocation_id and other.colocation_id == item.colocation_id)
    ]
    return min([*starts, item.locator_span.end + MAX_MATCH_CHARS, length])


def reread_post_citation(
    text: str,
    citations: Sequence[CitationRecord],
) -> tuple[CitationRecord, ...]:
    """Re-read each case citation's court and date within its own boundary.

    ``citations`` must already carry their co-location ids, because the boundary
    is defined in terms of them.
    """
    rebuilt: list[CitationRecord] = []
    for item in citations:
        if not isinstance(item.stated, FullCaseCitation):
            rebuilt.append(item)
            continue
        stop = max(_boundary(item, citations, len(text)), item.locator_span.end)
        found = _POST_CITATION.match(text[item.locator_span.end : stop])
        year = found.group("year") if found else None
        date = (
            CitationDate(year=year, month=found.group("month"), day=found.group("day"))
            if found and year
            else None
        )
        court_text = (found.group("court") or "").strip() if found else ""
        # This is still the rules reading, so what it finds is what the citation
        # was read as: both sides of the record move together. A court the
        # *reporter* names is not this scan's to take away -- `556 U.S. 662` is
        # the Supreme Court's and `5 N.C. App. 10` the North Carolina Court of
        # Appeals' whatever the parenthetical says -- so where the parenthetical
        # names none, the reporter is asked. The span widens with the date,
        # because the parenthetical is part of the citation.
        read = replace(
            item.source,
            date=date,
            court=resolve_court(court_text) or _from_reporter(item.source),
            span=Span(
                start=item.full_span.start,
                end=item.locator_span.end + found.end() if found else item.locator_span.end,
            ),
        )
        rebuilt.append(replace(item, source=read, stated=read))
    return tuple(rebuilt)


@cache
def _entry(edition: str | None) -> dict | None:
    """What reporters-db knows about the reporter this edition belongs to."""
    if not edition:
        return None
    for entries in REPORTERS.values():
        for entry in entries:
            if edition in entry["editions"]:
                return entry
    return None


def _from_reporter(citation: object) -> str | None:
    """The court the citation's reporter names, where it names one."""
    reporter = getattr(citation, "reporter", None)
    edition = getattr(reporter, "short_name", None) or getattr(reporter, "as_written", None)
    entry = _entry(edition)
    return court_from_reporter(
        edition,
        cite_type=entry.get("cite_type") if entry else None,
        name=entry.get("name") if entry else None,
    )
