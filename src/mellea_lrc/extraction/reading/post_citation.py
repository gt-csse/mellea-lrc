r"""Resolve court and date context after locator and colocation spans exist.

The two readers use eyecite's post-citation pattern inside the next unrelated
locator boundary. A parallel citation may read through its co-located neighbors
to the shared date; it stops before any other citation. Court and date are
separate passes so each can be inspected or replaced independently.

The docket audit runs independently before these readers. Court resolution
scans again to write metadata; a docket admitted through a colocated reporter
can still have court=None. Raw locator discovery remains independent of both.
"""

from __future__ import annotations

from dataclasses import replace
from functools import cache, lru_cache
from typing import TYPE_CHECKING

import eyecite.regexes
import regex as re  # eyecite matches with this; its patterns repeat group names
from eyecite.helpers import MAX_MATCH_CHARS
from reporters_db import REPORTERS

from mellea_lrc.core.citations import CitationDate, DocketCitation, FullCaseCitation
from mellea_lrc.core.spans import Span
from mellea_lrc.extraction.reading.courts import court_from_reporter, resolve_court
from mellea_lrc.extraction.reading.dockets import CourtCandidate, court_for_docket
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
        and not other.withdrawn
        and other.locator_span.start >= item.locator_span.end
        and not (item.colocation_id and other.colocation_id == item.colocation_id)
    ]
    return min([*starts, item.locator_span.end + MAX_MATCH_CHARS, length])


def _post_match(text: str, item: CitationRecord, citations: Sequence[CitationRecord]):
    """The bounded post-locator match shared by court and date readers."""
    stop = max(_boundary(item, citations, len(text)), item.locator_span.end)
    return _POST_CITATION.match(text[item.locator_span.end : stop])


def docket_court(
    text: str, item: CitationRecord, citations: Sequence[CitationRecord]
) -> CourtCandidate | None:
    """Read an explicit court after the whole group, before the next locator.

    Both the audit and the court-writing pass invoke this rule independently.
    Reporter-inferred courts are never treated as explicit docket courts.
    """
    members = [
        other
        for other in citations
        if not other.withdrawn
        and (other is item or (item.colocation_id and other.colocation_id == item.colocation_id))
    ]
    last = max(members or [item], key=lambda other: other.locator_span.end)
    return court_for_docket(text, last.locator_span.end, stop=_boundary(last, citations, len(text)))


def reread_courts(
    text: str,
    citations: Sequence[CitationRecord],
) -> tuple[CitationRecord, ...]:
    """Resolve courts after locator spans and co-location have been established."""
    rebuilt: list[CitationRecord] = []
    for item in citations:
        if item.withdrawn:
            rebuilt.append(item)
            continue
        if isinstance(item.stated, DocketCitation):
            candidate = docket_court(text, item, citations)
            if candidate is None:
                rebuilt.append(item)
                continue
            source = replace(
                item.source,
                court=candidate.court_id,
                court_name=candidate.court_name,
                court_text=candidate.text,
            )
            stated = replace(
                item.stated,
                court=candidate.court_id,
                court_name=candidate.court_name,
                court_text=candidate.text,
            )
            rebuilt.append(replace(item, source=source, stated=stated))
            continue
        if not isinstance(item.stated, FullCaseCitation):
            rebuilt.append(item)
            continue
        found = _post_match(text, item, citations)
        court_text = (found.group("court") or "").strip() if found else ""
        court = resolve_court(court_text) or _from_reporter(item.source)
        source = replace(item.source, court=court)
        stated = replace(item.stated, court=court)
        rebuilt.append(replace(item, source=source, stated=stated))
    return tuple(rebuilt)


def reread_dates(
    text: str,
    citations: Sequence[CitationRecord],
) -> tuple[CitationRecord, ...]:
    """Resolve dates after locator spans, bounded by co-location groups."""
    rebuilt: list[CitationRecord] = []
    for item in citations:
        if item.withdrawn or not isinstance(item.stated, (FullCaseCitation, DocketCitation)):
            rebuilt.append(item)
            continue
        found = _post_match(text, item, citations)
        year = found.group("year") if found else None
        date = (
            CitationDate(year=year, month=found.group("month"), day=found.group("day"))
            if found and year
            else None
        )
        span = Span(
            start=item.full_span.start,
            end=item.locator_span.end + found.end() if found else item.locator_span.end,
        )
        source = replace(item.source, date=date, span=span)
        stated = replace(item.stated, date=date, span=span)
        rebuilt.append(replace(item, source=source, stated=stated))
    return tuple(rebuilt)


def reread_post_citation(
    text: str,
    citations: Sequence[CitationRecord],
) -> tuple[CitationRecord, ...]:
    """Apply the court and date readers after co-location has defined bounds."""
    return reread_dates(text, reread_courts(text, citations))


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
