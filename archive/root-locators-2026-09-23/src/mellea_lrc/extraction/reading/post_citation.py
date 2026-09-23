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

from mellea_lrc.extraction.reading.courts import (
    court_from_reporter,
    court_from_reporter_and_level,
    resolve_court,
)
from mellea_lrc.extraction.reading.dockets import CourtCandidate, court_for_docket
from mellea_lrc.extraction.reading.pin_cites import relax
from mellea_lrc.model.citations import CitationDate, CitationField, DocketCitation, FullCaseCitation
from mellea_lrc.model.operations import update_field, update_fields
from mellea_lrc.model.record import Node, Reads
from mellea_lrc.model.spans import Span

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mellea_lrc.model.record import CitationRecord

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


_CITATION_PARENTHETICAL = re.compile(
    r"[^\S\r\n]*[(\[][^()\[\]\r\n]*(?:\b(?:1[789]|20)\d{2}\b|\bU\.S\.)[^()\[\]\r\n]*[)\]]",
    re.IGNORECASE,
)


def docket_parenthetical_context(
    text: str, item: CitationRecord, citations: Sequence[CitationRecord]
) -> bool:
    """Whether the docket's group has a citation-style parenthetical.

    A date establishes that a following parenthetical belongs to a citation even
    when it writes no resolvable court. ``U.S.`` does the same for a Supreme
    Court docket, whose court is intentionally left unset until validation.
    Judge initials and filing-caption parentheticals do neither.
    """
    members = [
        other
        for other in citations
        if not other.withdrawn
        and (other is item or (item.colocation_id and other.colocation_id == item.colocation_id))
    ]
    last = max(members or [item], key=lambda other: other.locator_span.end)
    stop = _boundary(last, citations, len(text))
    return _CITATION_PARENTHETICAL.match(text, last.locator_span.end, stop) is not None


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
        if isinstance(item.fields, DocketCitation):
            candidate = docket_court(text, item, citations)
            if candidate is None:
                rebuilt.append(item)
                continue
            revised = replace(item)
            update_fields(
                revised,
                Node(
                    node_id=f"court_read:{item.citation_id}",
                    reads=Reads.DOCUMENT,
                    stage="court_read",
                    made_by=__name__,
                    outcome="found",
                    message=candidate.text,
                ),
                {
                    CitationField.COURT: candidate.court_id,
                    CitationField.COURT_NAME: candidate.court_name,
                    CitationField.COURT_TEXT: candidate.text,
                },
                reason="Read the court after the docket locator.",
            )
            rebuilt.append(revised)
            continue
        if not isinstance(item.fields, FullCaseCitation):
            rebuilt.append(item)
            continue
        found = _post_match(text, item, citations)
        court_text = (found.group("court") or "").strip() if found else ""
        inferred = _from_reporter(item.fields)
        court = resolve_court(court_text) or court_from_reporter_and_level(court_text, inferred) or inferred
        revised = replace(item)
        update_field(
            revised,
            Node(
                node_id=f"court_read:{item.citation_id}",
                reads=Reads.DOCUMENT,
                stage="court_read",
                made_by=__name__,
                outcome="found" if court is not None else "not_found",
                message=court_text or None,
            ),
            CitationField.COURT,
            court,
            reason="Read or infer the court after the reporter locator.",
        )
        rebuilt.append(revised)
    return tuple(rebuilt)


def reread_dates(
    text: str,
    citations: Sequence[CitationRecord],
) -> tuple[CitationRecord, ...]:
    """Resolve dates after locator spans, bounded by co-location groups."""
    rebuilt: list[CitationRecord] = []
    for item in citations:
        if item.withdrawn or not isinstance(item.fields, (FullCaseCitation, DocketCitation)):
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
        revised = replace(item)
        update_fields(
            revised,
            Node(
                node_id=f"date_read:{item.citation_id}",
                reads=Reads.DOCUMENT,
                stage="date_read",
                made_by=__name__,
                outcome="found" if date is not None else "not_found",
                message=str(date) if date is not None else None,
            ),
            {CitationField.DATE: date, CitationField.SPAN: span},
            reason="Read the citation date and complete its full span.",
        )
        rebuilt.append(revised)
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
