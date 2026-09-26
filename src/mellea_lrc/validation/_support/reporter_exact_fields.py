"""Rule comparisons shared by unique and ambiguous reporter exact lookups."""

from __future__ import annotations

from datetime import date

from mellea_lrc.courtlistener import CourtListenerCluster, CourtListenerDocket
from mellea_lrc.model.citations import FullReporterCitation
from mellea_lrc.model.citations.fields.court import Court, court_id_if_unique
from mellea_lrc.model.citations.fields.reporter import normalize_reporter_locator
from mellea_lrc.model.citations.judgments import MatchResult
from mellea_lrc.model.citations.reporter_lookup import ReporterExactLookupQuery
from mellea_lrc.validation._support.party_names import compare_case_names


def locator_present(cluster: CourtListenerCluster, query: ReporterExactLookupQuery) -> bool | None:
    """Check listed cluster citations when present; unreadable lists stay unknown."""
    parsed_any = False
    for citation in cluster.citations:
        try:
            listed = normalize_reporter_locator(f"{citation.volume} {citation.reporter} {citation.page}")
        except ValueError:
            continue
        parsed_any = True
        if (listed.volume, listed.edition, listed.page) == (query.volume, query.edition, query.page):
            return True
    return False if parsed_any else None


def case_name_result(citation: FullReporterCitation, candidate: CourtListenerCluster) -> MatchResult:
    reading = citation.case_name[-1]
    if not reading.normalizable or not candidate.case_name_full:
        return MatchResult.UNDETERMINED
    return (
        MatchResult.MATCH
        if compare_case_names(reading.get_normalized(), candidate.case_name_full).qualifies
        else MatchResult.MISMATCH
    )


def candidate_court_id(
    candidate: CourtListenerCluster, docket: CourtListenerDocket | None = None
) -> str | None:
    """Use the linked docket when the citation-lookup cluster omits its court."""
    ids: set[str] = set()
    values = (candidate.court_id, candidate.court)
    if docket is not None:
        values += (docket.court_id, docket.court)
    for value in values:
        if not value:
            continue
        if "/courts/" in value:
            value = value.rstrip("/").rsplit("/", maxsplit=1)[-1]
        try:
            ids.add(Court.from_id(value).id)
        except ValueError:
            if mapped := court_id_if_unique(value):
                ids.add(mapped)
    return next(iter(ids)) if len(ids) == 1 else None


def court_result(
    citation: FullReporterCitation, candidate: CourtListenerCluster, docket: CourtListenerDocket | None
) -> MatchResult:
    reading = citation.court[-1]
    court_id = candidate_court_id(candidate, docket)
    if not reading.normalizable or court_id is None:
        return MatchResult.UNDETERMINED
    return MatchResult.MATCH if reading.get_normalized().id == court_id else MatchResult.MISMATCH


def date_result(citation: FullReporterCitation, candidate: CourtListenerCluster) -> MatchResult:
    reading = citation.date[-1]
    if not reading.normalizable:
        return MatchResult.UNDETERMINED
    source = reading.get_normalized()
    written = candidate.date_filed or ""
    try:
        filed = date.fromisoformat(written[:10])
    except ValueError:
        if len(written) == 4 and written.isdecimal() and source.month is None:
            return MatchResult.MATCH if source.year == int(written) else MatchResult.MISMATCH
        return MatchResult.UNDETERMINED
    if source.month is None:
        agrees = source.year == filed.year
    elif source.day is None:
        agrees = (source.year, source.month) == (filed.year, filed.month)
    else:
        agrees = (source.year, source.month, source.day) == (filed.year, filed.month, filed.day)
    return MatchResult.MATCH if agrees else MatchResult.MISMATCH
