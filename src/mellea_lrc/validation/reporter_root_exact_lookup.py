"""Rule-only exact retrieval and field judgments for reporter roots."""

from __future__ import annotations

from contextlib import ExitStack
from datetime import date
from typing import Protocol

from mellea_lrc.courtlistener import (
    CourtListenerCitationLookup,
    CourtListenerClient,
    CourtListenerCluster,
    CourtListenerError,
)
from mellea_lrc.model.citations import FullReporterCitation
from mellea_lrc.model.citations.fields.court import Court, court_id_if_unique
from mellea_lrc.model.citations.fields.reporter import normalize_reporter_locator
from mellea_lrc.model.citations.judgments import IdentityNextStep, IdentityVerdict, MatchResult
from mellea_lrc.model.citations.reporter_lookup import (
    ReporterExactLookup,
    ReporterExactLookupOutcome,
    ReporterExactLookupQuery,
)
from mellea_lrc.model.document import Document
from mellea_lrc.validation._support.party_names import compare_case_names

STAGE = "reporter_root_exact_lookup"


class ReporterLookupClient(Protocol):
    """The one CourtListener operation this stage needs."""

    def lookup_citation(self, volume: str, reporter: str, page: str) -> CourtListenerCitationLookup: ...


def _locator_present(cluster: CourtListenerCluster, query: ReporterExactLookupQuery) -> bool | None:
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


def _case_name_result(citation: FullReporterCitation, candidate: CourtListenerCluster) -> MatchResult:
    reading = citation.case_name[-1]
    if not reading.normalizable or not candidate.case_name_full:
        return MatchResult.UNDETERMINED
    return (
        MatchResult.MATCH
        if compare_case_names(reading.get_normalized(), candidate.case_name_full).qualifies
        else MatchResult.MISMATCH
    )


def _candidate_court_id(candidate: CourtListenerCluster) -> str | None:
    """Use a recognized provider court ID or label; conflicting values stay unknown."""
    ids: set[str] = set()
    for value in (candidate.court_id, candidate.court):
        if not value:
            continue
        try:
            ids.add(Court.from_id(value).id)
        except ValueError:
            if mapped := court_id_if_unique(value):
                ids.add(mapped)
    return next(iter(ids)) if len(ids) == 1 else None


def _court_result(citation: FullReporterCitation, candidate: CourtListenerCluster) -> MatchResult:
    reading = citation.court[-1]
    court_id = _candidate_court_id(candidate)
    if not reading.normalizable or court_id is None:
        return MatchResult.UNDETERMINED
    return MatchResult.MATCH if reading.get_normalized().id == court_id else MatchResult.MISMATCH


def _date_result(citation: FullReporterCitation, candidate: CourtListenerCluster) -> MatchResult:
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


def _judge_unique(citation: FullReporterCitation, query: ReporterExactLookupQuery) -> FullReporterCitation:
    """Compare one cluster, retaining independent field results and one route."""
    lookup = citation.reporter_exact_lookup
    if lookup is None or lookup.response is None or len(lookup.response.clusters) != 1:
        raise ValueError("Unique reporter judgment requires one saved candidate")
    candidate = lookup.response.clusters[0]
    field_results: list[MatchResult] = []
    if citation.case_name:
        result = _case_name_result(citation, candidate)
        citation = citation.with_case_name_judgment(len(citation.case_name) - 1, 0, result)
        field_results.append(result)
    if citation.court:
        court_reading = citation.court[-1]
        # The exact-lookup cluster commonly omits court metadata. A court
        # inferred solely from the reporter adds no independent written claim.
        if _candidate_court_id(candidate) is not None or court_reading.span is not None:
            result = _court_result(citation, candidate)
            citation = citation.with_court_judgment(len(citation.court) - 1, 0, result)
            field_results.append(result)
    # If either side has no date, there is no date opinion and no identity penalty.
    if citation.date and candidate.date_filed:
        result = _date_result(citation, candidate)
        citation = citation.with_date_judgment(len(citation.date) - 1, 0, result)
        field_results.append(result)
    if (
        citation.case_name_judgments
        and all(result is MatchResult.MATCH for result in field_results)
        and _locator_present(candidate, query) is not False
    ):
        return citation.with_identity_judgment(IdentityVerdict.CORRECT_IDENTITY)
    return citation.with_identity_judgment(IdentityVerdict.DEFERRED, IdentityNextStep.REVIEW)


def reporter_root_exact_lookup(
    document: Document,
    *,
    client: ReporterLookupClient | None = None,
) -> Document:
    """Look up each reporter root once and decide unique rule-checkable identities.

    A unique candidate with any disagreement goes to one later review; multiple
    candidates go to ambiguity review, and an absent candidate goes to search.
    Provider failures abort instead of masquerading as a lookup miss.
    """
    if STAGE in document.stage_runs:
        raise ValueError(f"Stage already completed: {STAGE}")
    if "roots" not in document.stage_runs:
        raise ValueError("Form roots before exact reporter lookup")
    roots = tuple(root for root in document.roots if isinstance(root, FullReporterCitation))
    with ExitStack() as stack:
        service = client
        for citation in roots:
            recorded = citation.record(STAGE)
            reading = citation.locator[-1]
            if not reading.normalizable:
                result = ReporterExactLookup(
                    node_id=recorded.nodes[-1].id,
                    outcome=ReporterExactLookupOutcome.UNNORMALIZABLE,
                )
                recorded = recorded.with_reporter_exact_lookup(result)
                recorded = recorded.with_identity_judgment(IdentityVerdict.DEFERRED, IdentityNextStep.SEARCH)
            else:
                locator = reading.get_normalized()
                query = ReporterExactLookupQuery(
                    volume=locator.volume,
                    edition=locator.edition,
                    page=locator.page,
                )
                if service is None:
                    service = stack.enter_context(CourtListenerClient())
                response = service.lookup_citation(str(query.volume), query.edition, query.page)
                if response.status not in {200, 300, 400, 404}:
                    raise CourtListenerError(
                        f"CourtListener citation lookup returned item status {response.status}",
                        failure_type="upstream_item_error",
                        upstream_status_code=response.status,
                    )
                count = len(response.clusters)
                outcome = (
                    ReporterExactLookupOutcome.NOT_FOUND
                    if count == 0
                    else ReporterExactLookupOutcome.UNIQUE
                    if count == 1
                    else ReporterExactLookupOutcome.AMBIGUOUS
                )
                result = ReporterExactLookup(
                    node_id=recorded.nodes[-1].id,
                    outcome=outcome,
                    query=query,
                    response=response,
                )
                recorded = recorded.with_reporter_exact_lookup(result)
                if outcome is ReporterExactLookupOutcome.UNIQUE:
                    recorded = _judge_unique(recorded, query)
                elif outcome is ReporterExactLookupOutcome.AMBIGUOUS:
                    recorded = recorded.with_identity_judgment(
                        IdentityVerdict.DEFERRED, IdentityNextStep.AMBIGUITY
                    )
                else:
                    recorded = recorded.with_identity_judgment(
                        IdentityVerdict.DEFERRED, IdentityNextStep.SEARCH
                    )
            document = document.replace_citation(recorded)
    return document.complete(STAGE)
