"""Rule-only exact retrieval for reporter roots.

This stage persists every CourtListener candidate and checks whether both
source parties occur in the returned full case name after Bluebook abbreviation
expansion. A candidate passing these preliminary checks is not yet an identity
judgment: court, date, and later ambiguity decisions remain separate work.
"""

from __future__ import annotations

from contextlib import ExitStack
from typing import Protocol

from mellea_lrc.courtlistener import (
    CourtListenerCitationLookup,
    CourtListenerClient,
    CourtListenerCluster,
    CourtListenerError,
)
from mellea_lrc.model.citations import FullReporterCitation
from mellea_lrc.model.citations.fields.reporter import normalize_reporter_locator
from mellea_lrc.model.citations.reporter_lookup import (
    ReporterExactCandidateCheck,
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


def _check_candidate(
    citation: FullReporterCitation,
    candidate: CourtListenerCluster,
    index: int,
    query: ReporterExactLookupQuery,
) -> ReporterExactCandidateCheck:
    name_source = candidate.case_name_full
    name_reading = citation.case_name[-1] if citation.case_name else None
    locator_present = _locator_present(candidate, query)
    if not name_source or name_reading is None or not name_reading.normalizable:
        return ReporterExactCandidateCheck(
            candidate_index=index,
            name_source=name_source,
            locator_present=locator_present,
        )
    match = compare_case_names(name_reading.get_normalized(), name_source)
    return ReporterExactCandidateCheck(
        candidate_index=index,
        name_source=name_source,
        plaintiff_present=match.plaintiff_present,
        defendant_present=match.defendant_present,
        subject_present=match.subject_present,
        name_rule_passed=match.qualifies,
        locator_present=locator_present,
        qualifies=match.qualifies and locator_present is not False,
    )


def reporter_root_exact_lookup(
    document: Document,
    *,
    client: ReporterLookupClient | None = None,
) -> Document:
    """Look up each reporter root once and persist deterministic candidate checks.

    Every candidate is retained in response order. A rule-positive candidate
    remains only a candidate for later identity validation; a rule-negative one
    is unresolved rather than declared to be a different case. Provider errors
    abort the stage, so a failed request cannot masquerade as a lookup miss.
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
                    candidate_checks=tuple(
                        _check_candidate(citation, candidate, index, query)
                        for index, candidate in enumerate(response.clusters)
                    ),
                )
            document = document.replace_citation(recorded.with_reporter_exact_lookup(result))
    return document.complete(STAGE)
