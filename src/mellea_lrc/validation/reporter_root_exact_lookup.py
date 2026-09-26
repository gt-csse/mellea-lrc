"""Rule-only exact retrieval and field judgments for reporter roots."""

from __future__ import annotations

from contextlib import ExitStack
from typing import Protocol

from mellea_lrc.courtlistener import (
    CourtListenerCitationLookup,
    CourtListenerClient,
    CourtListenerDocket,
    CourtListenerError,
)
from mellea_lrc.model.citations import FullReporterCitation
from mellea_lrc.model.citations.judgments import IdentityNextStep, IdentityVerdict, MatchResult
from mellea_lrc.model.citations.reporter_lookup import (
    ReporterExactDocket,
    ReporterExactLookup,
    ReporterExactLookupOutcome,
    ReporterExactLookupQuery,
)
from mellea_lrc.model.document import Document
from mellea_lrc.validation._support.reporter_exact_fields import (
    candidate_court_id,
    case_name_result,
    court_result,
    date_result,
    locator_present,
)

STAGE = "reporter_root_exact_lookup"


class ReporterLookupClient(Protocol):
    """The exact citation lookup and its linked docket court retrieval."""

    def lookup_citation(self, volume: str, reporter: str, page: str) -> CourtListenerCitationLookup: ...

    def get_docket(self, docket_id: str) -> CourtListenerDocket | None: ...


def _judge_unique(citation: FullReporterCitation, query: ReporterExactLookupQuery) -> FullReporterCitation:
    """Compare one cluster, retaining independent field results and one route."""
    lookup = citation.reporter_exact_lookup
    if lookup is None or lookup.response is None or len(lookup.response.clusters) != 1:
        raise ValueError("Unique reporter judgment requires one saved candidate")
    candidate = lookup.response.clusters[0]
    docket = citation.reporter_exact_docket.response if citation.reporter_exact_docket is not None else None
    field_results: list[MatchResult] = []
    if citation.case_name:
        result = case_name_result(citation, candidate)
        citation = citation.with_case_name_judgment(len(citation.case_name) - 1, 0, result)
        field_results.append(result)
    if citation.court:
        result = court_result(citation, candidate, docket)
        citation = citation.with_court_judgment(len(citation.court) - 1, 0, result)
        field_results.append(result)
    # If either side has no date, there is no date opinion and no identity penalty.
    if citation.date and candidate.date_filed:
        result = date_result(citation, candidate)
        citation = citation.with_date_judgment(len(citation.date) - 1, 0, result)
        field_results.append(result)
    if (
        citation.case_name_judgments
        and all(result is MatchResult.MATCH for result in field_results)
        and locator_present(candidate, query) is not False
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
    docket_cache: dict[str, CourtListenerDocket | None] = {}
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
                    candidate = response.clusters[0]
                    if recorded.court and candidate.docket_id and candidate_court_id(candidate) is None:
                        docket_id = candidate.docket_id
                        if docket_id not in docket_cache:
                            docket_cache[docket_id] = service.get_docket(docket_id)
                        recorded = recorded.with_reporter_exact_docket(
                            ReporterExactDocket(
                                node_id=recorded.nodes[-1].id,
                                docket_id=docket_id,
                                response=docket_cache[docket_id],
                            )
                        )
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
