"""Retrieve and save exact reporter-root candidates without judging them."""

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
from mellea_lrc.model.citations.reporter_lookup import (
    ReporterExactDocket,
    ReporterExactLookup,
    ReporterExactLookupOutcome,
    ReporterExactLookupQuery,
)
from mellea_lrc.model.document import Document
from mellea_lrc.validation.reporter_exact.fields import candidate_court_id

STAGE = "12.1_reporter_root_lookup"
UNIQUE_REVIEW_STAGE = "12.2_reporter_root_lookup_review"
AMBIGUOUS_DOCKETS_STAGE = "13.1_reporter_root_lookup_ambiguous_dockets"


class ReporterLookupClient(Protocol):
    """The exact citation lookup and its linked docket court retrieval."""

    def lookup_citation(self, volume: str, reporter: str, page: str) -> CourtListenerCitationLookup: ...

    def get_docket(self, docket_id: str) -> CourtListenerDocket | None: ...


def reporter_root_lookup(
    document: Document,
    *,
    client: ReporterLookupClient | None = None,
) -> Document:
    """Save exact lookup responses and unique-candidate linked court evidence.

    Field comparison belongs to the next stage. Provider failures abort
    instead of masquerading as a lookup miss.
    """
    if STAGE in document.stage_runs:
        raise ValueError(f"Stage already completed: {STAGE}")
    if "10_roots" not in document.stage_runs:
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
                recorded = recorded.with_route("reporter_root_search")
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
                    recorded = recorded.with_route(UNIQUE_REVIEW_STAGE)
                elif outcome is ReporterExactLookupOutcome.AMBIGUOUS:
                    recorded = recorded.with_route(AMBIGUOUS_DOCKETS_STAGE)
                else:
                    recorded = recorded.with_route("reporter_root_search")
            document = document.replace_citation(recorded)
    return document.complete(STAGE)
