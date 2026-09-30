"""Retrieve and save exact reporter-root candidates without judging them."""

from __future__ import annotations

from contextlib import ExitStack
from typing import Protocol

from mellea_lrc.courtlistener import (
    CourtListenerCitationLookup,
    CourtListenerClient,
    CourtListenerError,
)
from mellea_lrc.model.citations import FullReporterCitation
from mellea_lrc.model.citations.reporter_lookup import (
    ReporterExactLookup,
    ReporterExactLookupOutcome,
    ReporterExactLookupQuery,
)
from mellea_lrc.model.document import Document

STAGE = "12.1_reporter_root_lookup_cluster_retrieval"
DOCKET_RETRIEVAL_STAGE = "12.2_reporter_root_lookup_docket_retrieval"


class ReporterLookupClient(Protocol):
    """The exact reporter-citation lookup."""

    def lookup_citation(self, volume: str, reporter: str, page: str) -> CourtListenerCitationLookup: ...


def reporter_root_lookup_cluster_retrieval(
    document: Document,
    *,
    client: ReporterLookupClient | None = None,
) -> Document:
    """Save exact lookup responses for later linked-docket retrieval.

    Field comparison belongs to the next stage. Provider failures abort
    instead of masquerading as a lookup miss.
    """
    if STAGE in document.stage_runs:
        raise ValueError(f"Stage already completed: {STAGE}")
    if "10_roots" not in document.stage_runs:
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
                if outcome in {ReporterExactLookupOutcome.UNIQUE, ReporterExactLookupOutcome.AMBIGUOUS}:
                    recorded = recorded.with_route(DOCKET_RETRIEVAL_STAGE)
                else:
                    recorded = recorded.with_route("reporter_root_search")
            document = document.replace_citation(recorded)
    return document.complete(STAGE)
