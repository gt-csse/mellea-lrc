"""Retrieve linked docket evidence for unique and bounded ambiguous lookups."""

from __future__ import annotations

from contextlib import ExitStack
from typing import Protocol

from mellea_lrc.model.citations import FullReporterCitation
from mellea_lrc.model.citations.reporter_lookup import (
    REPORTER_LOOKUP_CANDIDATE_LIMIT,
    ReporterExactCandidateDocket,
    ReporterExactDocket,
    ReporterExactLookupOutcome,
)
from mellea_lrc.model.document import Document
from mellea_lrc.providers.courtlistener import CourtListenerClient, CourtListenerDocket
from mellea_lrc.validation.reporter_exact.fields import candidate_court_id

STAGE = "12.2_reporter_root_lookup_docket_retrieval"
LOOKUP_STAGE = "12.1_reporter_root_lookup_cluster_retrieval"
UNIQUE_REVIEW_STAGE = "13.1_reporter_root_lookup_unique_rule_judgment"
AMBIGUOUS_REVIEW_STAGE = "13.2_reporter_root_lookup_ambiguous_rule_judgment"


class ReporterDocketClient(Protocol):
    def get_docket(self, docket_id: str) -> CourtListenerDocket | None: ...


def reporter_root_lookup_docket_retrieval(
    document: Document,
    *,
    client: ReporterDocketClient | None = None,
) -> Document:
    """Save linked court evidence before either rule judgment starts."""
    if STAGE in document.stage_runs:
        raise ValueError(f"Stage already completed: {STAGE}")
    if LOOKUP_STAGE not in document.stage_runs:
        raise ValueError("Retrieve reporter clusters before linked docket evidence")
    docket_cache: dict[str, CourtListenerDocket | None] = {}
    with ExitStack() as stack:
        service = client
        for root in tuple(item for item in document.roots if isinstance(item, FullReporterCitation)):
            if root.next_stage != STAGE:
                continue
            lookup = root.reporter_exact_lookup
            if (
                lookup is None
                or lookup.response is None
                or lookup.outcome
                not in {
                    ReporterExactLookupOutcome.UNIQUE,
                    ReporterExactLookupOutcome.AMBIGUOUS,
                }
            ):
                raise ValueError("Linked docket retrieval requires saved candidates")
            recorded = root.record(STAGE)
            if lookup.outcome is ReporterExactLookupOutcome.UNIQUE:
                candidate = lookup.response.clusters[0]
                if recorded.court and candidate.docket_id and candidate_court_id(candidate) is None:
                    docket_id = candidate.docket_id
                    if docket_id not in docket_cache:
                        if service is None:
                            service = stack.enter_context(CourtListenerClient())
                        docket_cache[docket_id] = service.get_docket(docket_id)
                    recorded = recorded.with_reporter_exact_docket(
                        ReporterExactDocket(
                            node_id=recorded.nodes[-1].id,
                            docket_id=docket_id,
                            response=docket_cache[docket_id],
                        )
                    )
                document = document.replace_citation(recorded.with_route(UNIQUE_REVIEW_STAGE))
                continue
            if len(lookup.response.clusters) < REPORTER_LOOKUP_CANDIDATE_LIMIT:
                for index, candidate in enumerate(lookup.response.clusters):
                    if not recorded.court or not candidate.docket_id or candidate_court_id(candidate):
                        continue
                    docket_id = candidate.docket_id
                    if docket_id not in docket_cache:
                        if service is None:
                            service = stack.enter_context(CourtListenerClient())
                        docket_cache[docket_id] = service.get_docket(docket_id)
                    recorded = recorded.with_reporter_exact_candidate_docket(
                        ReporterExactCandidateDocket(
                            node_id=recorded.nodes[-1].id,
                            candidate_index=index,
                            docket_id=docket_id,
                            response=docket_cache[docket_id],
                        )
                    )
            document = document.replace_citation(recorded.with_route(AMBIGUOUS_REVIEW_STAGE))
    return document.complete(STAGE)
