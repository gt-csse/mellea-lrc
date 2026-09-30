"""Retrieve linked court evidence for bounded ambiguous reporter lookups."""

from __future__ import annotations

from contextlib import ExitStack
from typing import Protocol

from mellea_lrc.courtlistener import CourtListenerClient, CourtListenerDocket
from mellea_lrc.model.citations import FullReporterCitation
from mellea_lrc.model.citations.reporter_lookup import (
    ReporterExactCandidateDocket,
    ReporterExactLookupOutcome,
)
from mellea_lrc.model.document import Document
from mellea_lrc.validation.reporter_exact.fields import candidate_court_id

STAGE = "13.1_reporter_root_lookup_ambiguous_dockets"
LOOKUP_STAGE = "12.1_reporter_root_lookup"
REVIEW_STAGE = "13.2_reporter_root_lookup_ambiguous_review"
CANDIDATE_LIMIT = 20


class ReporterDocketClient(Protocol):
    def get_docket(self, docket_id: str) -> CourtListenerDocket | None: ...


def reporter_root_lookup_ambiguous_dockets(
    document: Document,
    *,
    client: ReporterDocketClient | None = None,
) -> Document:
    """Save every needed linked docket before ambiguous rule review starts."""
    if STAGE in document.stage_runs:
        raise ValueError(f"Stage already completed: {STAGE}")
    if LOOKUP_STAGE not in document.stage_runs:
        raise ValueError("Retrieve reporter lookups before ambiguous docket evidence")
    docket_cache: dict[str, CourtListenerDocket | None] = {}
    with ExitStack() as stack:
        service = client
        for root in tuple(item for item in document.roots if isinstance(item, FullReporterCitation)):
            if root.next_stage != STAGE:
                continue
            lookup = root.reporter_exact_lookup
            if (
                lookup is None
                or lookup.outcome is not ReporterExactLookupOutcome.AMBIGUOUS
                or lookup.response is None
            ):
                raise ValueError("Ambiguous docket retrieval requires saved candidates")
            recorded = root.record(STAGE)
            if len(lookup.response.clusters) < CANDIDATE_LIMIT:
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
            document = document.replace_citation(recorded.with_route(REVIEW_STAGE))
    return document.complete(STAGE)
