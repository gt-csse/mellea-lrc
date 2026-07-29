"""Retrieve a found citation's court through its CourtListener docket."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mellea_lrc.courtlistener import CourtListenerError
from mellea_lrc.validation.types import (
    CandidateEvaluationNode,
    DocketCourtRetrievalNode,
    DocketCourtRetrievalOutcome,
    ValidationNodeStatus,
)

if TYPE_CHECKING:
    from mellea_lrc.courtlistener.protocols import CourtListenerServiceClient
    from mellea_lrc.validation.types import CitationValidation


def run_docket_court_retrieval(
    validation: CitationValidation,
    *,
    candidate: CandidateEvaluationNode,
    client: CourtListenerServiceClient,
) -> DocketCourtRetrievalNode:
    """Retrieve the court ID for the exact lookup's linked docket."""
    docket_id = candidate.docket_id
    if not docket_id:
        return _node(
            validation,
            candidate,
            ValidationNodeStatus.SKIPPED,
            DocketCourtRetrievalOutcome.UNAVAILABLE,
            status_message="Skipped docket court retrieval because the citation record has no docket identifier.",
            outcome_message="Court retrieval is unavailable because the citation record has no docket identifier.",
        )
    try:
        docket = client.get_docket(docket_id)
    except CourtListenerError as exc:
        return _node(
            validation,
            candidate,
            ValidationNodeStatus.FAILED,
            DocketCourtRetrievalOutcome.FAILED,
            status_message="Docket court retrieval failed while calling CourtListener.",
            outcome_message="Court retrieval from the docket could not be completed.",
            error=exc.message,
        )
    if not docket.court_id:
        return _node(
            validation,
            candidate,
            ValidationNodeStatus.SUCCEEDED,
            DocketCourtRetrievalOutcome.UNAVAILABLE,
            status_message="Docket court retrieval completed.",
            outcome_message="The retrieved docket does not provide a court identifier.",
        )
    return _node(
        validation,
        candidate,
        ValidationNodeStatus.SUCCEEDED,
        DocketCourtRetrievalOutcome.FOUND,
        court_id=docket.court_id,
        status_message="Docket court retrieval completed.",
        outcome_message="Retrieved a court identifier from the citation docket.",
    )


def _node(
    validation: CitationValidation,
    candidate: CandidateEvaluationNode,
    status: ValidationNodeStatus,
    outcome: DocketCourtRetrievalOutcome,
    *,
    court_id: str | None = None,
    status_message: str | None = None,
    outcome_message: str | None = None,
    error: str | None = None,
) -> DocketCourtRetrievalNode:
    return DocketCourtRetrievalNode(
        node_id=f"{validation.citation_id}:docket_court_retrieval",
        status=status,
        outcome=outcome,
        docket_id=candidate.docket_id,
        court_id=court_id,
        depends_on=(candidate.node_id,),
        status_message=status_message,
        outcome_message=outcome_message,
        error=error,
    )
