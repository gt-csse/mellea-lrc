"""Retrieve a found citation's court through its CourtListener docket."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mellea_lrc.courtlistener import CourtListenerError
from mellea_lrc.validation.types import (
    DocketCourtRetrievalNode,
    DocketCourtRetrievalOutcome,
    ExactLocatorLookupNode,
    ValidationNodeStatus,
)

if TYPE_CHECKING:
    from mellea_lrc.courtlistener.protocols import CourtListenerServiceClient
    from mellea_lrc.validation.types import CitationValidation


def run_docket_court_retrieval(
    validation: CitationValidation,
    *,
    lookup: ExactLocatorLookupNode,
    client: CourtListenerServiceClient,
) -> DocketCourtRetrievalNode:
    """Retrieve the court ID for the exact lookup's linked docket."""
    docket_id = lookup.record.docket_id if lookup.record is not None else None
    if not docket_id:
        return _node(
            validation, lookup, ValidationNodeStatus.SKIPPED, DocketCourtRetrievalOutcome.UNAVAILABLE
        )
    try:
        docket = client.get_docket(docket_id)
    except CourtListenerError as exc:
        return _node(
            validation,
            lookup,
            ValidationNodeStatus.FAILED,
            DocketCourtRetrievalOutcome.FAILED,
            error=exc.message,
        )
    if not docket.court_id:
        return _node(
            validation, lookup, ValidationNodeStatus.SUCCEEDED, DocketCourtRetrievalOutcome.UNAVAILABLE
        )
    return _node(
        validation,
        lookup,
        ValidationNodeStatus.SUCCEEDED,
        DocketCourtRetrievalOutcome.FOUND,
        court_id=docket.court_id,
    )


def _node(
    validation: CitationValidation,
    lookup: ExactLocatorLookupNode,
    status: ValidationNodeStatus,
    outcome: DocketCourtRetrievalOutcome,
    *,
    court_id: str | None = None,
    error: str | None = None,
) -> DocketCourtRetrievalNode:
    return DocketCourtRetrievalNode(
        node_id=f"{validation.citation_id}:docket_court_retrieval",
        status=status,
        outcome=outcome,
        docket_id=lookup.record.docket_id if lookup.record is not None else None,
        court_id=court_id,
        depends_on=(lookup.node_id,),
        error=error,
    )
