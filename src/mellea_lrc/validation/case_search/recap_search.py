"""CourtListener RECAP search after case-name query preparation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mellea_lrc.courtlistener import CourtListenerError
from mellea_lrc.validation.types import (
    MelleaCaseNameQueryPreparationNode,
    MelleaCaseNameQueryPreparationOutcome,
    RecapSearchNode,
    RecapSearchOutcome,
    ValidationNodeStatus,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from mellea_lrc.courtlistener.protocols import CourtListenerServiceClient
    from mellea_lrc.validation.types import CitationValidation


def run_recap_search(
    validation: CitationValidation,
    *,
    preparation: MelleaCaseNameQueryPreparationNode,
    client: CourtListenerServiceClient,
) -> RecapSearchNode:
    """Search only CourtListener's RECAP corpus using one prepared query."""
    if preparation.outcome is not MelleaCaseNameQueryPreparationOutcome.PREPARED or preparation.query is None:
        return _node(
            validation,
            preparation,
            ValidationNodeStatus.SKIPPED,
            RecapSearchOutcome.UNAVAILABLE,
            query=preparation.query,
            status_message="Skipped RECAP search because no prepared RECAP query is available.",
            outcome_message="RECAP search is unavailable without a prepared query.",
        )
    try:
        result = client.search(preparation.query, "r")
    except CourtListenerError as exc:
        return _node(
            validation,
            preparation,
            ValidationNodeStatus.FAILED,
            RecapSearchOutcome.FAILED,
            query=preparation.query,
            status_message="RECAP search failed while calling CourtListener.",
            outcome_message="CourtListener RECAP search could not be completed.",
            error=exc.message,
        )
    return _node(
        validation,
        preparation,
        ValidationNodeStatus.SUCCEEDED,
        RecapSearchOutcome.SEARCHED,
        query=result.query,
        result_count=result.count,
        results=result.results,
        next_cursor=result.next_cursor,
        status_message="RECAP search completed.",
        outcome_message=f"CourtListener returned {result.count} RECAP search results.",
    )


def _node(
    validation: CitationValidation,
    preparation: MelleaCaseNameQueryPreparationNode,
    status: ValidationNodeStatus,
    outcome: RecapSearchOutcome,
    *,
    query: str | None,
    result_count: int | None = None,
    results: tuple[Mapping[str, object], ...] = (),
    next_cursor: str | None = None,
    status_message: str | None = None,
    outcome_message: str | None = None,
    error: str | None = None,
) -> RecapSearchNode:
    """Build one terminal RECAP-search node."""
    return RecapSearchNode(
        node_id=f"{validation.citation_id}:recap_search",
        status=status,
        outcome=outcome,
        query=query,
        result_count=result_count,
        results=results,
        next_cursor=next_cursor,
        depends_on=(preparation.node_id,),
        status_message=status_message,
        outcome_message=outcome_message,
        error=error,
    )
