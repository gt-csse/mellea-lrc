"""CourtListener opinion search after case-name query preparation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mellea_lrc.courtlistener import CourtListenerError
from mellea_lrc.validation.types import (
    MelleaCaseNameQueryPreparationNode,
    MelleaCaseNameQueryPreparationOutcome,
    OpinionSearchNode,
    OpinionSearchOutcome,
    ValidationNodeStatus,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from mellea_lrc.courtlistener.protocols import CourtListenerServiceClient
    from mellea_lrc.validation.types import CitationValidation


def run_opinion_search(
    validation: CitationValidation,
    *,
    preparation: MelleaCaseNameQueryPreparationNode,
    client: CourtListenerServiceClient,
) -> OpinionSearchNode:
    """Search only CourtListener's opinion corpus using one prepared query."""
    if preparation.outcome is not MelleaCaseNameQueryPreparationOutcome.PREPARED or preparation.query is None:
        return _node(
            validation,
            preparation,
            ValidationNodeStatus.SKIPPED,
            OpinionSearchOutcome.UNAVAILABLE,
            query=preparation.query,
        )
    try:
        result = client.search(preparation.query, "o")
    except CourtListenerError as exc:
        return _node(
            validation,
            preparation,
            ValidationNodeStatus.FAILED,
            OpinionSearchOutcome.FAILED,
            query=preparation.query,
            error=exc.message,
        )
    return _node(
        validation,
        preparation,
        ValidationNodeStatus.SUCCEEDED,
        OpinionSearchOutcome.SEARCHED,
        query=result.query,
        result_count=result.count,
        results=result.results,
        next_cursor=result.next_cursor,
    )


def _node(
    validation: CitationValidation,
    preparation: MelleaCaseNameQueryPreparationNode,
    status: ValidationNodeStatus,
    outcome: OpinionSearchOutcome,
    *,
    query: str | None,
    result_count: int | None = None,
    results: tuple[Mapping[str, object], ...] = (),
    next_cursor: str | None = None,
    error: str | None = None,
) -> OpinionSearchNode:
    """Build one terminal opinion-search node."""
    return OpinionSearchNode(
        node_id=f"{validation.citation_id}:opinion_search",
        status=status,
        outcome=outcome,
        query=query,
        result_count=result_count,
        results=results,
        next_cursor=next_cursor,
        depends_on=(preparation.node_id,),
        error=error,
    )
