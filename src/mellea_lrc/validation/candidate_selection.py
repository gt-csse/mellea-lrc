"""Bound candidate evaluation without truncating retrieved result sets."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mellea_lrc.validation.types import (
    CandidateSelectionNode,
    CandidateSelectionOutcome,
    ExactLocatorLookupNode,
    OpinionSearchNode,
    RecapSearchNode,
    ValidationNodeStatus,
)

if TYPE_CHECKING:
    from mellea_lrc.validation.types import CitationValidation

CANDIDATE_SELECTION_LIMIT = 3


def run_locator_candidate_selection(
    validation: CitationValidation,
    *,
    lookup: ExactLocatorLookupNode,
) -> CandidateSelectionNode:
    """Apply the common guard to an ambiguous exact-locator result set."""
    return _selection(
        node_id=f"{validation.citation_id}:locator_candidate_selection",
        retrieval_node_id=lookup.node_id,
        total_candidate_count=lookup.candidate_count,
    )


def run_opinion_search_candidate_selection(
    validation: CitationValidation,
    *,
    search: OpinionSearchNode,
) -> CandidateSelectionNode:
    """Apply the common guard to CourtListener opinion-search results."""
    return _selection(
        node_id=f"{validation.citation_id}:opinion_search_candidate_selection",
        retrieval_node_id=search.node_id,
        total_candidate_count=_required_result_count(search.result_count, search.node_id),
    )


def run_recap_search_candidate_selection(
    validation: CitationValidation,
    *,
    search: RecapSearchNode,
) -> CandidateSelectionNode:
    """Apply the common guard to CourtListener RECAP-search results."""
    return _selection(
        node_id=f"{validation.citation_id}:recap_search_candidate_selection",
        retrieval_node_id=search.node_id,
        total_candidate_count=_required_result_count(search.result_count, search.node_id),
    )


def _selection(
    *,
    node_id: str,
    retrieval_node_id: str,
    total_candidate_count: int,
) -> CandidateSelectionNode:
    """Create the shared selection decision for one retrieved result set."""
    selected_count = total_candidate_count if total_candidate_count <= CANDIDATE_SELECTION_LIMIT else 0
    outcome = (
        CandidateSelectionOutcome.ALL_SELECTED
        if selected_count == total_candidate_count
        else CandidateSelectionOutcome.DEFERRED_OVER_LIMIT
    )
    outcome_message = (
        f"All {total_candidate_count} returned candidates are within the current validation scope of "
        f"{CANDIDATE_SELECTION_LIMIT}."
        if outcome is CandidateSelectionOutcome.ALL_SELECTED
        else (
            f"Candidate validation is deferred because {total_candidate_count} returned candidates exceed "
            f"the current scope of {CANDIDATE_SELECTION_LIMIT}; further refinement is needed before "
            "selecting candidates."
        )
    )
    return CandidateSelectionNode(
        node_id=node_id,
        status=ValidationNodeStatus.SUCCEEDED,
        outcome=outcome,
        total_candidate_count=total_candidate_count,
        selected_candidate_count=selected_count,
        selection_limit=CANDIDATE_SELECTION_LIMIT,
        depends_on=(retrieval_node_id,),
        status_message="Candidate selection completed.",
        outcome_message=outcome_message,
    )


def _required_result_count(result_count: int | None, node_id: str) -> int:
    """Require a completed search count before making a selection decision."""
    if result_count is None:
        msg = f"Candidate selection requires a result count from {node_id!r}"
        raise ValueError(msg)
    return result_count
