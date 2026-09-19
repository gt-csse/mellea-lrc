"""Final identity decision for evaluated exact-locator candidates."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mellea_lrc.validation.types import (
    CandidateSelectionNode,
    LocatorCandidateAssessmentOutcome,
    LocatorCitationSummaryNode,
    LocatorIdentityResolutionNode,
    LocatorIdentityResolutionOutcome,
    ValidationNodeStatus,
)

if TYPE_CHECKING:
    from mellea_lrc.validation.types import CitationValidation


def run_locator_identity_resolution(
    validation: CitationValidation,
    *,
    summary: LocatorCitationSummaryNode,
) -> LocatorIdentityResolutionNode:
    """Resolve an exact locator when exactly one reviewed candidate matches.

    ``partial_match`` is intentionally insufficient: it is useful review
    evidence, but does not establish a root. This keeps a summary's strongest
    candidate score from silently becoming an identity decision.
    """
    matches = tuple(
        candidate
        for candidate in summary.candidates
        if candidate.outcome is LocatorCandidateAssessmentOutcome.MATCH
    )
    matching_candidate_indices = tuple(candidate.candidate_index for candidate in matches)
    if len(matches) == 1:
        candidate = matches[0]
        return LocatorIdentityResolutionNode(
            node_id=f"{validation.citation_id}:locator_identity_resolution",
            status=ValidationNodeStatus.SUCCEEDED,
            outcome=LocatorIdentityResolutionOutcome.RESOLVED,
            selected_candidate_index=candidate.candidate_index,
            selected_assessment_node_id=candidate.assessment_node_id,
            matching_candidate_indices=matching_candidate_indices,
            depends_on=(summary.node_id,),
            status_message="Locator identity resolution completed.",
            outcome_message=(
                f"Candidate {candidate.candidate_index} is the only confirmed exact-locator match."
            ),
        )
    if not matches:
        message = "No reviewed exact-locator candidate was confirmed as a match."
    else:
        candidate_numbers = ", ".join(str(index) for index in matching_candidate_indices)
        message = f"Candidates {candidate_numbers} are all confirmed matches; identity remains ambiguous."
    return LocatorIdentityResolutionNode(
        node_id=f"{validation.citation_id}:locator_identity_resolution",
        status=ValidationNodeStatus.SUCCEEDED,
        outcome=LocatorIdentityResolutionOutcome.UNRESOLVED,
        selected_candidate_index=None,
        selected_assessment_node_id=None,
        matching_candidate_indices=matching_candidate_indices,
        depends_on=(summary.node_id,),
        status_message="Locator identity resolution completed.",
        outcome_message=message,
    )


def run_deferred_locator_identity_resolution(
    validation: CitationValidation,
    *,
    selection: CandidateSelectionNode,
) -> LocatorIdentityResolutionNode:
    """Record that an over-limit candidate set was intentionally not decided."""
    return LocatorIdentityResolutionNode(
        node_id=f"{validation.citation_id}:locator_identity_resolution",
        status=ValidationNodeStatus.SUCCEEDED,
        outcome=LocatorIdentityResolutionOutcome.DEFERRED,
        selected_candidate_index=None,
        selected_assessment_node_id=None,
        matching_candidate_indices=(),
        depends_on=(selection.node_id,),
        status_message="Locator identity resolution deferred.",
        outcome_message=(
            f"{selection.total_candidate_count} exact-locator candidates exceed the review limit of "
            f"{selection.selection_limit}."
        ),
    )
