"""Terminal reporter-locator identity decisions from complete candidate evidence."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mellea_lrc.validation.types import (
    LocatorCandidateAssessmentOutcome,
    LocatorCitationSummaryNode,
    LocatorIdentityResolutionNode,
    LocatorIdentityResolutionOutcome,
    MelleaLocatorCandidateChoiceNode,
    MelleaLocatorCandidateChoiceOutcome,
    ValidationNodeStatus,
)

if TYPE_CHECKING:
    from mellea_lrc.validation.types import CitationValidation


def requires_mellea_locator_candidate_choice(summary: LocatorCitationSummaryNode) -> bool:
    """Return whether deterministic assessment evidence leaves identity undecided."""
    return len(_matching_candidates(summary)) != 1


def run_locator_identity_resolution(
    validation: CitationValidation,
    *,
    summary: LocatorCitationSummaryNode,
    choice: MelleaLocatorCandidateChoiceNode | None = None,
) -> LocatorIdentityResolutionNode:
    """Resolve a reviewed reporter locator from deterministic or model evidence.

    Exactly one confirmed candidate is admitted deterministically.  With zero
    or multiple confirmations, a grounded model choice is required: it reparses
    the stated fields and selects one of the complete candidate list or reports
    no match.  Candidate evidence stays in ``summary`` either way.
    """
    matches = _matching_candidates(summary)
    matching_candidate_indices = tuple(candidate.candidate_index for candidate in matches)
    if len(matches) == 1 and choice is None:
        candidate = matches[0]
        return _resolution(
            validation,
            outcome=LocatorIdentityResolutionOutcome.RESOLVED,
            selected_candidate_index=candidate.candidate_index,
            selected_assessment_node_id=candidate.assessment_node_id,
            matching_candidate_indices=matching_candidate_indices,
            selection_evidence_node_id=summary.node_id,
            depends_on=(summary.node_id,),
            status_message="Locator identity resolution completed.",
            outcome_message=(
                f"Candidate {candidate.candidate_index} is the only confirmed exact-locator match."
            ),
        )
    if choice is None:
        msg = "A model candidate choice is required when exact-locator evidence is not uniquely confirmed"
        raise ValueError(msg)
    if choice.outcome is MelleaLocatorCandidateChoiceOutcome.SELECTED:
        candidate = next(
            (item for item in summary.candidates if item.candidate_index == choice.selected_candidate_index),
            None,
        )
        if candidate is None:
            msg = "Model candidate choice selected a candidate absent from the locator summary"
            raise ValueError(msg)
        return _resolution(
            validation,
            outcome=LocatorIdentityResolutionOutcome.RESOLVED,
            selected_candidate_index=candidate.candidate_index,
            selected_assessment_node_id=candidate.assessment_node_id,
            matching_candidate_indices=matching_candidate_indices,
            selection_evidence_node_id=choice.node_id,
            depends_on=(summary.node_id, choice.node_id),
            status_message="Locator identity resolution completed.",
            outcome_message=(
                f"Grounded model choice selected candidate {candidate.candidate_index} after reparsing "
                "the local citation fields."
            ),
        )
    if choice.outcome is MelleaLocatorCandidateChoiceOutcome.NO_MATCH:
        return _resolution(
            validation,
            outcome=LocatorIdentityResolutionOutcome.NO_MATCH,
            matching_candidate_indices=matching_candidate_indices,
            selection_evidence_node_id=choice.node_id,
            depends_on=(summary.node_id, choice.node_id),
            status_message="Locator identity resolution completed.",
            outcome_message="No reviewed exact-locator candidate represents the reparsed citation fields.",
        )
    return _resolution(
        validation,
        outcome=LocatorIdentityResolutionOutcome.DEFERRED_TO_FUTURE_IMPLEMENTATION,
        matching_candidate_indices=matching_candidate_indices,
        selection_evidence_node_id=choice.node_id,
        depends_on=(summary.node_id, choice.node_id),
        status_message="Locator identity resolution deferred to future implementation.",
        outcome_message="Model candidate choice failed; no identity decision was admitted.",
    )


def run_search_deferred_locator_identity_resolution(
    validation: CitationValidation,
    *,
    depends_on: tuple[str, ...] = (),
    reason: str,
) -> LocatorIdentityResolutionNode:
    """Hand a reporter locator with no exact match to the search stage."""
    return _resolution(
        validation,
        outcome=LocatorIdentityResolutionOutcome.DEFERRED_TO_SEARCH,
        matching_candidate_indices=(),
        selection_evidence_node_id=None,
        depends_on=depends_on,
        status_message="Locator identity resolution deferred to search.",
        outcome_message=reason,
    )


def run_future_implementation_deferred_locator_identity_resolution(
    validation: CitationValidation,
    *,
    depends_on: tuple[str, ...] = (),
    reason: str,
) -> LocatorIdentityResolutionNode:
    """Record a bounded checkpoint result with no admitted next route yet."""
    return _resolution(
        validation,
        outcome=LocatorIdentityResolutionOutcome.DEFERRED_TO_FUTURE_IMPLEMENTATION,
        matching_candidate_indices=(),
        selection_evidence_node_id=None,
        depends_on=depends_on,
        status_message="Locator identity resolution deferred to future implementation.",
        outcome_message=reason,
    )


def _matching_candidates(summary: LocatorCitationSummaryNode):
    return tuple(
        candidate
        for candidate in summary.candidates
        if candidate.outcome is LocatorCandidateAssessmentOutcome.MATCH
    )


def _resolution(
    validation: CitationValidation,
    *,
    outcome: LocatorIdentityResolutionOutcome,
    matching_candidate_indices: tuple[int, ...],
    selection_evidence_node_id: str | None,
    depends_on: tuple[str, ...],
    status_message: str,
    outcome_message: str,
    selected_candidate_index: int | None = None,
    selected_assessment_node_id: str | None = None,
) -> LocatorIdentityResolutionNode:
    return LocatorIdentityResolutionNode(
        node_id=f"{validation.citation_id}:locator_identity_resolution",
        status=ValidationNodeStatus.SUCCEEDED,
        outcome=outcome,
        selected_candidate_index=selected_candidate_index,
        selected_assessment_node_id=selected_assessment_node_id,
        matching_candidate_indices=matching_candidate_indices,
        selection_evidence_node_id=selection_evidence_node_id,
        depends_on=depends_on,
        status_message=status_message,
        outcome_message=outcome_message,
    )
