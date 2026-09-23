"""Terminal reporter-locator identity decisions from complete candidate evidence."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mellea_lrc.validation.aggregation.locator_found import locator_candidate_contains_reporter_locator
from mellea_lrc.validation.types import (
    AggregatedFieldOutcome,
    CandidateEvaluationNode,
    CandidateEvaluationSource,
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


def run_unique_exact_locator_identity_resolution(
    validation: CitationValidation,
    *,
    summary: LocatorCitationSummaryNode,
) -> LocatorIdentityResolutionNode:
    """Decide a one-record exact lookup from its completed field assessments.

    The unique route has no representative to select. A model may reread and
    compare the local name once, but a later candidate-choice model must not
    override that field result.
    """
    if len(summary.candidates) != 1:
        raise ValueError("Unique exact-locator identity requires exactly one retrieved candidate")
    candidate = summary.candidates[0]
    if not _selected_candidate_contains_reporter_locator(
        validation, candidate_index=candidate.candidate_index
    ):
        outcome = LocatorIdentityResolutionOutcome.NO_MATCH
        reason = "The retrieved record does not contain the source reporter locator."
    elif candidate.case_name_outcome is AggregatedFieldOutcome.MISMATCH:
        outcome = LocatorIdentityResolutionOutcome.NO_MATCH
        reason = "The source-grounded case-name review disagrees with the retrieved record."
    elif candidate.court_outcome is AggregatedFieldOutcome.MISMATCH:
        outcome = LocatorIdentityResolutionOutcome.NO_MATCH
        reason = "The retrieved court conflicts with the citation's stated court."
    elif candidate.year_outcome is AggregatedFieldOutcome.MISMATCH:
        outcome = LocatorIdentityResolutionOutcome.NO_MATCH
        reason = "The retrieved opinion date conflicts with the cited date."
    elif candidate.case_name_outcome is not AggregatedFieldOutcome.MATCH:
        outcome = LocatorIdentityResolutionOutcome.DEFERRED_TO_SEARCH
        reason = "The case name could not be reviewed against the unique exact-lookup record."
    else:
        outcome = LocatorIdentityResolutionOutcome.RESOLVED
        reason = "The unique exact-lookup record agrees with the completed identity checks."
    resolved = outcome is LocatorIdentityResolutionOutcome.RESOLVED
    return _resolution(
        validation,
        outcome=outcome,
        selected_candidate_index=candidate.candidate_index if resolved else None,
        selected_assessment_node_id=candidate.assessment_node_id if resolved else None,
        matching_candidate_indices=(candidate.candidate_index,) if resolved else (),
        selection_evidence_node_id=summary.node_id,
        depends_on=(summary.node_id,),
        status_message="Unique exact-locator identity resolution completed.",
        outcome_message=reason,
    )


def run_locator_identity_resolution(
    validation: CitationValidation,
    *,
    summary: LocatorCitationSummaryNode,
    choice: MelleaLocatorCandidateChoiceNode | None = None,
) -> LocatorIdentityResolutionNode:
    """Resolve a reviewed reporter locator from deterministic or model evidence.

    Exactly one confirmed candidate is admitted deterministically.  With zero
    or multiple confirmations, a grounded model selects one of the complete
    candidate list or reports no match. Field re-extraction is a separate
    operation and appears in the trace only when it actually ran. Candidate
    evidence stays in ``summary`` either way.
    """
    matches = _matching_candidates(summary)
    matching_candidate_indices = tuple(candidate.candidate_index for candidate in matches)
    if len(summary.candidates) == 1 and choice is None:
        candidate = summary.candidates[0]
        if (
            candidate.case_name_outcome is AggregatedFieldOutcome.MATCH
            and candidate.court_outcome is not AggregatedFieldOutcome.MISMATCH
            and candidate.year_outcome is AggregatedFieldOutcome.MISMATCH
            and _selected_candidate_contains_reporter_locator(
                validation, candidate_index=candidate.candidate_index
            )
        ):
            return _resolution(
                validation,
                outcome=LocatorIdentityResolutionOutcome.NO_MATCH,
                matching_candidate_indices=matching_candidate_indices,
                selection_evidence_node_id=summary.node_id,
                depends_on=(summary.node_id,),
                status_message="Locator identity resolution completed.",
                outcome_message="The retrieved opinion date conflicts with the cited date.",
            )
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
            outcome_message=(f"Candidate {candidate.candidate_index} is the only confirmed locator match."),
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
        if not _selected_candidate_contains_reporter_locator(
            validation,
            candidate_index=candidate.candidate_index,
        ):
            return _resolution(
                validation,
                outcome=LocatorIdentityResolutionOutcome.NO_MATCH,
                matching_candidate_indices=matching_candidate_indices,
                selection_evidence_node_id=choice.node_id,
                depends_on=(summary.node_id, choice.node_id),
                status_message="Locator identity resolution completed.",
                outcome_message=(
                    f"Model selected candidate {candidate.candidate_index}, but the provider's citation "
                    "record does not contain the source reporter locator."
                ),
            )
        if _requires_future_court_or_date_semantics(candidate):
            conflicts = _conflicting_fields(candidate)
            return _resolution(
                validation,
                outcome=LocatorIdentityResolutionOutcome.NO_MATCH,
                matching_candidate_indices=matching_candidate_indices,
                selection_evidence_node_id=choice.node_id,
                depends_on=(summary.node_id, choice.node_id),
                status_message="Locator identity resolution completed.",
                outcome_message=(
                    f"Model selected candidate {candidate.candidate_index}, but its deterministic "
                    f"{conflicts} comparison conflicts with the stated fields."
                ),
            )
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
                f"Grounded model choice selected candidate {candidate.candidate_index} "
                "from the reviewed evidence."
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
            outcome_message="No reviewed exact-locator candidate represents the stated citation.",
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


def _selected_candidate_contains_reporter_locator(
    validation: CitationValidation,
    *,
    candidate_index: int,
) -> bool:
    """Keep a semantic selection from overriding a contradictory locator."""
    evidence = next(
        (
            node
            for node in validation.nodes
            if isinstance(node, CandidateEvaluationNode)
            and node.source is CandidateEvaluationSource.LOCATOR_LOOKUP
            and node.candidate_index == candidate_index
        ),
        None,
    )
    return evidence is None or locator_candidate_contains_reporter_locator(validation, evidence)


def _requires_future_court_or_date_semantics(candidate) -> bool:
    """Keep unrepaired court/year disagreement out of an admitted identity."""
    return (
        candidate.court_outcome is AggregatedFieldOutcome.MISMATCH
        or candidate.year_outcome is AggregatedFieldOutcome.MISMATCH
    )


def _conflicting_fields(candidate) -> str:
    """Name the deterministic evidence that makes a selected identity wrong."""
    fields: list[str] = []
    if candidate.court_outcome is AggregatedFieldOutcome.MISMATCH:
        fields.append("court")
    if candidate.year_outcome is AggregatedFieldOutcome.MISMATCH:
        fields.append("date")
    return " and ".join(fields)


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
