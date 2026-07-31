"""Shared field-evidence reduction for candidates returned by a search corpus."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar

from mellea_lrc.validation.types import (
    AggregatedFieldOutcome,
    CandidateEvaluationNode,
    CandidateEvaluationSource,
    CourtCheckNode,
    FieldCheckOutcome,
    SearchCandidateAssessmentOutcome,
    ValidationNode,
    YearCheckNode,
)

if TYPE_CHECKING:
    from mellea_lrc.validation.candidate_state import CandidateValidationState
    from mellea_lrc.validation.types import CitationValidation


NodeT = TypeVar("NodeT", bound=ValidationNode)


@dataclass(frozen=True, slots=True)
class SearchCandidateAssessment:
    """Common assessment facts projected by opinion and RECAP search routes."""

    outcome: SearchCandidateAssessmentOutcome
    extracted_case_name: str | None
    retrieved_case_name: str | None
    case_name_outcome: AggregatedFieldOutcome
    case_name_evidence: str
    extracted_year: str | None
    retrieved_year: str | None
    year_outcome: AggregatedFieldOutcome
    extracted_court_id: str | None
    retrieved_court_id: str | None
    court_outcome: AggregatedFieldOutcome
    depends_on: tuple[str, str, str]


def evaluate_search_candidate(
    validation: CitationValidation,
    *,
    candidate: CandidateEvaluationNode,
    state: CandidateValidationState,
) -> SearchCandidateAssessment:
    """Reduce the common case-name, year, and court checks for one candidate."""
    year = _required_child(validation, YearCheckNode, candidate.node_id)
    court = _required_child(validation, CourtCheckNode, candidate.node_id)
    case_name = state.require_case_name_result()
    year_outcome = _field_outcome(year.outcome)
    court_outcome = _field_outcome(court.outcome)
    outcome = _assessment_outcome(
        candidate,
        case_name.outcome,
        year_outcome,
        court_outcome,
    )
    return SearchCandidateAssessment(
        outcome=outcome,
        extracted_case_name=state.displayed_case_name(None),
        retrieved_case_name=candidate.case_name,
        case_name_outcome=case_name.outcome,
        case_name_evidence=case_name.evidence,
        extracted_year=year.extracted_year,
        retrieved_year=year.retrieved_year,
        year_outcome=year_outcome,
        extracted_court_id=court.extracted_court_id,
        retrieved_court_id=court.retrieved_court_id,
        court_outcome=court_outcome,
        depends_on=(case_name.dependency_id, year.node_id, court.node_id),
    )


def assessment_message(
    candidate: CandidateEvaluationNode,
    outcome: SearchCandidateAssessmentOutcome,
) -> str:
    """Describe the intentionally limited conclusion from search metadata."""
    if outcome is SearchCandidateAssessmentOutcome.POSSIBLE_MATCH:
        if candidate.source is CandidateEvaluationSource.RECAP_SEARCH:
            return "Case name and court support this retrieved RECAP candidate as a possible match."
        return "Case name, year, and court support this retrieved candidate as a possible match."
    return "Available search evidence does not support this candidate as a possible match."


def _assessment_outcome(
    candidate: CandidateEvaluationNode,
    case_name: AggregatedFieldOutcome,
    year: AggregatedFieldOutcome,
    court: AggregatedFieldOutcome,
) -> SearchCandidateAssessmentOutcome:
    if candidate.source is CandidateEvaluationSource.RECAP_SEARCH:
        return (
            SearchCandidateAssessmentOutcome.POSSIBLE_MATCH
            if case_name is AggregatedFieldOutcome.MATCH and court is AggregatedFieldOutcome.MATCH
            else SearchCandidateAssessmentOutcome.MISMATCH
        )
    if (
        case_name is AggregatedFieldOutcome.MATCH
        and year is AggregatedFieldOutcome.MATCH
        and court is AggregatedFieldOutcome.MATCH
    ):
        return SearchCandidateAssessmentOutcome.POSSIBLE_MATCH
    return SearchCandidateAssessmentOutcome.MISMATCH


def _field_outcome(outcome: FieldCheckOutcome) -> AggregatedFieldOutcome:
    return AggregatedFieldOutcome(outcome.value)


def _required_child(
    validation: CitationValidation,
    node_type: type[NodeT],
    candidate_node_id: str,
) -> NodeT:
    matches = [
        node
        for node in validation.nodes
        if isinstance(node, node_type) and candidate_node_id in node.depends_on
    ]
    if len(matches) != 1:
        msg = f"Search-candidate assessment requires one {node_type.__name__} below {candidate_node_id!r}"
        raise ValueError(msg)
    return matches[0]
