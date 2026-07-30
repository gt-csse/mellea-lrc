"""Assessment projection for one completed opinion-search candidate."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

from mellea_lrc.validation.types import (
    AggregatedFieldOutcome,
    CandidateEvaluationNode,
    CandidateEvaluationSource,
    CourtCheckNode,
    ExactCaseNameCheckNode,
    FieldCheckOutcome,
    MelleaCaseNameCheckNode,
    MelleaCaseNameCheckOutcome,
    OpinionSearchCandidateAssessmentNode,
    SearchCandidateAssessmentOutcome,
    ValidationNode,
    ValidationNodeStatus,
    YearCheckNode,
)

if TYPE_CHECKING:
    from mellea_lrc.validation.types import CitationValidation


NodeT = TypeVar("NodeT", bound=ValidationNode)


def run_opinion_search_candidate_assessment(
    validation: CitationValidation,
    *,
    candidate: CandidateEvaluationNode,
) -> OpinionSearchCandidateAssessmentNode:
    """Reduce one completed opinion-search subtree into a candidate conclusion."""
    if candidate.source is not CandidateEvaluationSource.OPINION_SEARCH:
        msg = "Opinion-search assessment requires an opinion-search candidate"
        raise ValueError(msg)
    exact = _required_child(validation, ExactCaseNameCheckNode, candidate.node_id)
    year = _required_child(validation, YearCheckNode, candidate.node_id)
    court = _required_child(validation, CourtCheckNode, candidate.node_id)
    case_name_outcome, case_name_evidence, case_name_node_id = _case_name_conclusion(
        validation,
        exact,
    )
    year_outcome = _field_outcome(year.outcome)
    court_outcome = _field_outcome(court.outcome)
    outcome = _assessment_outcome(case_name_outcome, year_outcome, court_outcome)
    return OpinionSearchCandidateAssessmentNode(
        node_id=f"{candidate.node_id}:opinion_search_candidate_assessment",
        status=ValidationNodeStatus.SUCCEEDED,
        outcome=outcome,
        candidate_index=candidate.candidate_index,
        extracted_citation=validation.citation.matched_text,
        extracted_case_name=exact.extracted_case_name,
        retrieved_case_name=exact.retrieved_case_name,
        case_name_outcome=case_name_outcome,
        case_name_evidence=case_name_evidence,
        extracted_year=year.extracted_year,
        retrieved_year=year.retrieved_year,
        year_outcome=year_outcome,
        extracted_court_id=court.extracted_court_id,
        retrieved_court_id=court.retrieved_court_id,
        court_outcome=court_outcome,
        docket_id=candidate.docket_id,
        depends_on=(case_name_node_id, year.node_id, court.node_id),
        status_message="Opinion-search candidate assessment completed.",
        outcome_message=_assessment_message(outcome),
    )


def _case_name_conclusion(
    validation: CitationValidation,
    exact: ExactCaseNameCheckNode,
) -> tuple[AggregatedFieldOutcome, str, str]:
    if exact.outcome is FieldCheckOutcome.MATCH:
        return AggregatedFieldOutcome.MATCH, "exact", exact.node_id
    if exact.outcome is FieldCheckOutcome.UNAVAILABLE:
        return AggregatedFieldOutcome.UNAVAILABLE, "exact", exact.node_id
    semantic_checks = [
        node
        for node in validation.nodes
        if isinstance(node, MelleaCaseNameCheckNode) and exact.node_id in node.depends_on
    ]
    if not semantic_checks:
        return AggregatedFieldOutcome.MISMATCH, "exact", exact.node_id
    semantic = semantic_checks[-1]
    return _mellea_outcome(semantic.outcome), "mellea", semantic.node_id


def _assessment_outcome(
    case_name: AggregatedFieldOutcome,
    year: AggregatedFieldOutcome,
    court: AggregatedFieldOutcome,
) -> SearchCandidateAssessmentOutcome:
    if (
        case_name is AggregatedFieldOutcome.MATCH
        and year is AggregatedFieldOutcome.MATCH
        and court is AggregatedFieldOutcome.MATCH
    ):
        return SearchCandidateAssessmentOutcome.POSSIBLE_MATCH
    return SearchCandidateAssessmentOutcome.MISMATCH


def _assessment_message(outcome: SearchCandidateAssessmentOutcome) -> str:
    if outcome is SearchCandidateAssessmentOutcome.POSSIBLE_MATCH:
        return "Case name, year, and court support this opinion as a possible match."
    return "Available opinion-search evidence does not support this candidate as a possible match."


def _field_outcome(outcome: FieldCheckOutcome) -> AggregatedFieldOutcome:
    return AggregatedFieldOutcome(outcome.value)


def _mellea_outcome(outcome: MelleaCaseNameCheckOutcome) -> AggregatedFieldOutcome:
    if outcome is MelleaCaseNameCheckOutcome.MATCH:
        return AggregatedFieldOutcome.MATCH
    if outcome is MelleaCaseNameCheckOutcome.MISMATCH:
        return AggregatedFieldOutcome.MISMATCH
    if outcome is MelleaCaseNameCheckOutcome.UNAVAILABLE:
        return AggregatedFieldOutcome.UNAVAILABLE
    return AggregatedFieldOutcome.FAILED


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
        msg = (
            "Opinion-search candidate assessment requires one "
            f"{node_type.__name__} below {candidate_node_id!r}"
        )
        raise ValueError(msg)
    return matches[0]
