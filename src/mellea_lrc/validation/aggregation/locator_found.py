"""Candidate assessment and terminal summary for one found locator."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mellea_lrc.validation.aggregation.citation_summary_candidate import (
    citation_summary_candidate,
)
from mellea_lrc.validation.aggregation.citation_summary_outcome import (
    overall_citation_outcome,
)
from mellea_lrc.validation.types import (
    AggregatedFieldOutcome,
    CandidateEvaluationNode,
    CandidateProvenance,
    CourtCheckNode,
    DocketCourtRetrievalNode,
    ExactCaseNameCheckNode,
    FieldCheckOutcome,
    LocatorCandidateAssessmentNode,
    LocatorCandidateAssessmentOutcome,
    LocatorCitationSummaryNode,
    LocatorCitationSummaryOutcome,
    ValidationNodeStatus,
    YearCheckNode,
)

if TYPE_CHECKING:
    from mellea_lrc.validation.candidate_state import CandidateValidationState
    from mellea_lrc.validation.types import CitationValidation


def run_locator_candidate_assessment(
    validation: CitationValidation,
    *,
    candidate: CandidateEvaluationNode,
    state: CandidateValidationState,
) -> LocatorCandidateAssessmentNode:
    """Reduce one completed locator candidate subtree into a conclusion."""
    exact = _required_child(validation, ExactCaseNameCheckNode, candidate.node_id)
    year = _required_child(validation, YearCheckNode, candidate.node_id)
    court = _required_court_check(validation, candidate.node_id)
    case_name = state.require_case_name_result()
    year_outcome = _field_outcome(year.outcome)
    court_outcome = _field_outcome(court.outcome)
    outcome = _assessment_outcome(case_name.outcome, year_outcome, court_outcome)
    return LocatorCandidateAssessmentNode(
        node_id=f"{candidate.node_id}:locator_candidate_assessment",
        status=ValidationNodeStatus.SUCCEEDED,
        outcome=outcome,
        candidate_index=candidate.candidate_index,
        extracted_citation=validation.citation.matched_text,
        extracted_case_name=state.displayed_case_name(exact.extracted_case_name),
        retrieved_case_name=exact.retrieved_case_name,
        case_name_outcome=case_name.outcome,
        case_name_evidence=case_name.evidence,
        extracted_year=year.extracted_year,
        retrieved_year=year.retrieved_year,
        year_outcome=year_outcome,
        extracted_court_id=court.extracted_court_id,
        retrieved_court_id=court.retrieved_court_id,
        court_outcome=court_outcome,
        docket_id=candidate.docket_id,
        depends_on=(case_name.dependency_id, year.node_id, court.node_id),
        status_message="Locator candidate assessment completed.",
        outcome_message=_assessment_message(outcome),
    )


def run_locator_citation_summary(validation: CitationValidation) -> LocatorCitationSummaryNode:
    """List every evaluated locator candidate without selecting one."""
    assessments = tuple(node for node in validation.nodes if isinstance(node, LocatorCandidateAssessmentNode))
    if not assessments:
        msg = "Locator citation summary requires at least one candidate assessment"
        raise ValueError(msg)
    candidates = tuple(
        citation_summary_candidate(
            assessment,
            provenance=CandidateProvenance.OPINION,
        )
        for assessment in assessments
    )
    return LocatorCitationSummaryNode(
        node_id=f"{validation.citation_id}:locator_citation_summary",
        status=ValidationNodeStatus.SUCCEEDED,
        outcome=LocatorCitationSummaryOutcome.COMPLETE,
        overall_outcome=overall_citation_outcome(candidate.outcome for candidate in candidates),
        candidates=candidates,
        depends_on=tuple(candidate.assessment_node_id for candidate in candidates),
        status_message="Locator citation summary completed.",
        outcome_message=(f"Listed {len(candidates)} evaluated locator candidates without selecting one."),
    )


def _assessment_outcome(
    case_name: AggregatedFieldOutcome,
    year: AggregatedFieldOutcome,
    court: AggregatedFieldOutcome,
) -> LocatorCandidateAssessmentOutcome:
    if case_name is AggregatedFieldOutcome.MISMATCH:
        return LocatorCandidateAssessmentOutcome.MISMATCH
    if case_name is not AggregatedFieldOutcome.MATCH:
        return LocatorCandidateAssessmentOutcome.MISMATCH
    if AggregatedFieldOutcome.MISMATCH in (year, court):
        return LocatorCandidateAssessmentOutcome.PARTIAL_MATCH
    return LocatorCandidateAssessmentOutcome.MATCH


def _assessment_message(outcome: LocatorCandidateAssessmentOutcome) -> str:
    return {
        LocatorCandidateAssessmentOutcome.MATCH: "Case name, year, and court do not disagree with this candidate.",
        LocatorCandidateAssessmentOutcome.MISMATCH: "The retrieved candidate has a different case name.",
        LocatorCandidateAssessmentOutcome.PARTIAL_MATCH: "Case name matches; verify the differing year or court.",
    }[outcome]


def _field_outcome(outcome: FieldCheckOutcome) -> AggregatedFieldOutcome:
    return AggregatedFieldOutcome(outcome.value)


def _required_child(
    validation: CitationValidation,
    node_type: type[ExactCaseNameCheckNode] | type[YearCheckNode],
    candidate_node_id: str,
) -> ExactCaseNameCheckNode | YearCheckNode:
    matches = [
        node
        for node in validation.nodes
        if isinstance(node, node_type) and candidate_node_id in node.depends_on
    ]
    if len(matches) != 1:
        msg = f"Locator candidate assessment requires one {node_type.__name__} below {candidate_node_id!r}"
        raise ValueError(msg)
    return matches[0]


def _required_court_check(validation: CitationValidation, candidate_node_id: str) -> CourtCheckNode:
    docket = _required_child(validation, DocketCourtRetrievalNode, candidate_node_id)
    matches = [
        node
        for node in validation.nodes
        if isinstance(node, CourtCheckNode) and docket.node_id in node.depends_on
    ]
    if len(matches) != 1:
        msg = f"Locator candidate assessment requires one court check below {candidate_node_id!r}"
        raise ValueError(msg)
    return matches[0]
