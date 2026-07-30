"""Candidate assessment and terminal summary for one found locator."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mellea_lrc.validation.aggregation.citation_summary_candidate import citation_summary_candidate
from mellea_lrc.validation.aggregation.citation_summary_outcome import overall_citation_outcome
from mellea_lrc.validation.types import (
    AggregatedFieldOutcome,
    CandidateEvaluationNode,
    CandidateProvenance,
    CitationSummaryCandidate,
    CourtCheckNode,
    DocketCourtRetrievalNode,
    ExactCaseNameCheckNode,
    ExactLocatorLookupNode,
    FieldCheckOutcome,
    LocatorCandidateAssessmentNode,
    LocatorCandidateAssessmentOutcome,
    LocatorCitationSummaryNode,
    LocatorCitationSummaryOutcome,
    LocatorLookupOutcome,
    MelleaCaseNameCheckNode,
    MelleaCaseNameCheckOutcome,
    MelleaPinpointCheckOutcome,
    MelleaReextractedCaseNameCheckNode,
    ValidationNodeStatus,
    YearCheckNode,
)

if TYPE_CHECKING:
    from mellea_lrc.validation.types import CitationValidation


def run_locator_candidate_assessment(
    validation: CitationValidation,
    *,
    candidate: CandidateEvaluationNode,
) -> LocatorCandidateAssessmentNode:
    """Reduce one complete unique-locator candidate subtree into a conclusion."""
    exact = _required_child(validation, ExactCaseNameCheckNode, candidate.node_id)
    year = _required_child(validation, YearCheckNode, candidate.node_id)
    court = _required_court_check(validation, candidate.node_id)
    case_name_outcome, evidence, case_name_node_id = _case_name_conclusion(validation, exact)
    year_outcome = _field_outcome(year.outcome)
    court_outcome = _field_outcome(court.outcome)
    outcome = _assessment_outcome(case_name_outcome, year_outcome, court_outcome)
    return LocatorCandidateAssessmentNode(
        node_id=f"{candidate.node_id}:locator_candidate_assessment",
        status=ValidationNodeStatus.SUCCEEDED,
        outcome=outcome,
        candidate_index=candidate.candidate_index,
        extracted_citation=validation.citation.matched_text,
        extracted_case_name=exact.extracted_case_name,
        retrieved_case_name=exact.retrieved_case_name,
        case_name_outcome=case_name_outcome,
        case_name_evidence=evidence,
        extracted_year=year.extracted_year,
        retrieved_year=year.retrieved_year,
        year_outcome=year_outcome,
        extracted_court_id=court.extracted_court_id,
        retrieved_court_id=court.retrieved_court_id,
        court_outcome=court_outcome,
        docket_id=candidate.docket_id,
        depends_on=(case_name_node_id, year.node_id, court.node_id),
        status_message="Locator candidate assessment completed.",
        outcome_message=_assessment_message(outcome),
    )


def run_locator_citation_summary(
    validation: CitationValidation,
    *,
    assessment: LocatorCandidateAssessmentNode,
) -> LocatorCitationSummaryNode:
    """Expose the candidate assessment as the terminal found-locator summary."""
    candidate = citation_summary_candidate(
        validation,
        assessment,
        provenance=CandidateProvenance.OPINION,
    )
    return LocatorCitationSummaryNode(
        node_id=f"{validation.citation_id}:locator_citation_summary",
        status=ValidationNodeStatus.SUCCEEDED,
        outcome=LocatorCitationSummaryOutcome.COMPLETE,
        overall_outcome=overall_citation_outcome((candidate.outcome,)),
        pinpoint_requires_review=_pinpoint_requires_review(validation, (candidate,)),
        candidates=(candidate,),
        depends_on=(
            assessment.node_id,
            *((candidate.pinpoint.node_id,) if candidate.pinpoint is not None else ()),
        ),
        status_message="Locator citation summary completed.",
        outcome_message="Listed the one fully evaluated locator candidate.",
    )


def _pinpoint_requires_review(
    validation: CitationValidation,
    candidates: tuple[CitationSummaryCandidate, ...],
) -> bool | None:
    """Flag a matched unique locator whose pinpoint check did not support it."""
    lookup = next(
        (node for node in validation.nodes if isinstance(node, ExactLocatorLookupNode)),
        None,
    )
    # Temporary found-route-only aggregation: reporter-page and pinpoint work
    # currently run only below one exact locator result. Revisit this signal
    # when candidate aggregation gains pinpoint semantics for other routes.
    if lookup is None or lookup.outcome is not LocatorLookupOutcome.FOUND:
        return None
    matched = tuple(
        candidate for candidate in candidates if candidate.outcome is LocatorCandidateAssessmentOutcome.MATCH
    )
    if not matched:
        return None
    return any(
        candidate.pinpoint is None or candidate.pinpoint.outcome is not MelleaPinpointCheckOutcome.SUPPORTS
        for candidate in matched
    )


def _case_name_conclusion(
    validation: CitationValidation,
    exact: ExactCaseNameCheckNode,
) -> tuple[AggregatedFieldOutcome, str, str]:
    if exact.outcome is FieldCheckOutcome.MATCH:
        return AggregatedFieldOutcome.MATCH, "exact", exact.node_id
    if exact.outcome is FieldCheckOutcome.UNAVAILABLE:
        return AggregatedFieldOutcome.UNAVAILABLE, "exact", exact.node_id
    reextracted = _last_node(validation, MelleaReextractedCaseNameCheckNode)
    if reextracted is not None:
        return _mellea_outcome(reextracted.outcome), "mellea_reextracted", reextracted.node_id
    semantic = _last_node(validation, MelleaCaseNameCheckNode)
    if semantic is not None:
        return _mellea_outcome(semantic.outcome), "mellea", semantic.node_id
    return AggregatedFieldOutcome.MISMATCH, "exact", exact.node_id


def _assessment_outcome(
    case_name: AggregatedFieldOutcome,
    year: AggregatedFieldOutcome,
    court: AggregatedFieldOutcome,
) -> LocatorCandidateAssessmentOutcome:
    if case_name is AggregatedFieldOutcome.MISMATCH:
        return LocatorCandidateAssessmentOutcome.MISMATCH
    if case_name is not AggregatedFieldOutcome.MATCH:
        return LocatorCandidateAssessmentOutcome.INCONCLUSIVE
    if AggregatedFieldOutcome.MISMATCH in (year, court):
        return LocatorCandidateAssessmentOutcome.PARTIAL_MATCH
    return LocatorCandidateAssessmentOutcome.MATCH


def _assessment_message(outcome: LocatorCandidateAssessmentOutcome) -> str:
    return {
        LocatorCandidateAssessmentOutcome.MATCH: "Case name, year, and court do not disagree with this candidate.",
        LocatorCandidateAssessmentOutcome.MISMATCH: "The retrieved candidate has a different case name.",
        LocatorCandidateAssessmentOutcome.PARTIAL_MATCH: "Case name matches; verify the differing year or court.",
        LocatorCandidateAssessmentOutcome.INCONCLUSIVE: "Available evidence does not permit a candidate conclusion.",
    }[outcome]


def _field_outcome(outcome: FieldCheckOutcome) -> AggregatedFieldOutcome:
    return AggregatedFieldOutcome(outcome.value)


def _mellea_outcome(outcome: MelleaCaseNameCheckOutcome) -> AggregatedFieldOutcome:
    if outcome is MelleaCaseNameCheckOutcome.MATCH:
        return AggregatedFieldOutcome.MATCH
    if outcome is MelleaCaseNameCheckOutcome.MISMATCH:
        return AggregatedFieldOutcome.MISMATCH
    return AggregatedFieldOutcome.FAILED


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


def _last_node(
    validation: CitationValidation,
    node_type: type[MelleaCaseNameCheckNode] | type[MelleaReextractedCaseNameCheckNode],
) -> MelleaCaseNameCheckNode | MelleaReextractedCaseNameCheckNode | None:
    return next((node for node in reversed(validation.nodes) if isinstance(node, node_type)), None)
