"""Assessment projection for one completed RECAP-search candidate."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mellea_lrc.validation.aggregation.search_candidate import (
    assessment_message,
    evaluate_search_candidate,
)
from mellea_lrc.validation.types import (
    CandidateEvaluationNode,
    CandidateEvaluationSource,
    RecapSearchCandidateAssessmentNode,
    ValidationNodeStatus,
)

if TYPE_CHECKING:
    from mellea_lrc.validation.types import CitationValidation


def run_recap_search_candidate_assessment(
    validation: CitationValidation,
    *,
    candidate: CandidateEvaluationNode,
) -> RecapSearchCandidateAssessmentNode:
    """Reduce one completed RECAP-search subtree into a candidate conclusion."""
    if candidate.source is not CandidateEvaluationSource.RECAP_SEARCH:
        msg = "RECAP-search assessment requires a RECAP-search candidate"
        raise ValueError(msg)
    assessment = evaluate_search_candidate(validation, candidate=candidate)
    return RecapSearchCandidateAssessmentNode(
        node_id=f"{candidate.node_id}:recap_search_candidate_assessment",
        status=ValidationNodeStatus.SUCCEEDED,
        outcome=assessment.outcome,
        candidate_index=candidate.candidate_index,
        extracted_citation=validation.citation.matched_text,
        extracted_case_name=assessment.extracted_case_name,
        retrieved_case_name=assessment.retrieved_case_name,
        case_name_outcome=assessment.case_name_outcome,
        case_name_evidence=assessment.case_name_evidence,
        extracted_year=assessment.extracted_year,
        retrieved_year=assessment.retrieved_year,
        year_outcome=assessment.year_outcome,
        extracted_court_id=assessment.extracted_court_id,
        retrieved_court_id=assessment.retrieved_court_id,
        court_outcome=assessment.court_outcome,
        docket_id=candidate.docket_id,
        depends_on=assessment.depends_on,
        status_message="RECAP-search candidate assessment completed.",
        outcome_message=assessment_message(candidate, assessment.outcome),
    )
