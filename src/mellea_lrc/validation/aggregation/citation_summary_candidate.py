"""Flat candidate projections shared by terminal citation summaries."""

from __future__ import annotations

from mellea_lrc.validation.types import (
    CandidateProvenance,
    CitationSummaryCandidate,
    LocatorCandidateAssessmentNode,
    OpinionSearchCandidateAssessmentNode,
    RecapSearchCandidateAssessmentNode,
)

CandidateAssessmentNode = (
    LocatorCandidateAssessmentNode | OpinionSearchCandidateAssessmentNode | RecapSearchCandidateAssessmentNode
)


def citation_summary_candidate(
    assessment: CandidateAssessmentNode,
    *,
    provenance: CandidateProvenance,
) -> CitationSummaryCandidate:
    """Project one candidate assessment into the shared summary contract."""
    return CitationSummaryCandidate(
        provenance=provenance,
        candidate_index=assessment.candidate_index,
        assessment_node_id=assessment.node_id,
        outcome=assessment.outcome,
        extracted_citation=assessment.extracted_citation,
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
        docket_id=assessment.docket_id,
    )
