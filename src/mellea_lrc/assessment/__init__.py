"""Mellea-assisted citation assessment helpers."""

from mellea_lrc.assessment.citation import (
    CitationAssessmentBundle,
    assess_found_citation,
    first_cluster_case_name,
    first_cluster_year,
)
from mellea_lrc.assessment.deterministic import (
    assess_case_name_exact_match,
    assess_year_exact_match,
    build_extracted_case_name,
    find_text_span_near_full_span,
    get_extended_span_text,
)
from mellea_lrc.assessment.pipeline import MelleaCallContext, run_assessment, run_assessment_async
from mellea_lrc.assessment.llm import (
    ReextractionStatus,
    assess_case_name_with_mellea,
    reextract_case_name,
    validate_proposal,
)
from mellea_lrc.assessment.types import (
    CaseNameAssessment,
    CaseNameAssessmentRun,
    CaseNameAssessmentStatus,
    CitationAssessment,
    DocumentAssessment,
    ModifiedExtractedCitation,
    ModifiedExtractedCitationProposal,
    YearAssessment,
    YearAssessmentStatus,
)

__all__ = [
    "CaseNameAssessment",
    "CaseNameAssessmentRun",
    "CaseNameAssessmentStatus",
    "CitationAssessment",
    "CitationAssessmentBundle",
    "DocumentAssessment",
    "MelleaCallContext",
    "ModifiedExtractedCitation",
    "ModifiedExtractedCitationProposal",
    "ReextractionStatus",
    "YearAssessment",
    "YearAssessmentStatus",
    "assess_case_name_exact_match",
    "assess_case_name_with_mellea",
    "assess_found_citation",
    "assess_year_exact_match",
    "build_extracted_case_name",
    "find_text_span_near_full_span",
    "first_cluster_case_name",
    "first_cluster_year",
    "get_extended_span_text",
    "reextract_case_name",
    "run_assessment",
    "run_assessment_async",
    "validate_proposal",
]
