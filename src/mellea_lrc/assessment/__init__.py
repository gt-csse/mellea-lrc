"""Mellea-assisted citation assessment helpers."""

from mellea_lrc.assessment.case_name import assess_case_name_exact_match, build_extracted_case_name
from mellea_lrc.assessment.citation import assess_year_exact_match
from mellea_lrc.assessment.context import find_text_span_near_full_span, get_extended_span_text
from mellea_lrc.assessment.pipeline import MelleaCallContext, run_assessment
from mellea_lrc.assessment.types import (
    CaseNameAssessment,
    CaseNameAssessmentRun,
    CaseNameAssessmentStatus,
    CitationAssessment,
    CitationAssessmentStatus,
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
    "CitationAssessmentStatus",
    "DocumentAssessment",
    "MelleaCallContext",
    "ModifiedExtractedCitation",
    "ModifiedExtractedCitationProposal",
    "YearAssessment",
    "YearAssessmentStatus",
    "assess_case_name_exact_match",
    "assess_year_exact_match",
    "build_extracted_case_name",
    "find_text_span_near_full_span",
    "get_extended_span_text",
    "run_assessment",
]
