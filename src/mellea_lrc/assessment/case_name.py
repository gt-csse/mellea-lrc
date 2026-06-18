"""Case-name assessment helpers."""

from mellea_lrc.assessment.types import CaseNameAssessment, CaseNameAssessmentStatus
from mellea_lrc.core.citations import FullCaseCitation


def build_extracted_case_name(citation: FullCaseCitation) -> str | None:
    """Build a display case name from extracted party fields."""
    if citation.plaintiff and citation.defendant:
        return f"{citation.plaintiff} v. {citation.defendant}"
    return citation.plaintiff or citation.defendant


def assess_case_name_exact_match(
    *,
    citation_id: str,
    extracted_case_name: str | None,
    courtlistener_case_name: str | None,
) -> CaseNameAssessment:
    """Short-circuit case-name assessment when exact string equality is enough."""
    if not extracted_case_name:
        return CaseNameAssessment(
            citation_id=citation_id,
            status=CaseNameAssessmentStatus.EXTRACTION_ERROR,
            extracted_case_name=extracted_case_name,
            courtlistener_case_name=courtlistener_case_name,
            message="No extracted case name is available.",
        )
    if not courtlistener_case_name:
        return CaseNameAssessment(
            citation_id=citation_id,
            status=CaseNameAssessmentStatus.EXTRACTION_ERROR,
            extracted_case_name=extracted_case_name,
            courtlistener_case_name=courtlistener_case_name,
            message="No CourtListener case name is available.",
        )
    if extracted_case_name == courtlistener_case_name:
        return CaseNameAssessment(
            citation_id=citation_id,
            status=CaseNameAssessmentStatus.EXACT_MATCH,
            extracted_case_name=extracted_case_name,
            courtlistener_case_name=courtlistener_case_name,
            message="Extracted case name exactly matches CourtListener.",
        )
    return CaseNameAssessment(
        citation_id=citation_id,
        status=CaseNameAssessmentStatus.NEEDS_SEMANTIC_ASSESSMENT,
        extracted_case_name=extracted_case_name,
        courtlistener_case_name=courtlistener_case_name,
        message="Extracted case name differs from CourtListener and needs semantic assessment.",
    )
