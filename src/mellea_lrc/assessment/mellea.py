"""Mellea-backed semantic assessment for citations."""

from typing import Literal, cast

from mellea import MelleaSession, generative

from mellea_lrc.assessment.case_name import assess_case_name_exact_match
from mellea_lrc.assessment.types import CaseNameAssessment, CaseNameAssessmentStatus

CaseNameSemanticVerdict = Literal["semantic_match", "extraction_error"]


@generative
def classify_case_name_semantic_match(
    document_context: str,
    extracted_case_name: str,
    courtlistener_case_name: str,
) -> CaseNameSemanticVerdict:
    """Assess whether an extracted legal case name is acceptable.

    Return "semantic_match" when the extracted case name is a valid legal citation
    form for the same case identified by the CourtListener case name. Treat common
    shortening, party-name abbreviation, omitted institutional suffixes, and normal
    legal citation style variation as acceptable when the surrounding document
    context supports that reading.

    Return "extraction_error" when the extracted case name identifies a different
    case, omits a party that should have been extracted from the visible citation
    context, includes text that is not part of the case name, or otherwise appears
    incorrectly extracted.

    Do not judge volume, reporter, page, pin cite, court, or year.
    """


def assess_case_name_with_mellea(
    session: MelleaSession,
    *,
    citation_id: str,
    extracted_case_name: str | None,
    courtlistener_case_name: str | None,
    document_context: str,
) -> CaseNameAssessment:
    """Assess one case name, using Mellea only when exact equality is not enough."""
    exact_result = assess_case_name_exact_match(
        citation_id=citation_id,
        extracted_case_name=extracted_case_name,
        courtlistener_case_name=courtlistener_case_name,
    )
    if exact_result.status != CaseNameAssessmentStatus.NEEDS_SEMANTIC_ASSESSMENT:
        return exact_result

    verdict = classify_case_name_semantic_match(
        session,
        document_context=document_context,
        extracted_case_name=cast("str", extracted_case_name),
        courtlistener_case_name=cast("str", courtlistener_case_name),
    )
    status = CaseNameAssessmentStatus(cast("str", verdict))
    return CaseNameAssessment(
        citation_id=citation_id,
        status=status,
        extracted_case_name=extracted_case_name,
        courtlistener_case_name=courtlistener_case_name,
        message=_message_for_semantic_status(status),
    )


def _message_for_semantic_status(status: CaseNameAssessmentStatus) -> str:
    if status == CaseNameAssessmentStatus.SEMANTIC_MATCH:
        return "Extracted case name semantically matches CourtListener."
    return "Extracted case name does not appear to match the CourtListener case."
