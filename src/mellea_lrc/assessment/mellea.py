"""Mellea-backed semantic assessment for citations."""

import json
from typing import Literal, cast

from mellea import MelleaSession, generative
from mellea.core import Requirement
from mellea.stdlib.requirements import simple_validate
from pydantic import TypeAdapter, ValidationError

from mellea_lrc.assessment.case_name import assess_case_name_exact_match
from mellea_lrc.assessment.types import (
    CaseNameAssessment,
    CaseNameAssessmentStatus,
    ModifiedExtractedCitation,
)

CaseNameSemanticVerdict = Literal["semantic_match", "extraction_error"]
MODIFIED_EXTRACTION_ADAPTER = TypeAdapter(ModifiedExtractedCitation)


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


@generative
def propose_modified_extracted_citation(
    document_context: str,
    extracted_case_name: str,
    courtlistener_case_name: str,
) -> ModifiedExtractedCitation:
    """Propose a corrected case-name extraction from the visible citation context.

    Return corrected plaintiff and defendant strings, or a corrected case_name string,
    only when the corrected text is visible in document_context. Copy strings exactly
    from document_context. Do not use the CourtListener case name to invent text that
    is not present in document_context.

    If no grounded correction is visible in document_context, return an empty
    ModifiedExtractedCitation.
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
    if exact_result.status == CaseNameAssessmentStatus.EXACT_MATCH:
        return exact_result
    if not courtlistener_case_name:
        return exact_result

    try:
        return _assess_case_name_with_mellea_after_exact(
            session,
            citation_id=citation_id,
            extracted_case_name=extracted_case_name,
            courtlistener_case_name=courtlistener_case_name,
            document_context=document_context,
        )
    except RuntimeError:
        raise
    except Exception as exc:
        msg = f"Mellea case-name assessment failed: {exc}"
        raise RuntimeError(msg) from exc


def _assess_case_name_with_mellea_after_exact(
    session: MelleaSession,
    *,
    citation_id: str,
    extracted_case_name: str | None,
    courtlistener_case_name: str,
    document_context: str,
) -> CaseNameAssessment:
    """Run semantic and modified-extraction case-name checks."""
    if extracted_case_name:
        verdict = classify_case_name_semantic_match(
            session,
            document_context=document_context,
            extracted_case_name=extracted_case_name,
            courtlistener_case_name=courtlistener_case_name,
        )
        status = CaseNameAssessmentStatus(cast("str", verdict))
        if status == CaseNameAssessmentStatus.SEMANTIC_MATCH:
            return CaseNameAssessment(
                citation_id=citation_id,
                status=status,
                extracted_case_name=extracted_case_name,
                courtlistener_case_name=courtlistener_case_name,
                message=_message_for_semantic_status(status),
            )

    modified = propose_modified_extracted_citation(
        session,
        document_context=document_context,
        extracted_case_name=extracted_case_name or "",
        courtlistener_case_name=courtlistener_case_name,
        requirements=[modified_extracted_citation_is_in_context_requirement(document_context)],
    )
    if not modified.valid(document_context):
        return CaseNameAssessment(
            citation_id=citation_id,
            status=CaseNameAssessmentStatus.EXTRACTION_ERROR,
            extracted_case_name=extracted_case_name,
            courtlistener_case_name=courtlistener_case_name,
            message="No grounded modified extraction was found in the document context.",
        )

    modified_case_name = modified.extracted_case_name
    modified_exact = assess_case_name_exact_match(
        citation_id=citation_id,
        extracted_case_name=modified_case_name,
        courtlistener_case_name=courtlistener_case_name,
    )
    if modified_exact.status == CaseNameAssessmentStatus.EXACT_MATCH:
        return _extraction_error_from_modified_match(
            citation_id=citation_id,
            extracted_case_name=extracted_case_name,
            courtlistener_case_name=courtlistener_case_name,
            modified_case_name=modified_case_name,
            modified_match_status=modified_exact.status,
        )

    modified_verdict = classify_case_name_semantic_match(
        session,
        document_context=document_context,
        extracted_case_name=cast("str", modified_case_name),
        courtlistener_case_name=courtlistener_case_name,
    )
    modified_status = CaseNameAssessmentStatus(cast("str", modified_verdict))
    if modified_status == CaseNameAssessmentStatus.SEMANTIC_MATCH:
        return _extraction_error_from_modified_match(
            citation_id=citation_id,
            extracted_case_name=extracted_case_name,
            courtlistener_case_name=courtlistener_case_name,
            modified_case_name=modified_case_name,
            modified_match_status=modified_status,
        )

    return CaseNameAssessment(
        citation_id=citation_id,
        status=CaseNameAssessmentStatus.EXTRACTION_ERROR,
        extracted_case_name=extracted_case_name,
        courtlistener_case_name=courtlistener_case_name,
        message="Modified extraction is grounded but still does not match CourtListener.",
        modified_extracted_case_name=modified_case_name,
        modified_match_status=modified_status.value,
    )


def modified_extracted_citation_is_in_context_requirement(document_context: str) -> Requirement:
    """Build the Mellea requirement that prevents ungrounded modified citations."""

    def validate(output: str) -> tuple[bool, str]:
        modified = _modified_extracted_citation_from_output(output)
        if modified is None:
            return False, "Output could not be parsed as ModifiedExtractedCitation."
        if modified.valid(document_context):
            return True, ""
        return False, "Modified citation fields must appear in document_context."

    return Requirement(
        "Every non-empty field in ModifiedExtractedCitation must be copied from document_context.",
        validation_fn=simple_validate(validate),
    )


def _extraction_error_from_modified_match(
    *,
    citation_id: str,
    extracted_case_name: str | None,
    courtlistener_case_name: str,
    modified_case_name: str | None,
    modified_match_status: CaseNameAssessmentStatus,
) -> CaseNameAssessment:
    return CaseNameAssessment(
        citation_id=citation_id,
        status=CaseNameAssessmentStatus.EXTRACTION_ERROR,
        extracted_case_name=extracted_case_name,
        courtlistener_case_name=courtlistener_case_name,
        message="Original extraction appears to need correction; grounded modified extraction matches CourtListener.",
        modified_extracted_case_name=modified_case_name,
        modified_match_status=modified_match_status.value,
    )


def _modified_extracted_citation_from_output(output: str) -> ModifiedExtractedCitation | None:
    try:
        payload = json.loads(output)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    result = payload.get("result", payload)
    try:
        return MODIFIED_EXTRACTION_ADAPTER.validate_python(result)
    except ValidationError:
        return None


def _message_for_semantic_status(status: CaseNameAssessmentStatus) -> str:
    if status == CaseNameAssessmentStatus.SEMANTIC_MATCH:
        return "Extracted case name semantically matches CourtListener."
    return "Extracted case name does not appear to match the CourtListener case."
