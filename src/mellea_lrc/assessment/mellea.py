"""Mellea-backed semantic assessment for citations."""

import json
import os
from typing import Literal, cast

from mellea import MelleaSession, generative
from mellea.core import Requirement
from mellea.stdlib.requirements import simple_validate
from mellea.stdlib.sampling import RejectionSamplingStrategy
from pydantic import TypeAdapter, ValidationError

from mellea_lrc.assessment.case_name import assess_case_name_exact_match
from mellea_lrc.assessment.types import (
    CaseNameAssessment,
    CaseNameAssessmentRun,
    CaseNameAssessmentStatus,
    ModifiedExtractedCitationProposal,
    is_in_context,
)

CaseNameSemanticVerdict = Literal["semantic_match", "extraction_error"]
MODIFIED_EXTRACTION_ADAPTER = TypeAdapter(ModifiedExtractedCitationProposal)
CASE_NAME_SEMANTIC_MAX_TOKENS = 16
MODIFIED_EXTRACTION_MAX_TOKENS = 512
MODIFIED_EXTRACTION_STRATEGY = RejectionSamplingStrategy(loop_budget=3)


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
) -> ModifiedExtractedCitationProposal:
    """Propose a corrected case-name extraction from the visible citation context.

    Return corrected plaintiff and defendant strings, or a corrected case_name string,
    only when the corrected text is visible in document_context. Copy strings exactly
    from document_context. Do not use the CourtListener case name to invent text that
    is not present in document_context.

    If no grounded correction is visible in document_context, return an empty
    ModifiedExtractedCitationProposal.
    """


@generative
def propose_modified_extracted_citation_json(
    document_context: str,
    extracted_case_name: str,
    courtlistener_case_name: str,
) -> str:
    """Propose a corrected case-name extraction as compact JSON.

    Return exactly one JSON object with these keys: plaintiff, defendant,
    case_name. Each value must be a string copied exactly from document_context
    or null. Do not add markdown, comments, or explanatory text.

    Use case_name when the corrected full case name appears as one visible string.
    Use plaintiff and defendant when the corrected parties appear separately.
    If no grounded correction is visible in document_context, return:
    {"plaintiff": null, "defendant": null, "case_name": null}
    """


def assess_case_name_with_mellea(
    session: MelleaSession,
    *,
    citation_id: str,
    extracted_case_name: str | None,
    courtlistener_case_name: str | None,
    document_context: str,
) -> CaseNameAssessmentRun:
    """Assess one case name, using Mellea only when exact equality is not enough."""
    exact_result = assess_case_name_exact_match(
        citation_id=citation_id,
        extracted_case_name=extracted_case_name,
        courtlistener_case_name=courtlistener_case_name,
    )
    if exact_result.status == CaseNameAssessmentStatus.EXACT_MATCH:
        return CaseNameAssessmentRun(assessment=exact_result)
    if not courtlistener_case_name:
        return CaseNameAssessmentRun(assessment=exact_result)

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
) -> CaseNameAssessmentRun:
    """Run semantic and modified-extraction case-name checks."""
    if extracted_case_name:
        verdict = classify_case_name_semantic_match(
            session,
            document_context=document_context,
            extracted_case_name=extracted_case_name,
            courtlistener_case_name=courtlistener_case_name,
            model_options=_structured_model_options(max_tokens=CASE_NAME_SEMANTIC_MAX_TOKENS),
        )
        status = CaseNameAssessmentStatus(cast("str", verdict))
        if status == CaseNameAssessmentStatus.SEMANTIC_MATCH:
            return CaseNameAssessmentRun(
                assessment=CaseNameAssessment(
                    citation_id=citation_id,
                    status=status,
                    extracted_case_name=extracted_case_name,
                    courtlistener_case_name=courtlistener_case_name,
                    message=_message_for_semantic_status(status),
                ),
            )

    modified = _propose_modified_extracted_citation(
        session,
        document_context=document_context,
        extracted_case_name=extracted_case_name or "",
        courtlistener_case_name=courtlistener_case_name,
    )
    if not modified.valid(document_context):
        return CaseNameAssessmentRun(
            assessment=CaseNameAssessment(
                citation_id=citation_id,
                status=CaseNameAssessmentStatus.EXTRACTION_ERROR,
                extracted_case_name=extracted_case_name,
                courtlistener_case_name=courtlistener_case_name,
                message="No grounded modified extraction was found in the document context.",
            ),
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
            modified_citation=modified,
            reassessment=modified_exact,
            modified_match_status=modified_exact.status,
        )

    modified_verdict = classify_case_name_semantic_match(
        session,
        document_context=document_context,
        extracted_case_name=cast("str", modified_case_name),
        courtlistener_case_name=courtlistener_case_name,
        model_options=_structured_model_options(max_tokens=CASE_NAME_SEMANTIC_MAX_TOKENS),
    )
    modified_status = CaseNameAssessmentStatus(cast("str", modified_verdict))
    if modified_status == CaseNameAssessmentStatus.SEMANTIC_MATCH:
        reassessment = CaseNameAssessment(
            citation_id=citation_id,
            status=modified_status,
            extracted_case_name=modified_case_name,
            courtlistener_case_name=courtlistener_case_name,
            message=_message_for_semantic_status(modified_status),
        )
        return _extraction_error_from_modified_match(
            citation_id=citation_id,
            extracted_case_name=extracted_case_name,
            courtlistener_case_name=courtlistener_case_name,
            modified_citation=modified,
            reassessment=reassessment,
            modified_match_status=modified_status,
        )

    reassessment = CaseNameAssessment(
        citation_id=citation_id,
        status=CaseNameAssessmentStatus.EXTRACTION_ERROR,
        extracted_case_name=modified_case_name,
        courtlistener_case_name=courtlistener_case_name,
        message=_message_for_semantic_status(modified_status),
    )
    return CaseNameAssessmentRun(
        assessment=CaseNameAssessment(
            citation_id=citation_id,
            status=CaseNameAssessmentStatus.EXTRACTION_ERROR,
            extracted_case_name=extracted_case_name,
            courtlistener_case_name=courtlistener_case_name,
            message="Modified extraction is grounded but still does not match CourtListener.",
        ),
        modified_citation=modified,
        reassessment=reassessment,
    )


def modified_extracted_citation_is_in_context_requirement(document_context: str) -> Requirement:
    """Build the Mellea requirement that prevents ungrounded modified citations."""

    def validate(output: str) -> tuple[bool, str]:
        modified = _modified_extracted_citation_from_output(output)
        if modified is None:
            return False, "Output could not be parsed as ModifiedExtractedCitationProposal."
        values = tuple(
            value for value in (modified.plaintiff, modified.defendant, modified.case_name) if value
        )
        if all(is_in_context(value, document_context) for value in values):
            return True, ""
        return False, "Modified citation fields must appear in document_context."

    return Requirement(
        "Every non-empty field in ModifiedExtractedCitationProposal must be copied from document_context.",
        validation_fn=simple_validate(validate),
    )


def _propose_modified_extracted_citation(
    session: MelleaSession,
    *,
    document_context: str,
    extracted_case_name: str,
    courtlistener_case_name: str,
) -> ModifiedExtractedCitationProposal:
    requirement = modified_extracted_citation_is_in_context_requirement(document_context)
    try:
        return propose_modified_extracted_citation(
            session,
            document_context=document_context,
            extracted_case_name=extracted_case_name,
            courtlistener_case_name=courtlistener_case_name,
            requirements=[requirement],
            strategy=MODIFIED_EXTRACTION_STRATEGY,
            model_options=_structured_model_options(max_tokens=MODIFIED_EXTRACTION_MAX_TOKENS),
        )
    except Exception:
        raw = propose_modified_extracted_citation_json(
            session,
            document_context=document_context,
            extracted_case_name=extracted_case_name,
            courtlistener_case_name=courtlistener_case_name,
            requirements=[requirement],
            strategy=MODIFIED_EXTRACTION_STRATEGY,
            model_options={"temperature": 0, "max_tokens": MODIFIED_EXTRACTION_MAX_TOKENS},
        )
        modified = _modified_extracted_citation_from_output(str(raw))
        if modified is not None:
            return modified
        return ModifiedExtractedCitationProposal()


def _extraction_error_from_modified_match(
    *,
    citation_id: str,
    extracted_case_name: str | None,
    courtlistener_case_name: str,
    modified_citation: ModifiedExtractedCitationProposal,
    reassessment: CaseNameAssessment,
    modified_match_status: CaseNameAssessmentStatus,
) -> CaseNameAssessmentRun:
    return CaseNameAssessmentRun(
        assessment=CaseNameAssessment(
            citation_id=citation_id,
            status=CaseNameAssessmentStatus.EXTRACTION_ERROR,
            extracted_case_name=extracted_case_name,
            courtlistener_case_name=courtlistener_case_name,
            message=(
                "Original extraction appears to need correction; "
                f"grounded modified extraction is {modified_match_status.value.replace('_', ' ')}."
            ),
        ),
        modified_citation=modified_citation,
        reassessment=reassessment,
    )


def _modified_extracted_citation_from_output(output: str) -> ModifiedExtractedCitationProposal | None:
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


def _structured_model_options(*, max_tokens: int) -> dict[str, object]:
    options: dict[str, object] = {"temperature": 0, "max_tokens": max_tokens}
    api_base = os.environ.get("MELLEA_LRC_ASSESSMENT_API_BASE", "")
    require_parameters = os.environ.get("MELLEA_LRC_ASSESSMENT_REQUIRE_PARAMETERS", "")
    if "openrouter.ai" in api_base and require_parameters.lower() in {"1", "true", "yes"}:
        options["extra_body"] = {"provider": {"require_parameters": True}}
    return options


def _message_for_semantic_status(status: CaseNameAssessmentStatus) -> str:
    if status == CaseNameAssessmentStatus.SEMANTIC_MATCH:
        return "Extracted case name semantically matches CourtListener."
    return "Extracted case name does not appear to match the CourtListener case."
