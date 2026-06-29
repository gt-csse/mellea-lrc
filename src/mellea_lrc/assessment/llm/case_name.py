"""Mellea orchestration for one case-name assessment."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from mellea_lrc.assessment.deterministic.case_name import (
    assess_case_name_exact_match,
    build_case_name_assessment,
)
from mellea_lrc.assessment.llm.classify import (
    classify_non_semantic_with_mellea,
    is_semantic_match_with_mellea,
)
from mellea_lrc.assessment.llm.reextract import ReextractionResult, ReextractionStatus, reextract_case_name
from mellea_lrc.assessment.types import (
    CaseNameAssessment,
    CaseNameAssessmentRun,
    CaseNameReassessed,
    CaseNameReassessmentFailed,
    CaseNameReassessmentNotRequired,
    CaseNameReextractionFailed,
    CaseNameAssessmentStatus,
)

if TYPE_CHECKING:
    from mellea import MelleaSession


async def assess_case_name_with_mellea(
    session: MelleaSession,
    *,
    citation_id: str,
    extracted_case_name: str | None,
    courtlistener_case_name: str | None,
    document_context: str,
) -> CaseNameAssessmentRun:
    """Assess one case name: exact → semantic → re-extract → reassess."""
    exact_result = assess_case_name_exact_match(
        citation_id=citation_id,
        extracted_case_name=extracted_case_name,
        courtlistener_case_name=courtlistener_case_name,
    )
    if exact_result.status == CaseNameAssessmentStatus.EXACT_MATCH:
        return CaseNameAssessmentRun(
            assessment=exact_result,
            reassessment=CaseNameReassessmentNotRequired(),
        )
    if not courtlistener_case_name:
        return CaseNameAssessmentRun(
            assessment=exact_result,
            reassessment=CaseNameReassessmentNotRequired(),
        )

    try:
        return await _assess_case_name_with_mellea_after_exact(
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


async def _assess_case_name_with_mellea_after_exact(
    session: MelleaSession,
    *,
    citation_id: str,
    extracted_case_name: str | None,
    courtlistener_case_name: str,
    document_context: str,
) -> CaseNameAssessmentRun:
    """Run semantic match, then always re-extract when that fails."""
    if extracted_case_name and await is_semantic_match_with_mellea(
        session,
        local_context=document_context,
        extracted_case_name=extracted_case_name,
        retrieved_case_name=courtlistener_case_name,
    ):
        return CaseNameAssessmentRun(
            assessment=build_case_name_assessment(
                citation_id,
                CaseNameAssessmentStatus.SEMANTIC_MATCH,
                extracted_case_name,
                courtlistener_case_name,
            ),
            reassessment=CaseNameReassessmentNotRequired(),
        )

    first_pass = build_case_name_assessment(
        citation_id,
        CaseNameAssessmentStatus.NEEDS_ASSESSMENT,
        extracted_case_name,
        courtlistener_case_name,
        message="Case name failed semantic match; re-extraction attempted.",
    )
    try:
        reextraction = await reextract_case_name(
            session,
            document_context=document_context,
            extracted_case_name=extracted_case_name,
            courtlistener_case_name=courtlistener_case_name,
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        return CaseNameAssessmentRun(
            assessment=build_case_name_assessment(
                citation_id,
                CaseNameAssessmentStatus.REEXTRACTION_FAIL,
                extracted_case_name,
                courtlistener_case_name,
                message=f"Case-name re-extraction failed: {error}",
            ),
            reassessment=CaseNameReextractionFailed(error=error),
        )
    return await _run_reassessment_after_reextraction(
        session,
        citation_id=citation_id,
        first_pass=first_pass,
        reextraction=reextraction,
        courtlistener_case_name=courtlistener_case_name,
        document_context=document_context,
    )


async def _run_reassessment_after_reextraction(
    session: MelleaSession,
    *,
    citation_id: str,
    first_pass: CaseNameAssessment,
    reextraction: ReextractionResult,
    courtlistener_case_name: str,
    document_context: str,
) -> CaseNameAssessmentRun:
    if reextraction.status != ReextractionStatus.ACCEPTED or reextraction.proposal is None:
        error = reextraction.error_message or reextraction.status.value
        return CaseNameAssessmentRun(
            assessment=build_case_name_assessment(
                citation_id,
                CaseNameAssessmentStatus.REEXTRACTION_FAIL,
                first_pass.extracted_case_name,
                courtlistener_case_name,
                message=(
                    "Case-name re-extraction failed: "
                    f"{error}"
                ),
                chat_history=reextraction.chat_history,
            ),
            reassessment=CaseNameReextractionFailed(error=error),
        )

    try:
        reassessment = await _assess_reextracted_case_name(
            session,
            citation_id=citation_id,
            corrected_case_name=cast("str", reextraction.proposal.case_name),
            courtlistener_case_name=courtlistener_case_name,
            document_context=document_context,
        )
    except Exception as exc:
        return CaseNameAssessmentRun(
            assessment=first_pass,
            reassessment=CaseNameReassessmentFailed(
                modified_citation=reextraction.proposal,
                error=f"{type(exc).__name__}: {exc}",
            ),
        )
    return CaseNameAssessmentRun(
        assessment=first_pass,
        reassessment=CaseNameReassessed(
            modified_citation=reextraction.proposal,
            reassessment=reassessment,
        ),
    )


async def _assess_reextracted_case_name(
    session: MelleaSession,
    *,
    citation_id: str,
    corrected_case_name: str,
    courtlistener_case_name: str,
    document_context: str,
) -> CaseNameAssessment:
    """Assess a re-extracted case name: exact → semantic → different_case | irregular_form."""
    exact = assess_case_name_exact_match(
        citation_id=citation_id,
        extracted_case_name=corrected_case_name,
        courtlistener_case_name=courtlistener_case_name,
    )
    if exact.status == CaseNameAssessmentStatus.EXACT_MATCH:
        return exact
    if await is_semantic_match_with_mellea(
        session,
        local_context=document_context,
        extracted_case_name=corrected_case_name,
        retrieved_case_name=courtlistener_case_name,
    ):
        return build_case_name_assessment(
            citation_id,
            CaseNameAssessmentStatus.SEMANTIC_MATCH,
            corrected_case_name,
            courtlistener_case_name,
        )
    status = await classify_non_semantic_with_mellea(
        session,
        local_context=document_context,
        extracted_case_name=corrected_case_name,
        retrieved_case_name=courtlistener_case_name,
    )
    return build_case_name_assessment(
        citation_id,
        status,
        corrected_case_name,
        courtlistener_case_name,
    )
