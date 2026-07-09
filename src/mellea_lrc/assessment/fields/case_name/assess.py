"""Case-name assessment orchestration."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mellea_lrc.assessment.fields.case_name.classify import (
    CASE_NAME_VERDICT_MAX_TOKENS,
    semantic_match_case_name,
)
from mellea_lrc.assessment.fields.case_name.compare import (
    assess_case_name_exact_match,
    build_case_name_assessment,
)
from mellea_lrc.assessment.fields.case_name.reextract import (
    ReextractionResult,
    ReextractionStatus,
    reextract_case_name,
)
from mellea_lrc.assessment.model_options import structured_model_options
from mellea_lrc.assessment.types.case_name import (
    CaseNameAssessment,
    CaseNameAssessmentRun,
    CaseNameAssessmentStatus,
    CaseNameReassessed,
    CaseNameReassessmentFailed,
    CaseNameReassessmentNotRequired,
    CaseNameReextractionFailed,
    ReextractedCaseName,
)

if TYPE_CHECKING:
    from mellea import MelleaSession

    from mellea_lrc.assessment.context import DocumentTextWindow


async def assess_case_name_with_mellea(
    session: MelleaSession,
    *,
    extracted_case_name: str | None,
    courtlistener_case_name: str | None,
    document_context: DocumentTextWindow,
    citation_locator: str | None = None,
) -> CaseNameAssessmentRun:
    """Assess one case name: exact, semantic, re-extract, then reassess."""
    exact_result = assess_case_name_exact_match(
        extracted_case_name=extracted_case_name,
        courtlistener_case_name=courtlistener_case_name,
    )
    if exact_result is not None:
        return CaseNameAssessmentRun(
            initial=exact_result,
            followup=CaseNameReassessmentNotRequired(),
        )
    assert courtlistener_case_name is not None, "exact comparison handles missing retrieved names"

    try:
        return await _assess_after_exact(
            session,
            extracted_case_name=extracted_case_name,
            courtlistener_case_name=courtlistener_case_name,
            document_context=document_context,
            citation_locator=citation_locator,
        )
    except RuntimeError:
        raise
    except Exception as exc:
        msg = f"Mellea case-name assessment failed: {exc}"
        raise RuntimeError(msg) from exc


async def _assess_after_exact(
    session: MelleaSession,
    *,
    extracted_case_name: str | None,
    courtlistener_case_name: str,
    document_context: DocumentTextWindow,
    citation_locator: str | None,
) -> CaseNameAssessmentRun:
    if (
        extracted_case_name
        and await semantic_match_case_name(
            session,
            local_context=document_context.text,
            extracted_case_name=extracted_case_name,
            retrieved_case_name=courtlistener_case_name,
            model_options=structured_model_options(max_tokens=CASE_NAME_VERDICT_MAX_TOKENS),
        )
        == "semantic_match"
    ):
        return CaseNameAssessmentRun(
            initial=build_case_name_assessment(
                CaseNameAssessmentStatus.SEMANTIC_MATCH,
                extracted_case_name,
                courtlistener_case_name,
            ),
            followup=CaseNameReassessmentNotRequired(),
        )

    initial = build_case_name_assessment(
        CaseNameAssessmentStatus.NOT_SEMANTIC_MATCH,
        extracted_case_name,
        courtlistener_case_name,
        message="Case name failed semantic match; re-extraction attempted.",
    )
    try:
        reextraction = await reextract_case_name(
            session,
            document_context=document_context.text,
            extracted_case_name=extracted_case_name,
            courtlistener_case_name=courtlistener_case_name,
            citation_locator=citation_locator,
        )
    except Exception as exc:
        return CaseNameAssessmentRun(
            initial=initial,
            followup=CaseNameReextractionFailed(error=f"{type(exc).__name__}: {exc}"),
        )
    return await _run_reassessment(
        session,
        initial=initial,
        reextraction=reextraction,
        courtlistener_case_name=courtlistener_case_name,
        document_context=document_context,
    )


async def _run_reassessment(
    session: MelleaSession,
    *,
    initial: CaseNameAssessment,
    reextraction: ReextractionResult,
    courtlistener_case_name: str,
    document_context: DocumentTextWindow,
) -> CaseNameAssessmentRun:
    if reextraction.status != ReextractionStatus.ACCEPTED or reextraction.proposal is None:
        error = reextraction.error_message or reextraction.status.value
        return CaseNameAssessmentRun(
            initial=initial,
            followup=CaseNameReextractionFailed(error=error),
        )

    grounded = document_context.locate(reextraction.proposal.case_name)
    if grounded is None:
        return CaseNameAssessmentRun(
            initial=initial,
            followup=CaseNameReextractionFailed(
                error="Accepted case-name proposal could not be grounded to document offsets"
            ),
        )
    reextracted = ReextractedCaseName(
        case_name=grounded.text,
        case_name_span=grounded.span,
    )
    try:
        result = await _assess_reextracted_case_name(
            session,
            corrected_case_name=reextracted.case_name,
            courtlistener_case_name=courtlistener_case_name,
            document_context=document_context.text,
        )
    except Exception as exc:
        return CaseNameAssessmentRun(
            initial=initial,
            followup=CaseNameReassessmentFailed(
                reextracted_case_name=reextracted,
                error=f"{type(exc).__name__}: {exc}",
            ),
        )
    return CaseNameAssessmentRun(
        initial=initial,
        followup=CaseNameReassessed(
            reextracted_case_name=reextracted,
            result=result,
        ),
    )


async def _assess_reextracted_case_name(
    session: MelleaSession,
    *,
    corrected_case_name: str,
    courtlistener_case_name: str,
    document_context: str,
) -> CaseNameAssessment:
    exact = assess_case_name_exact_match(
        extracted_case_name=corrected_case_name,
        courtlistener_case_name=courtlistener_case_name,
    )
    if exact is not None:
        return exact
    if (
        await semantic_match_case_name(
            session,
            local_context=document_context,
            extracted_case_name=corrected_case_name,
            retrieved_case_name=courtlistener_case_name,
            model_options=structured_model_options(max_tokens=CASE_NAME_VERDICT_MAX_TOKENS),
        )
        == "semantic_match"
    ):
        return build_case_name_assessment(
            CaseNameAssessmentStatus.SEMANTIC_MATCH,
            corrected_case_name,
            courtlistener_case_name,
        )
    return build_case_name_assessment(
        CaseNameAssessmentStatus.NOT_SEMANTIC_MATCH,
        corrected_case_name,
        courtlistener_case_name,
    )
