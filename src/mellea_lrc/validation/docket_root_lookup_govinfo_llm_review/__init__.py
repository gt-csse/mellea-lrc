"""Record GovInfo docket package reviews on a Document."""

from __future__ import annotations

from mellea_lrc.llm.profiles import load_profile
from mellea_lrc.model.citations import FullDocketCitation
from mellea_lrc.model.citations.docket_lookup import (
    DocketLookupCaseNameAssessment,
    DocketLookupFieldAssessment,
    DocketLookupReviewDecision,
)
from mellea_lrc.model.citations.govinfo_lookup import GovInfoDocketReview
from mellea_lrc.model.citations.judgments import MatchResult
from mellea_lrc.model.document import Document
from mellea_lrc.validation.docket_review.fields import append_corrections
from mellea_lrc.validation.docket_root_lookup_govinfo_llm_review.reviewer import (
    GovInfoDocketReviewContext,
    GovInfoDocketReviewer,
    GovInfoDocketReviewOutcome,
    IvrGovInfoDocketReviewer,
)
from mellea_lrc.validation.fields_aggregated_identity import SUBSTAGE as NEXT_SUBSTAGE

SUBSTAGE = "validate_roots.docket_lookup.govinfo_review"


def _no_candidate_decision(context: GovInfoDocketReviewContext) -> DocketLookupReviewDecision:
    reason = "No shortlisted GovInfo package is available for comparison."
    unavailable = DocketLookupFieldAssessment(
        propose_replacement=False, quote=None, result=MatchResult.UNAVAILABLE, reason=reason
    )
    return DocketLookupReviewDecision(
        selected_candidate_index=None,
        docket_number=unavailable,
        case_name=DocketLookupCaseNameAssessment(
            propose_replacement=False,
            quote=None,
            normalized=None,
            result=MatchResult.UNAVAILABLE,
            reason=reason,
        ),
        court=unavailable,
        date=unavailable,
        reason=(
            "The saved GovInfo search is incomplete and yielded no shortlisted package."
            if context.search_status["incomplete"]
            else "GovInfo returned no shortlisted case package."
        ),
    )


async def docket_root_lookup_govinfo_llm_review(
    document: Document, *, reviewer: GovInfoDocketReviewer | None = None
) -> Document:
    """Review each unresolved docket root's saved GovInfo shortlist once."""
    if SUBSTAGE in document.substage_runs:
        raise ValueError(f"Substage already completed: {SUBSTAGE}")
    if "validate_roots.docket_lookup.govinfo_retrieval" not in document.substage_runs:
        raise ValueError("Complete GovInfo docket lookup before its review")
    service = reviewer
    for root in tuple(item for item in document.roots if isinstance(item, FullDocketCitation)):
        if root.govinfo_docket_lookup is None:
            continue
        context = GovInfoDocketReviewContext.from_document(document, root)
        recorded = root.record(SUBSTAGE)
        if not context.candidates:
            review = GovInfoDocketReview(
                node_id=recorded.nodes[-1].id, decision=_no_candidate_decision(context)
            )
        else:
            if service is None:
                service = IvrGovInfoDocketReviewer.from_profile(load_profile(SUBSTAGE))
            result = await service(context)
            outcome = (
                result
                if isinstance(result, GovInfoDocketReviewOutcome)
                else GovInfoDocketReviewOutcome(decision=result)
            )
            failure = outcome.failure_reason
            if outcome.run is not None and not outcome.run.success:
                failure = failure or outcome.run.failure_reason or "IVR review failed"
            if outcome.decision is not None and (error := context.choice_error(outcome.decision)):
                failure = error
            if failure is None and outcome.decision is not None:
                corrections = context.grounded_corrections(outcome.decision)
                if corrections is None:
                    raise ValueError("Accepted GovInfo review has ungrounded corrections")
                recorded = append_corrections(recorded, document.text, corrections, outcome.decision)
            review = (
                GovInfoDocketReview(
                    node_id=recorded.nodes[-1].id,
                    ivr=outcome.run,
                    failure_reason=failure or "Model review produced no decision",
                )
                if failure is not None or outcome.decision is None
                else GovInfoDocketReview(
                    node_id=recorded.nodes[-1].id,
                    decision=outcome.decision,
                    ivr=outcome.run,
                )
            )
        recorded = recorded.with_govinfo_docket_review(review)
        if review.decision is not None and review.decision.selected_candidate_index is not None:
            recorded = recorded.with_route(NEXT_SUBSTAGE)
        document = document.replace_citation(recorded)
    return document.complete_substage(SUBSTAGE)
