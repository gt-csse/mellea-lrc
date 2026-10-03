"""Review one saved CourtListener shortlist for each docket root."""

from __future__ import annotations

from mellea_lrc.llm.profiles import OPENROUTER_LUNA
from mellea_lrc.model.citations import FullDocketCitation
from mellea_lrc.model.citations.docket_lookup import (
    DocketLookupCaseNameAssessment,
    DocketLookupFieldAssessment,
    DocketLookupReview,
    DocketLookupReviewDecision,
)
from mellea_lrc.model.citations.judgments import MatchResult
from mellea_lrc.model.document import Document
from mellea_lrc.validation.docket_review.fields import append_corrections

from .reviewer import (
    DocketLookupReviewContext,
    DocketLookupReviewer,
    DocketLookupReviewOutcome,
    IvrDocketLookupReviewer,
)

STAGE = "17_docket_root_lookup_courtlistener_llm_review"
MODEL_PROFILE = OPENROUTER_LUNA
NEXT_STAGE = "fields_aggregated_identity"


def _no_candidate_decision(context: DocketLookupReviewContext) -> DocketLookupReviewDecision:
    reason = "No shortlisted CourtListener record is available for comparison."
    unavailable = DocketLookupFieldAssessment(
        propose_replacement=False, quote=None, result=MatchResult.UNAVAILABLE, reason=reason
    )
    selection_reason = (
        "The search stopped before all results were available, and no candidate was shortlisted "
        "from the saved pages."
        if context.search_status["incomplete"]
        else "The docket search produced no shortlisted candidate to select."
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
        reason=selection_reason,
    )


async def docket_root_lookup_courtlistener_llm_review(
    document: Document, *, reviewer: DocketLookupReviewer | None = None
) -> Document:
    """Review each docket root once, retaining the choice or complete failure."""
    if STAGE in document.stage_runs:
        raise ValueError(f"Stage already completed: {STAGE}")
    if "16_docket_root_lookup_courtlistener_retrieval" not in document.stage_runs:
        raise ValueError("Complete docket root lookup before its model review")
    service = reviewer
    for root in tuple(item for item in document.roots if isinstance(item, FullDocketCitation)):
        if root.docket_lookup is None:
            raise ValueError("Docket root review requires a saved lookup on every docket root")
        context = DocketLookupReviewContext.from_document(document, root)
        recorded = root.record(STAGE)
        if not context.candidates:
            review = DocketLookupReview(
                node_id=recorded.nodes[-1].id, decision=_no_candidate_decision(context)
            )
        else:
            if service is None:
                service = IvrDocketLookupReviewer.from_profile(MODEL_PROFILE)
            result = await service(context)
            outcome = (
                result
                if isinstance(result, DocketLookupReviewOutcome)
                else DocketLookupReviewOutcome(decision=result)
            )
            failure = outcome.failure_reason
            if outcome.run is not None and not outcome.run.success:
                failure = failure or outcome.run.failure_reason or "IVR review failed"
            if outcome.decision is not None and (error := context.choice_error(outcome.decision)):
                failure = error
            if failure is None and outcome.decision is not None:
                corrections = context.grounded_corrections(outcome.decision)
                if corrections is None:
                    raise ValueError("Accepted docket review has ungrounded corrections")
                recorded = append_corrections(recorded, document.text, corrections, outcome.decision)
            review = (
                DocketLookupReview(
                    node_id=recorded.nodes[-1].id,
                    ivr=outcome.run,
                    failure_reason=failure or "Model review produced no decision",
                )
                if failure is not None or outcome.decision is None
                else DocketLookupReview(
                    node_id=recorded.nodes[-1].id,
                    decision=outcome.decision,
                    ivr=outcome.run,
                )
            )
        recorded = recorded.with_docket_lookup_review(review)
        if review.decision is not None and review.decision.selected_candidate_index is not None:
            # The selected record's field judgments are saved here; overall
            # identity is a separate decision that can run after other reviews.
            recorded = recorded.with_route(NEXT_STAGE)
        document = document.replace_citation(recorded)
    return document.complete(STAGE)
