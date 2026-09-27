"""Record GovInfo docket package reviews on a Document."""

from __future__ import annotations

from mellea_lrc.model.citations import FullDocketCitation
from mellea_lrc.model.citations.govinfo_lookup import GovInfoDocketReview
from mellea_lrc.model.document import Document
from mellea_lrc.validation.govinfo_docket_lookup_review.reviewer import (
    GovInfoDocketReviewContext,
    GovInfoDocketReviewer,
    GovInfoDocketReviewOutcome,
    IvrGovInfoDocketReviewer,
    _append_corrections,
    _no_candidate_decision,
)

STAGE = "19_govinfo_docket_lookup_review"


async def govinfo_docket_lookup_review(
    document: Document, *, reviewer: GovInfoDocketReviewer | None = None
) -> Document:
    """Review each unresolved docket root's saved GovInfo shortlist once."""
    if STAGE in document.stage_runs:
        raise ValueError(f"Stage already completed: {STAGE}")
    if "18_govinfo_docket_lookup" not in document.stage_runs:
        raise ValueError("Complete GovInfo docket lookup before its review")
    service = reviewer
    for root in tuple(item for item in document.roots if isinstance(item, FullDocketCitation)):
        if root.govinfo_docket_lookup is None:
            continue
        context = GovInfoDocketReviewContext.from_document(document, root)
        recorded = root.record(STAGE)
        if not context.candidates:
            review = GovInfoDocketReview(
                node_id=recorded.nodes[-1].id, decision=_no_candidate_decision(context)
            )
        else:
            if service is None:
                service = IvrGovInfoDocketReviewer.from_env()
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
                recorded = _append_corrections(recorded, document.text, corrections, outcome.decision)
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
        document = document.replace_citation(recorded.with_govinfo_docket_review(review))
    return document.complete(STAGE)
