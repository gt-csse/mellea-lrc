"""Shared execution of occurrence-local support reviews, below substage APIs."""

from mellea_lrc.model.citations.citation import Citation
from mellea_lrc.model.citations.reporter_pinpoint import OpinionReviewScope, ReporterCitationSupportReview
from mellea_lrc.model.document import Document
from mellea_lrc.validation.reporter_pinpoint_review.reviewer import (
    ReporterPinpointReviewContext,
    ReporterPinpointReviewer,
    ReporterPinpointReviewOutcome,
)


async def review_citation(
    document: Document,
    citation: Citation,
    *,
    substage: str,
    scope: OpinionReviewScope,
    reviewer: ReporterPinpointReviewer,
) -> Document:
    context = ReporterPinpointReviewContext.from_document(document, citation, scope)
    outcome = None
    try:
        response = await reviewer(context)
        outcome = (
            response
            if isinstance(response, ReporterPinpointReviewOutcome)
            else ReporterPinpointReviewOutcome(response)
        )
        if outcome.decision is not None and outcome.failure_reason is not None:
            raise ValueError("Pinpoint reviewer returned both a decision and failure")
        accepted = context.ground(outcome.decision, "validation") if outcome.decision is not None else ()
    except Exception as error:
        outcome = ReporterPinpointReviewOutcome(
            None,
            run=outcome.run if outcome is not None else None,
            failure_reason=f"{type(error).__name__}: {error}",
        )
        accepted = ()
    recorded = citation.record(substage)
    indices = []
    for passage in accepted:
        indices.append(len(recorded.reporter_opinion_evidence))
        recorded = recorded.with_reporter_opinion_evidence(
            passage.model_copy(update={"node_id": recorded.nodes[-1].id})
        )
    recorded = recorded.with_reporter_support_review(
        ReporterCitationSupportReview(
            node_id=recorded.nodes[-1].id,
            evidence_index=context.evidence_index,
            scope=scope,
            decision=outcome.decision,
            opinion_evidence_indices=tuple(indices),
            ivr=outcome.run,
            failure_reason=(outcome.failure_reason or "No support decision")
            if outcome.decision is None
            else None,
        )
    )
    return document.replace_citation(recorded)
