"""Choose opinion-page representatives only for unresolved ambiguous citations."""

from __future__ import annotations

from mellea_lrc.llm.profiles import load_profile
from mellea_lrc.model.citations.reporter_page_resolution import (
    ReporterCitationOpinionReview,
    ReporterPageResolutionOutcome,
)
from mellea_lrc.model.citations.tags import CitationTagKind
from mellea_lrc.model.document import Document
from mellea_lrc.validation.reporter_citation_opinion_review.reviewer import (
    IvrReporterCitationOpinionReviewer,
    ReporterCitationOpinionContext,
    ReporterCitationOpinionOutcome,
    ReporterCitationOpinionReviewer,
)

SUBSTAGE = "validate_pincite.citation_preparation.opinion_review"
SOURCE_SUBSTAGE = "validate_pincite.citation_preparation.page_resolution"


async def review_reporter_citation_opinions(
    document: Document, *, reviewer: ReporterCitationOpinionReviewer | None = None
) -> Document:
    """Select writing representatives using saved pages and the citing occurrence.

    Only ambiguous resolutions need a review. A null choice remains unresolved;
    no source field, identity, attachment, or proposition-support verdict changes.
    """
    if SUBSTAGE in document.substage_runs:
        raise ValueError(f"Substage already completed: {SUBSTAGE}")
    if SOURCE_SUBSTAGE not in document.substage_runs:
        raise ValueError("Resolve reporter citation pages before opinion review")
    service = reviewer
    for citation in document.citations:
        if (
            citation.has_tag(CitationTagKind.TABLE_OF_AUTHORITIES)
            or not citation.reporter_page_resolutions
            or citation.reporter_page_resolutions[-1].outcome is not ReporterPageResolutionOutcome.AMBIGUOUS
        ):
            continue
        context = ReporterCitationOpinionContext.from_document(document, citation)
        if service is None:
            service = IvrReporterCitationOpinionReviewer.from_profile(load_profile(SUBSTAGE))
        try:
            result = await service(context)
        except Exception as error:
            result = ReporterCitationOpinionOutcome(None, failure_reason=f"{type(error).__name__}: {error}")
        outcome = (
            result
            if isinstance(result, ReporterCitationOpinionOutcome)
            else ReporterCitationOpinionOutcome(result)
        )
        if outcome.decision is not None:
            if outcome.failure_reason is not None:
                raise ValueError("Opinion reviewer returned both a decision and a failure")
            if (error := context.decision_error(outcome.decision)) is not None:
                raise ValueError(error)
        recorded = citation.record(SUBSTAGE)
        recorded = recorded.with_reporter_opinion_review(
            ReporterCitationOpinionReview(
                node_id=recorded.nodes[-1].id,
                resolution_index=len(citation.reporter_page_resolutions) - 1,
                decision=outcome.decision,
                ivr=outcome.run,
                failure_reason=outcome.failure_reason
                if outcome.decision is None and outcome.failure_reason is not None
                else "Opinion review produced no decision"
                if outcome.decision is None
                else None,
            )
        )
        document = document.replace_citation(recorded)
    return document.complete_substage(SUBSTAGE)
