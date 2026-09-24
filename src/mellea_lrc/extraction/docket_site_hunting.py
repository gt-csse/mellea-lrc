"""Iteratively admit reviewed docket locators before colocation and field reading."""

from __future__ import annotations

from mellea_lrc.extraction._site_hunting.candidates import suspected_dockets
from mellea_lrc.extraction._site_hunting.review import (
    DocketReviewOutcome,
    DocketSiteReviewer,
    IvrDocketReviewer,
    grounded_docket_decision,
)
from mellea_lrc.model.citations import FullDocketCitation
from mellea_lrc.model.document import Document
from mellea_lrc.model.site_review import SiteReview

STAGE = "docket_locator_site_hunting"


async def hunt_docket_locators(
    document: Document,
    *,
    reviewer: DocketSiteReviewer | None = None,
) -> Document:
    """Review one site at a time; accepted sites affect the next proposal mask."""
    if STAGE in document.stage_runs:
        raise ValueError(f"Stage already completed: {STAGE}")
    if "full_reporter_locators" not in document.stage_runs or "docket_locators" not in document.stage_runs:
        raise ValueError("Run both rule locator stages before docket site hunting")
    if "colocations" in document.stage_runs:
        raise ValueError("Docket site hunting must precede colocation")
    inspected: set[tuple[int, int]] = set()
    current = document
    while True:
        candidate = next(
            (
                site
                for site in suspected_dockets(current)
                if (site.locator_span.start, site.locator_span.end) not in inspected
            ),
            None,
        )
        if candidate is None:
            return current.complete(STAGE)
        inspected.add((candidate.locator_span.start, candidate.locator_span.end))
        if reviewer is None:
            reviewer = IvrDocketReviewer.from_env()
        review = await reviewer(candidate)
        outcome = review if isinstance(review, DocketReviewOutcome) else DocketReviewOutcome(review)
        decision = outcome.decision
        citation_id: str | None = None
        if decision is None:
            result = "failed"
            reason = outcome.failure_reason or "Model review did not produce a valid decision"
        elif not decision.is_docket_citation:
            result = "declined"
            reason = decision.reason
        elif not grounded_docket_decision(candidate, decision):
            result = "failed"
            reason = "Proposed locator or docket number did not ground to the candidate source span"
        else:
            result = "accepted"
            reason = decision.reason
            citation_id = f"docket:{candidate.locator_span.start}:{candidate.locator_span.end}"
            current = current.add_citation(
                FullDocketCitation.from_locator(
                    citation_id=citation_id,
                    stage=STAGE,
                    source=current.text,
                    span=candidate.locator_span,
                    number_span=candidate.number_span,
                )
            )
        current = current.add_site_review(
            SiteReview(
                stage=STAGE,
                candidate_span=candidate.locator_span,
                candidate_text=candidate.locator_text,
                outcome=result,
                reason=reason,
                proposed_locator=decision.locator if decision is not None else None,
                proposed_identifier=decision.docket_number if decision is not None else None,
                citation_id=citation_id,
                ivr=outcome.run,
            )
        )
