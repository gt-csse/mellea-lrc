"""Ask once per fuzzy docket-root neighborhood which cited cases are identical."""

from __future__ import annotations

from mellea_lrc.extraction.docket_root_llm_reassignment.context import (
    DocketRootReviewContext,
    candidate_components,
)
from mellea_lrc.extraction.docket_root_llm_reassignment.reviewer import (
    DocketRootReviewer,
    DocketRootReviewOutcome,
    IvrDocketRootReviewer,
)
from mellea_lrc.llm.profiles import OPENROUTER_LUNA
from mellea_lrc.model.citations import latest
from mellea_lrc.model.citations.docket_root_llm_reassignment import DocketRootReview
from mellea_lrc.model.document import Document

STAGE = "11_docket_root_llm_reassignment"
MODEL_PROFILE = OPENROUTER_LUNA


async def docket_root_llm_reassignment(
    document: Document, *, reviewer: DocketRootReviewer | None = None
) -> Document:
    """Review fuzzy docket-root neighborhoods and append confirmed root links.

    The threshold proposes review batches only. The model partitions each batch;
    program code chooses the earliest root of each same-case group and reassigns
    every occurrence attached to any losing root. No root assignment is erased.
    """
    if STAGE in document.stage_runs:
        raise ValueError(f"Stage already completed: {STAGE}")
    if "10_roots" not in document.stage_runs:
        raise ValueError("Form roots before reviewing docket-root equivalence")

    service = reviewer
    for roots in candidate_components(document):
        context = DocketRootReviewContext.from_document(document, roots)
        if service is None:
            service = IvrDocketRootReviewer.from_profile(MODEL_PROFILE)
        returned = await service(context)
        outcome = (
            returned
            if isinstance(returned, DocketRootReviewOutcome)
            else DocketRootReviewOutcome(decision=returned)
        )
        decision = outcome.decision
        error = context.partition_error(decision) if decision is not None else None
        if error is not None and outcome.run is None:
            raise ValueError(error)
        if error is not None:
            decision = None
        anchor = roots[0].record(STAGE)
        anchor = anchor.with_docket_root_review(
            DocketRootReview(
                node_id=anchor.nodes[-1].id,
                candidate_ids=tuple(root.id for root in roots),
                decision=decision,
                ivr=outcome.run,
                failure_reason=error
                or outcome.failure_reason
                or ("Model review produced no partition" if decision is None else None),
            )
        )
        document = document.replace_citation(anchor)
        if decision is None:
            continue

        for group in decision.groups:
            if len(group) < 2:
                continue
            winner = roots[min(group)].id
            losers = {roots[index].id for index in group if roots[index].id != winner}
            # A previously deduplicated occurrence points to its former root.
            # Move the entire attachment set so no leaf is stranded on a root
            # that this review has turned into another root's leaf.
            for citation in tuple(document.full_locators):
                if latest(citation.root_id) in losers:
                    document = document.replace_citation(citation.record(STAGE).with_root(winner))
    return document.complete(STAGE)
