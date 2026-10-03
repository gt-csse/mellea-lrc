"""Review supra citations routed here, preserving every decision and failure."""

from mellea_lrc.extraction.context.leaves import require_leaves
from mellea_lrc.extraction.leaf_attribution_review.reviewer import (
    IvrLeafReviewer,
    LeafReviewContext,
    LeafReviewer,
    LeafReviewOutcome,
    apply_review,
)
from mellea_lrc.llm.profiles import OPENROUTER_LUNA
from mellea_lrc.model.citations import SupraCitation
from mellea_lrc.model.document import Document

STAGE = "38_supra_attribution_llm"
MODEL_PROFILE = OPENROUTER_LUNA


async def review_supra_attributions(document: Document, *, reviewer: LeafReviewer | None = None) -> Document:
    require_leaves(document, STAGE)
    for citation in document.short_citations:
        if not isinstance(citation, SupraCitation) or citation.next_stage != STAGE:
            continue
        context = LeafReviewContext.from_document(document, citation)
        if reviewer is None:
            reviewer = IvrLeafReviewer.from_profile(MODEL_PROFILE)
        answer = await reviewer(context)
        outcome = answer if isinstance(answer, LeafReviewOutcome) else LeafReviewOutcome(answer)
        updated = apply_review(citation.record(STAGE), context, outcome)
        document = document.replace_citation(updated)
    return document.complete(STAGE)
