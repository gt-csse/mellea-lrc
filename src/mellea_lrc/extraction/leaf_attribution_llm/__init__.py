"""Review only short forms routed here; preserve every choice and failure."""

from mellea_lrc.extraction.leaf_attribution_llm.reviewer import (
    IvrLeafReviewer,
    LeafReviewContext,
    LeafReviewer,
    LeafReviewOutcome,
    apply_review,
)
from mellea_lrc.extraction.leaf_reading import require_leaves
from mellea_lrc.model.document import Document

STAGE = "35_leaf_attribution_llm"


async def review_leaf_attributions(document: Document, *, reviewer: LeafReviewer | None = None) -> Document:
    require_leaves(document, STAGE)
    for citation in document.short_citations:
        if citation.next_stage != STAGE:
            continue
        context = LeafReviewContext.from_document(document, citation)
        if reviewer is None:
            reviewer = IvrLeafReviewer.from_env()
        answer = await reviewer(context)
        outcome = answer if isinstance(answer, LeafReviewOutcome) else LeafReviewOutcome(answer)
        updated = apply_review(citation.record(STAGE), context, outcome)
        document = document.replace_citation(updated)
    return document.complete(STAGE)
