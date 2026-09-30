"""Audit case Id. antecedents, including noncase sources absent from eyecite."""

from mellea_lrc.extraction.leaf_attribution_llm.reviewer import (
    IvrLeafReviewer,
    LeafReviewContext,
    LeafReviewer,
    LeafReviewOutcome,
    apply_review,
)
from mellea_lrc.extraction.leaf_reading import require_leaves
from mellea_lrc.model.citations import IdCitation
from mellea_lrc.model.document import Document

STAGE = "37_id_attribution_llm"


async def review_id_attributions(document: Document, *, reviewer: LeafReviewer | None = None) -> Document:
    require_leaves(document, STAGE)
    if "36_id_attribution" not in document.stage_runs:
        raise ValueError("Run Id. rule attribution before semantic review")
    for citation in document.short_citations:
        if not isinstance(citation, IdCitation):
            continue
        # Review in source order. Every admitted/withdrawn Id. updates the
        # source tree before the following Id. is considered. Rule attribution
        # is retained; it is not proof that the last source was a case rather
        # than a declaration, procedural rule, or another noncase document.
        context = LeafReviewContext.from_document(document, citation)
        if reviewer is None:
            reviewer = IvrLeafReviewer.from_env()
        answer = await reviewer(context)
        outcome = answer if isinstance(answer, LeafReviewOutcome) else LeafReviewOutcome(answer)
        document = document.replace_citation(apply_review(citation.record(STAGE), context, outcome))
    return document.complete(STAGE)
