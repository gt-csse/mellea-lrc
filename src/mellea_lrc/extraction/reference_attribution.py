"""Attach created name-and-pin references without rereading their fields."""

from mellea_lrc.extraction.context.leaves import name_candidates, require_leaves
from mellea_lrc.extraction.leaf_attribution_review.reviewer import (
    IvrLeafReviewer,
    LeafReviewContext,
    LeafReviewer,
    LeafReviewOutcome,
    apply_review,
)
from mellea_lrc.extraction.reference_citations import SUBSTAGE as CREATION_SUBSTAGE
from mellea_lrc.llm.profiles import load_profile
from mellea_lrc.model.citations import AttributionResult, ReferenceCitation
from mellea_lrc.model.document import Document

SUBSTAGE = "grow_leaves.reference_citations.attribution"


async def attribute_reference_citations(
    document: Document, *, review: bool = True, reviewer: LeafReviewer | None = None
) -> Document:
    """Attach a unique name match, with optional review of unresolved choices.

    Discovery already requires an explicit pinpoint. Unlike a preceding-only
    short reporter, a named reference can introduce a following full citation;
    candidates therefore come from all formed roots in the filing. This substage
    changes attachments and records its reasoning, not the source readings.
    """
    require_leaves(document, SUBSTAGE)
    if CREATION_SUBSTAGE not in document.substage_runs:
        raise ValueError("Create reference citations before attributing them")
    for citation in document.short_citations:
        if not isinstance(citation, ReferenceCitation):
            continue
        candidates = name_candidates(document, citation.case_name[-1].quote, before=None)
        attach = len(candidates) == 1
        updated = citation.record(SUBSTAGE).with_attribution(
            candidates,
            AttributionResult.ATTACHED if attach else AttributionResult.UNRESOLVED,
            "Unique source-name agreement" if attach else "Semantic review required",
        )
        if attach:
            updated = updated.with_root(candidates[0])
        else:
            updated = updated.withdraw()
            if review:
                context = LeafReviewContext.from_document(document, updated)
                if reviewer is None:
                    reviewer = IvrLeafReviewer.from_profile(load_profile(SUBSTAGE))
                answer = await reviewer(context)
                outcome = answer if isinstance(answer, LeafReviewOutcome) else LeafReviewOutcome(answer)
                updated = apply_review(updated.record(SUBSTAGE), context, outcome)
        document = document.replace_citation(updated)
    return document.complete_substage(SUBSTAGE)
