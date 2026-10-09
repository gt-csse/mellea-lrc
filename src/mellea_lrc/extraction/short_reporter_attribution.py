"""Attach created short reporters, independently of source-field reading."""

from mellea_lrc.extraction.context.leaves import name_candidates, reporter_candidates, require_leaves
from mellea_lrc.extraction.leaf_attribution_review.reviewer import (
    IvrLeafReviewer,
    LeafReviewContext,
    LeafReviewer,
    LeafReviewOutcome,
    apply_review,
)
from mellea_lrc.extraction.short_reporter_case_names import SUBSTAGE as NAME_SUBSTAGE
from mellea_lrc.llm.profiles import load_profile
from mellea_lrc.model.citations import AttributionResult
from mellea_lrc.model.document import Document

SUBSTAGE = "grow_leaves.short_reporter_citations.attribution"


async def attribute_short_reporter_citations(
    document: Document, *, review: bool = True, reviewer: LeafReviewer | None = None
) -> Document:
    """Use reporter/name rules, then optionally review unresolved attachment.

    This substage consumes the completed colocation and name checkpoints. It never rereads
    or changes locator, case-name or pinpoint fields. Further attribution
    augmentation can stay here without changing the creation substage.
    """
    require_leaves(document, SUBSTAGE)
    if NAME_SUBSTAGE not in document.substage_runs:
        raise ValueError("Read short reporter case names before attributing them")
    for citation in document.short_reporters:
        candidates = ()
        if citation.short_locator[-1].normalizable:
            locator = citation.short_locator[-1].get_normalized()
            candidates = reporter_candidates(
                document, locator.volume, locator.edition, citation.site_span.start
            )
        named = (
            name_candidates(document, citation.case_name[-1].quote, citation.site_span.start)
            if citation.case_name
            else ()
        )
        passing = tuple(root for root in candidates if not citation.case_name or root in named)
        attach = len(passing) == 1
        updated = citation.record(SUBSTAGE).with_attribution(
            candidates,
            AttributionResult.ATTACHED if attach else AttributionResult.UNRESOLVED,
            "Unique preceding locator/name agreement" if attach else "Semantic review required",
        )
        if attach:
            updated = updated.with_root(passing[0])
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
