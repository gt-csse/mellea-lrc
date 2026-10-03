"""Attach created short reporters, independently of source-field reading."""

from mellea_lrc.extraction.context.leaves import name_candidates, reporter_candidates, require_leaves
from mellea_lrc.extraction.leaf_attribution_review.reviewer import (
    IvrLeafReviewer,
    LeafReviewContext,
    LeafReviewer,
    LeafReviewOutcome,
    apply_review,
)
from mellea_lrc.extraction.short_reporter_case_names import STAGE as NAME_STAGE
from mellea_lrc.llm.profiles import OPENROUTER_LUNA
from mellea_lrc.model.citations import AttributionResult
from mellea_lrc.model.document import Document

STAGE = "29_short_reporter_attribution"
MODEL_PROFILE = OPENROUTER_LUNA


async def attribute_short_reporter_citations(
    document: Document, *, review: bool = True, reviewer: LeafReviewer | None = None
) -> Document:
    """Use reporter/name rules, then optionally review unresolved attachment.

    This stage consumes the completed colocation and name checkpoints. It never rereads
    or changes locator, case-name or pinpoint fields. Further attribution
    augmentation can stay here without changing the creation stage.
    """
    require_leaves(document, STAGE)
    if NAME_STAGE not in document.stage_runs:
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
        updated = citation.record(STAGE).with_attribution(
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
                    reviewer = IvrLeafReviewer.from_profile(MODEL_PROFILE)
                answer = await reviewer(context)
                outcome = answer if isinstance(answer, LeafReviewOutcome) else LeafReviewOutcome(answer)
                updated = apply_review(updated.record(STAGE), context, outcome)
        document = document.replace_citation(updated)
    return document.complete(STAGE)
