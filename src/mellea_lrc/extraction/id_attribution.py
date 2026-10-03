"""Attribute each Id. before proceeding to the next link in its chain."""

from mellea_lrc.extraction.context.leaves import require_leaves
from mellea_lrc.extraction.id_citations import STAGE as CREATION_STAGE
from mellea_lrc.extraction.leaf_attribution_review.reviewer import (
    IvrLeafReviewer,
    LeafReviewContext,
    LeafReviewer,
    LeafReviewOutcome,
    apply_review,
)
from mellea_lrc.llm.profiles import OPENROUTER_LUNA
from mellea_lrc.model.citation_chronology import citation_chronology
from mellea_lrc.model.citations import AttributionResult, IdCitation, latest
from mellea_lrc.model.citations.history import WITHDRAWN_ROOT_ID
from mellea_lrc.model.document import Document

STAGE = "33_id_attribution"
MODEL_PROFILE = OPENROUTER_LUNA


async def attribute_id_citations(
    document: Document, *, review: bool = True, reviewer: LeafReviewer | None = None
) -> Document:
    """Apply chronology rules and optionally audit each Id.'s antecedent.

    Review includes rule-attached Ids: a procedural rule or declaration can
    interrupt the case chain without being recognized by eyecite. Rule and
    model decisions have separate citation-local nodes in this atomic stage.
    Each final attachment is visible before the following Id. is considered.
    Source span and pinpoint readings remain unchanged.
    """
    require_leaves(document, STAGE)
    if CREATION_STAGE not in document.stage_runs:
        raise ValueError("Create Id. citations before attributing them")
    # Do not filter this stream to known cases: statute/journal/unknown events
    # break a case antecedent. Root identity verdicts and pin page ranges do
    # not change which authority a subsequent Id. refers to.
    prior: str | None = None
    roots = {root.id: root for root in document.roots}
    for _, _, citation in citation_chronology(document):
        if citation is None:
            prior = None
            continue
        if isinstance(citation, IdCitation):
            updated = citation.record(STAGE)
            # A named reference may point forward, but Id. cannot inherit a
            # full citation that has not appeared at its own source position.
            candidates = (
                (prior,) if prior in roots and roots[prior].site_span.start < citation.site_span.start else ()
            )
            if candidates:
                updated = updated.with_attribution(
                    candidates, AttributionResult.ATTACHED, "Immediately preceding resolved case citation"
                ).with_root(prior)
            else:
                updated = updated.with_attribution(
                    (), AttributionResult.UNRESOLVED, "No resolved case antecedent in citation chronology"
                ).withdraw()
            document = document.replace_citation(updated)
            if review:
                context = LeafReviewContext.from_document(document, updated)
                if reviewer is None:
                    reviewer = IvrLeafReviewer.from_profile(MODEL_PROFILE)
                answer = await reviewer(context)
                outcome = answer if isinstance(answer, LeafReviewOutcome) else LeafReviewOutcome(answer)
                updated = apply_review(updated.record(STAGE), context, outcome)
                document = document.replace_citation(updated)
            citation = updated
        root = latest(citation.root_id)
        prior = root if root in roots and root != WITHDRAWN_ROOT_ID else None
    return document.complete(STAGE)
