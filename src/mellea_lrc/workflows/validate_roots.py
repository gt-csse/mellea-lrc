"""Compose the currently implemented root-identity validation stages."""

from __future__ import annotations

from mellea_lrc.model.document import Document
from mellea_lrc.validation import (
    docket_root_lookup,
    docket_root_lookup_review,
    reporter_root_lookup,
    reporter_root_lookup_ambiguous,
    reporter_root_lookup_ambiguous_llm,
    reporter_root_lookup_unique_llm,
)


async def validate_roots(document: Document) -> Document:
    """Run docket candidate review, then reporter exact lookup and review.

    Docket retrieval and its model choice preserve independent field assessments;
    overall docket identity is a subsequent stage. Reporter search and
    large-candidate review likewise remain later stages.
    """
    document = docket_root_lookup(document)
    document = await docket_root_lookup_review(document)
    document = reporter_root_lookup(document)
    document = reporter_root_lookup_ambiguous(document)
    document = await reporter_root_lookup_unique_llm(document)
    document = await reporter_root_lookup_ambiguous_llm(document)
    # TODO: Review opinion evidence for a securely identified record whose
    # cited court or date still disagrees. The cluster's linked docket may
    # contain imported court metadata that identifies the case but assigns
    # the wrong court. A docket filing date marks case initiation, not an
    # opinion date. The cluster date is shared across grouped opinions, while
    # the specific subopinion or order may print another signed or issued
    # date. Retrieve that opinion/subopinion text or its original court PDF
    # and compare its header and signature before changing field judgments.
    return document
