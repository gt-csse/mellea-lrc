"""Compose independent short-form discovery, reading, and attribution stages."""

from mellea_lrc.extraction.id_attribution import attribute_id_citations
from mellea_lrc.extraction.id_citations import find_id_citations
from mellea_lrc.extraction.leaf_attribution_review.reviewer import LeafReviewer
from mellea_lrc.extraction.reference_attribution import attribute_reference_citations
from mellea_lrc.extraction.reference_citations import find_reference_citations
from mellea_lrc.extraction.short_reporter_attribution import attribute_short_reporter_citations
from mellea_lrc.extraction.short_reporter_case_names import resolve_short_reporter_case_names
from mellea_lrc.extraction.short_reporter_colocations import resolve_short_reporter_colocations
from mellea_lrc.extraction.short_reporter_locator import find_short_reporter_citations
from mellea_lrc.extraction.supra_attribution_llm import review_supra_attributions
from mellea_lrc.extraction.supra_attribution_rule import attribute_supra_citations_rule
from mellea_lrc.extraction.supra_case_names import resolve_supra_case_names
from mellea_lrc.extraction.supra_citations import find_supra_citations
from mellea_lrc.extraction.supra_pin_cites import resolve_supra_pin_cites
from mellea_lrc.model.document import Document


async def grow_leaves(
    document: Document, *, review_leaves: bool = True, reviewer: LeafReviewer | None = None
) -> Document:
    """Grow leaves from formed roots, whether identity validation ran or not.

    This workflow neither searches external records nor revalidates root
    identity. Leaf review chooses an antecedent from the existing source tree.
    Pinpoint validity against an opinion remains a later validation stage.
    """
    document = find_short_reporter_citations(document)
    document = resolve_short_reporter_colocations(document)
    document = resolve_short_reporter_case_names(document)
    document = await attribute_short_reporter_citations(document, review=review_leaves, reviewer=reviewer)
    document = find_reference_citations(document)
    document = await attribute_reference_citations(document, review=review_leaves, reviewer=reviewer)
    document = find_id_citations(document)
    document = await attribute_id_citations(document, review=review_leaves, reviewer=reviewer)
    document = find_supra_citations(document)
    document = resolve_supra_case_names(document)
    document = resolve_supra_pin_cites(document)
    document = attribute_supra_citations_rule(document)
    if review_leaves:
        document = await review_supra_attributions(document, reviewer=reviewer)
    return document
