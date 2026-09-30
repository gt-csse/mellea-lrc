"""Compose independent short-form discovery, reading, and attribution stages."""

from mellea_lrc.extraction.id_attribution import attribute_id_citations
from mellea_lrc.extraction.id_attribution_llm import review_id_attributions
from mellea_lrc.extraction.id_citations import find_id_citations
from mellea_lrc.extraction.leaf_attribution_llm import review_leaf_attributions
from mellea_lrc.extraction.leaf_attribution_llm.reviewer import LeafReviewer
from mellea_lrc.extraction.leaf_attribution_rule import attribute_leaves_rule
from mellea_lrc.extraction.leaf_case_names import resolve_leaf_case_names
from mellea_lrc.extraction.leaf_pin_cites import resolve_leaf_pin_cites
from mellea_lrc.extraction.reference_citations import find_reference_citations
from mellea_lrc.extraction.short_reporter_locator import find_short_reporter_citations
from mellea_lrc.extraction.supra_citations import find_supra_citations
from mellea_lrc.model.document import Document


async def grow_leaves(
    document: Document, *, review_leaves: bool = True, reviewer: LeafReviewer | None = None
) -> Document:
    """Grow leaves from formed roots, whether identity validation ran or not.

    This workflow neither searches external records nor revalidates root
    identity. Leaf review chooses an antecedent from the existing source tree.
    Pinpoint validity against an opinion remains a later validation workflow.
    """
    document = find_short_reporter_citations(document)
    document = find_supra_citations(document)
    document = find_id_citations(document)
    document = find_reference_citations(document)
    document = resolve_leaf_case_names(document)
    document = resolve_leaf_pin_cites(document)
    document = attribute_leaves_rule(document)
    if review_leaves:
        document = await review_leaf_attributions(document, reviewer=reviewer)
    document = attribute_id_citations(document)
    if review_leaves:
        document = await review_id_attributions(document, reviewer=reviewer)
    return document
