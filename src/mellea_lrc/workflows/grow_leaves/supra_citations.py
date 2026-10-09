"""Compose the supra citations stage of grow_leaves."""

from __future__ import annotations

from mellea_lrc.extraction.leaf_attribution_review.reviewer import LeafReviewer
from mellea_lrc.extraction.supra_attribution_llm import SUBSTAGE as REVIEW
from mellea_lrc.extraction.supra_attribution_llm import review_supra_attributions
from mellea_lrc.extraction.supra_attribution_rule import SUBSTAGE as RULE
from mellea_lrc.extraction.supra_attribution_rule import attribute_supra_citations_rule
from mellea_lrc.extraction.supra_case_names import SUBSTAGE as CASE_NAMES
from mellea_lrc.extraction.supra_case_names import resolve_supra_case_names
from mellea_lrc.extraction.supra_citations import SUBSTAGE as DISCOVERY
from mellea_lrc.extraction.supra_citations import find_supra_citations
from mellea_lrc.extraction.supra_pin_cites import SUBSTAGE as PIN_CITES
from mellea_lrc.extraction.supra_pin_cites import resolve_supra_pin_cites
from mellea_lrc.model.document import Document
from mellea_lrc.workflows import Checkpoint, _finish_stage, _require_stage_prefix, _save_checkpoint

STAGE = "grow_leaves.supra_citations"


async def grow_supra_leaves(
    document: Document,
    *,
    review_leaves: bool = True,
    reviewer: LeafReviewer | None = None,
    checkpoint: Checkpoint | None = None,
) -> Document:
    _require_stage_prefix(document, STAGE, omitted_substages=() if review_leaves else (REVIEW,))
    if DISCOVERY not in document.substage_runs:
        document = find_supra_citations(document)
        _save_checkpoint(document, checkpoint)
    if CASE_NAMES not in document.substage_runs:
        document = resolve_supra_case_names(document)
        _save_checkpoint(document, checkpoint)
    if PIN_CITES not in document.substage_runs:
        document = resolve_supra_pin_cites(document)
        _save_checkpoint(document, checkpoint)
    if RULE not in document.substage_runs:
        document = attribute_supra_citations_rule(document)
        _save_checkpoint(document, checkpoint)
    if review_leaves and REVIEW not in document.substage_runs:
        document = await review_supra_attributions(document, reviewer=reviewer)
        _save_checkpoint(document, checkpoint)

    return _finish_stage(document, STAGE, checkpoint)
