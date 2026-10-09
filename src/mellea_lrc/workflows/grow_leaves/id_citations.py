"""Compose the id citations stage of grow_leaves."""

from __future__ import annotations

from mellea_lrc.extraction.id_attribution import SUBSTAGE as ATTRIBUTION
from mellea_lrc.extraction.id_attribution import attribute_id_citations
from mellea_lrc.extraction.id_citations import SUBSTAGE as DISCOVERY
from mellea_lrc.extraction.id_citations import find_id_citations
from mellea_lrc.extraction.leaf_attribution_review.reviewer import LeafReviewer
from mellea_lrc.model.document import Document
from mellea_lrc.workflows import Checkpoint, _finish_stage, _require_stage_prefix, _save_checkpoint

STAGE = "grow_leaves.id_citations"


async def grow_id_leaves(
    document: Document,
    *,
    review_leaves: bool = True,
    reviewer: LeafReviewer | None = None,
    checkpoint: Checkpoint | None = None,
) -> Document:
    _require_stage_prefix(document, STAGE, omitted_substages=())
    if DISCOVERY not in document.substage_runs:
        document = find_id_citations(document)
        _save_checkpoint(document, checkpoint)
    if ATTRIBUTION not in document.substage_runs:
        document = await attribute_id_citations(document, review=review_leaves, reviewer=reviewer)
        _save_checkpoint(document, checkpoint)

    return _finish_stage(document, STAGE, checkpoint)
