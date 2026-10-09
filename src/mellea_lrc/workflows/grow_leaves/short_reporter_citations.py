"""Compose the short reporter citations stage of grow_leaves."""

from __future__ import annotations

from mellea_lrc.extraction.leaf_attribution_review.reviewer import LeafReviewer
from mellea_lrc.extraction.short_reporter_attribution import SUBSTAGE as ATTRIBUTION
from mellea_lrc.extraction.short_reporter_attribution import attribute_short_reporter_citations
from mellea_lrc.extraction.short_reporter_case_names import SUBSTAGE as CASE_NAMES
from mellea_lrc.extraction.short_reporter_case_names import resolve_short_reporter_case_names
from mellea_lrc.extraction.short_reporter_colocations import SUBSTAGE as COLOCATIONS
from mellea_lrc.extraction.short_reporter_colocations import resolve_short_reporter_colocations
from mellea_lrc.extraction.short_reporter_locator import SUBSTAGE as DISCOVERY
from mellea_lrc.extraction.short_reporter_locator import find_short_reporter_citations
from mellea_lrc.model.document import Document
from mellea_lrc.workflows import Checkpoint, _finish_stage, _require_stage_prefix, _save_checkpoint

STAGE = "grow_leaves.short_reporter_citations"


async def grow_short_reporter_leaves(
    document: Document,
    *,
    review_leaves: bool = True,
    reviewer: LeafReviewer | None = None,
    checkpoint: Checkpoint | None = None,
) -> Document:
    _require_stage_prefix(document, STAGE, omitted_substages=())
    if DISCOVERY not in document.substage_runs:
        document = find_short_reporter_citations(document)
        _save_checkpoint(document, checkpoint)
    if COLOCATIONS not in document.substage_runs:
        document = resolve_short_reporter_colocations(document)
        _save_checkpoint(document, checkpoint)
    if CASE_NAMES not in document.substage_runs:
        document = resolve_short_reporter_case_names(document)
        _save_checkpoint(document, checkpoint)
    if ATTRIBUTION not in document.substage_runs:
        document = await attribute_short_reporter_citations(document, review=review_leaves, reviewer=reviewer)
        _save_checkpoint(document, checkpoint)

    return _finish_stage(document, STAGE, checkpoint)
