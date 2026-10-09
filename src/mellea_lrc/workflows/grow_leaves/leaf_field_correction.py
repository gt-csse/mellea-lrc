"""Compose the leaf field correction stage of grow_leaves."""

from __future__ import annotations

from mellea_lrc.extraction.leaf_field_corrections import SUBSTAGE as REVIEW
from mellea_lrc.extraction.leaf_field_corrections import correct_leaf_fields
from mellea_lrc.extraction.leaf_field_corrections.reviewer import LeafFieldCorrectionReviewer
from mellea_lrc.model.document import Document
from mellea_lrc.workflows import Checkpoint, _finish_stage, _require_stage_prefix, _save_checkpoint

STAGE = "grow_leaves.leaf_field_correction"


async def correct_leaf_readings(
    document: Document,
    *,
    review_leaves: bool = True,
    correction_reviewer: LeafFieldCorrectionReviewer | None = None,
    checkpoint: Checkpoint | None = None,
) -> Document:
    _require_stage_prefix(document, STAGE, omitted_substages=())
    if REVIEW not in document.substage_runs:
        document = await correct_leaf_fields(document, review=review_leaves, reviewer=correction_reviewer)
        _save_checkpoint(document, checkpoint)

    return _finish_stage(document, STAGE, checkpoint)
