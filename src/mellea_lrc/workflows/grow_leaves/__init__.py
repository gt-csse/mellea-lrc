"""Grow leaves through four citation forms and grounded field correction."""

from __future__ import annotations

from mellea_lrc.extraction.correct_leaf_fields.reviewer import LeafFieldCorrectionReviewer
from mellea_lrc.extraction.leaf_attribution_review.reviewer import LeafReviewer
from mellea_lrc.model.document import Document
from mellea_lrc.workflows import Checkpoint, validate_workflow_checkpoint
from mellea_lrc.workflows.grow_leaves.correct_leaf_readings import correct_leaf_readings
from mellea_lrc.workflows.grow_leaves.grow_id_leaves import grow_id_leaves
from mellea_lrc.workflows.grow_leaves.grow_reference_leaves import grow_reference_leaves
from mellea_lrc.workflows.grow_leaves.grow_short_reporter_leaves import grow_short_reporter_leaves
from mellea_lrc.workflows.grow_leaves.grow_supra_leaves import grow_supra_leaves


async def grow_leaves(
    document: Document,
    *,
    review_leaves: bool = True,
    reviewer: LeafReviewer | None = None,
    correction_reviewer: LeafFieldCorrectionReviewer | None = None,
    checkpoint: Checkpoint | None = None,
) -> Document:
    """Attach short forms to formed roots through five resumable groups."""
    omitted = () if review_leaves else ("grow_leaves.grow_supra_leaves.review_attributions_llm",)
    validate_workflow_checkpoint(document, "grow_leaves", omitted_substages=omitted)
    if "grow_leaves.grow_short_reporter_leaves" not in document.stage_runs:
        document = await grow_short_reporter_leaves(
            document, review_leaves=review_leaves, reviewer=reviewer, checkpoint=checkpoint
        )
    if "grow_leaves.grow_reference_leaves" not in document.stage_runs:
        document = await grow_reference_leaves(
            document, review_leaves=review_leaves, reviewer=reviewer, checkpoint=checkpoint
        )
    if "grow_leaves.grow_id_leaves" not in document.stage_runs:
        document = await grow_id_leaves(
            document, review_leaves=review_leaves, reviewer=reviewer, checkpoint=checkpoint
        )
    if "grow_leaves.grow_supra_leaves" not in document.stage_runs:
        document = await grow_supra_leaves(
            document, review_leaves=review_leaves, reviewer=reviewer, checkpoint=checkpoint
        )
    if "grow_leaves.correct_leaf_readings" not in document.stage_runs:
        document = await correct_leaf_readings(
            document,
            review_leaves=review_leaves,
            correction_reviewer=correction_reviewer,
            checkpoint=checkpoint,
        )
    return document


__all__ = [
    "correct_leaf_readings",
    "grow_id_leaves",
    "grow_leaves",
    "grow_reference_leaves",
    "grow_short_reporter_leaves",
    "grow_supra_leaves",
]
