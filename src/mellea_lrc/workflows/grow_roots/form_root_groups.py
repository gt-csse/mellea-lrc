"""Compose the root formation stage of grow_roots."""

from __future__ import annotations

from mellea_lrc.extraction.form_roots import SUBSTAGE as RULE
from mellea_lrc.extraction.form_roots import form_roots
from mellea_lrc.extraction.reassign_docket_roots_llm import SUBSTAGE as REASSIGNMENT
from mellea_lrc.extraction.reassign_docket_roots_llm import reassign_docket_roots_llm
from mellea_lrc.extraction.reassign_docket_roots_llm.reviewer import DocketRootReviewer
from mellea_lrc.model.document import Document
from mellea_lrc.workflows import Checkpoint, _finish_stage, _require_stage_prefix, _save_checkpoint

STAGE = "grow_roots.form_root_groups"


async def form_root_groups(
    document: Document,
    *,
    review_docket_roots: bool = False,
    docket_root_reviewer: DocketRootReviewer | None = None,
    checkpoint: Checkpoint | None = None,
) -> Document:
    _require_stage_prefix(document, STAGE, omitted_substages=() if review_docket_roots else (REASSIGNMENT,))
    if RULE not in document.substage_runs:
        document = form_roots(document)
        _save_checkpoint(document, checkpoint)
    if review_docket_roots and REASSIGNMENT not in document.substage_runs:
        document = await reassign_docket_roots_llm(document, reviewer=docket_root_reviewer)
        _save_checkpoint(document, checkpoint)

    return _finish_stage(document, STAGE, checkpoint)
