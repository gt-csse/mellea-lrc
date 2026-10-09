"""Compose the root formation stage of grow_roots."""

from __future__ import annotations

from mellea_lrc.extraction.docket_root_llm_reassignment import SUBSTAGE as REASSIGNMENT
from mellea_lrc.extraction.docket_root_llm_reassignment import docket_root_llm_reassignment
from mellea_lrc.extraction.docket_root_llm_reassignment.reviewer import DocketRootReviewer
from mellea_lrc.extraction.roots import SUBSTAGE as RULE
from mellea_lrc.extraction.roots import form_roots
from mellea_lrc.model.document import Document
from mellea_lrc.workflows import Checkpoint, _finish_stage, _require_stage_prefix, _save_checkpoint

STAGE = "grow_roots.root_formation"


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
        document = await docket_root_llm_reassignment(document, reviewer=docket_root_reviewer)
        _save_checkpoint(document, checkpoint)

    return _finish_stage(document, STAGE, checkpoint)
