"""Compose the support review stage of validate_pincite."""

from __future__ import annotations

from mellea_lrc.model.document import Document
from mellea_lrc.validation.reporter_citation_full_opinion_review import SUBSTAGE as OPINIONS
from mellea_lrc.validation.reporter_citation_full_opinion_review import review_reporter_citation_full_opinions
from mellea_lrc.validation.reporter_citation_pinpoint_judgment import SUBSTAGE as JUDGMENT
from mellea_lrc.validation.reporter_citation_pinpoint_judgment import judge_reporter_citation_pinpoints
from mellea_lrc.validation.reporter_citation_pinpoint_page_review import SUBSTAGE as PAGES
from mellea_lrc.validation.reporter_citation_pinpoint_page_review import (
    review_reporter_citation_pinpoint_pages,
)
from mellea_lrc.validation.reporter_pinpoint_review.reviewer import ReporterPinpointReviewer
from mellea_lrc.workflows import Checkpoint, _finish_stage, _require_stage_prefix, _save_checkpoint

STAGE = "validate_pincite.support_review"


async def review_citation_support(
    document: Document,
    *,
    page_reviewer: ReporterPinpointReviewer | None = None,
    full_opinion_reviewer: ReporterPinpointReviewer | None = None,
    checkpoint: Checkpoint | None = None,
) -> Document:
    _require_stage_prefix(document, STAGE, omitted_substages=())
    if PAGES not in document.substage_runs:
        document = await review_reporter_citation_pinpoint_pages(document, reviewer=page_reviewer)
        _save_checkpoint(document, checkpoint)
    if OPINIONS not in document.substage_runs:
        document = await review_reporter_citation_full_opinions(document, reviewer=full_opinion_reviewer)
        _save_checkpoint(document, checkpoint)
    if JUDGMENT not in document.substage_runs:
        document = judge_reporter_citation_pinpoints(document)
        _save_checkpoint(document, checkpoint)

    return _finish_stage(document, STAGE, checkpoint)
