"""Compose the support review stage of validate_all_citation_content."""

from __future__ import annotations

from mellea_lrc.model.document import Document
from mellea_lrc.validation.judge_reporter_citation_content import SUBSTAGE as JUDGMENT
from mellea_lrc.validation.judge_reporter_citation_content import judge_reporter_citation_content
from mellea_lrc.validation.reporter_pinpoint_review.reviewer import ReporterPinpointReviewer
from mellea_lrc.validation.review_reporter_citation_full_opinions_llm import SUBSTAGE as OPINIONS
from mellea_lrc.validation.review_reporter_citation_full_opinions_llm import (
    review_reporter_citation_full_opinions_llm,
)
from mellea_lrc.validation.review_reporter_citation_pages_llm import SUBSTAGE as PAGES
from mellea_lrc.validation.review_reporter_citation_pages_llm import (
    review_reporter_citation_pages_llm,
)
from mellea_lrc.workflows import Checkpoint, _finish_stage, _require_stage_prefix, _save_checkpoint

STAGE = "validate_all_citation_content.review_citation_content"


async def review_citation_content(
    document: Document,
    *,
    page_reviewer: ReporterPinpointReviewer | None = None,
    full_opinion_reviewer: ReporterPinpointReviewer | None = None,
    checkpoint: Checkpoint | None = None,
) -> Document:
    _require_stage_prefix(document, STAGE, omitted_substages=())
    if PAGES not in document.substage_runs:
        document = await review_reporter_citation_pages_llm(document, reviewer=page_reviewer)
        _save_checkpoint(document, checkpoint)
    if OPINIONS not in document.substage_runs:
        document = await review_reporter_citation_full_opinions_llm(document, reviewer=full_opinion_reviewer)
        _save_checkpoint(document, checkpoint)
    if JUDGMENT not in document.substage_runs:
        document = judge_reporter_citation_content(document)
        _save_checkpoint(document, checkpoint)

    return _finish_stage(document, STAGE, checkpoint)
