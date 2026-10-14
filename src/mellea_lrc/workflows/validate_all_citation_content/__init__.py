"""Validate citation support through opinion preparation, evidence, and review."""

from __future__ import annotations

from mellea_lrc.model.document import Document
from mellea_lrc.validation.read_reporter_citation_propositions.reviewer import (
    ReporterCitationPropositionReviewer,
)
from mellea_lrc.validation.reporter_pinpoint_review.reviewer import ReporterPinpointReviewer
from mellea_lrc.validation.retrieve_reporter_root_opinions import OpinionClient
from mellea_lrc.validation.select_reporter_citation_opinions.reviewer import ReporterCitationOpinionReviewer
from mellea_lrc.workflows import Checkpoint, validate_workflow_checkpoint
from mellea_lrc.workflows.validate_all_citation_content.prepare_citation_evidence import (
    prepare_citation_evidence,
)
from mellea_lrc.workflows.validate_all_citation_content.prepare_root_opinions import prepare_root_opinions
from mellea_lrc.workflows.validate_all_citation_content.review_citation_content import review_citation_content


async def validate_all_citation_content(
    document: Document,
    *,
    client: OpinionClient | None = None,
    review_opinions: bool = True,
    reviewer: ReporterCitationOpinionReviewer | None = None,
    proposition_reviewer: ReporterCitationPropositionReviewer | None = None,
    page_reviewer: ReporterPinpointReviewer | None = None,
    full_opinion_reviewer: ReporterPinpointReviewer | None = None,
    checkpoint: Checkpoint | None = None,
) -> Document:
    """Resume three groups while keeping opinion bodies owned by roots."""
    validate_workflow_checkpoint(document, "validate_all_citation_content")
    if "validate_all_citation_content.prepare_root_opinions" not in document.stage_runs:
        document = prepare_root_opinions(document, client=client, checkpoint=checkpoint)
    if "validate_all_citation_content.prepare_citation_evidence" not in document.stage_runs:
        document = await prepare_citation_evidence(
            document,
            review_opinions=review_opinions,
            reviewer=reviewer,
            proposition_reviewer=proposition_reviewer,
            checkpoint=checkpoint,
        )
    if "validate_all_citation_content.review_citation_content" not in document.stage_runs:
        document = await review_citation_content(
            document,
            page_reviewer=page_reviewer,
            full_opinion_reviewer=full_opinion_reviewer,
            checkpoint=checkpoint,
        )
    return document


__all__ = [
    "prepare_citation_evidence",
    "prepare_root_opinions",
    "review_citation_content",
    "validate_all_citation_content",
]
