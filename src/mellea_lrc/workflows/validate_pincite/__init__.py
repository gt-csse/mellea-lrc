"""Validate citation support through opinion preparation, evidence, and review."""

from __future__ import annotations

from mellea_lrc.model.document import Document
from mellea_lrc.validation.reporter_citation_opinion_review.reviewer import ReporterCitationOpinionReviewer
from mellea_lrc.validation.reporter_citation_propositions.reviewer import ReporterCitationPropositionReviewer
from mellea_lrc.validation.reporter_pinpoint_review.reviewer import ReporterPinpointReviewer
from mellea_lrc.validation.reporter_root_opinion_retrieval import OpinionClient
from mellea_lrc.workflows import Checkpoint, _require_workflow_prefix
from mellea_lrc.workflows.validate_pincite.citation_preparation import prepare_citation_evidence
from mellea_lrc.workflows.validate_pincite.opinion_preparation import prepare_root_opinions
from mellea_lrc.workflows.validate_pincite.support_review import review_citation_support


async def validate_pincite(
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
    _require_workflow_prefix(document, "validate_pincite")
    if "validate_pincite.opinion_preparation" not in document.stage_runs:
        document = prepare_root_opinions(document, client=client, checkpoint=checkpoint)
    if "validate_pincite.citation_preparation" not in document.stage_runs:
        document = await prepare_citation_evidence(
            document,
            review_opinions=review_opinions,
            reviewer=reviewer,
            proposition_reviewer=proposition_reviewer,
            checkpoint=checkpoint,
        )
    if "validate_pincite.support_review" not in document.stage_runs:
        document = await review_citation_support(
            document,
            page_reviewer=page_reviewer,
            full_opinion_reviewer=full_opinion_reviewer,
            checkpoint=checkpoint,
        )
    return document


__all__ = [
    "prepare_citation_evidence",
    "prepare_root_opinions",
    "review_citation_support",
    "validate_pincite",
]
