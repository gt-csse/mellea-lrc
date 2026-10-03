"""Review only located pinpoint pages before reading the whole opinion."""

from mellea_lrc.llm.profiles import NRP_GLM
from mellea_lrc.model.citations.reporter_pinpoint import OpinionReviewScope, PinpointEvidenceOutcome
from mellea_lrc.model.document import Document
from mellea_lrc.validation.reporter_pinpoint_review import review_citation
from mellea_lrc.validation.reporter_pinpoint_review.reviewer import (
    IvrReporterPinpointReviewer,
    ReporterPinpointReviewer,
)

STAGE = "45_reporter_citation_pinpoint_page_review"
MODEL_PROFILE = NRP_GLM
SOURCE_STAGE = "44_reporter_citation_pinpoint_evidence"


async def review_reporter_citation_pinpoint_pages(
    document: Document, *, reviewer: ReporterPinpointReviewer | None = None
) -> Document:
    if STAGE in document.stage_runs:
        raise ValueError(f"Stage already completed: {STAGE}")
    if SOURCE_STAGE not in document.stage_runs:
        raise ValueError("Prepare reporter pinpoint evidence before page review")
    service = reviewer
    for citation in document.citations:
        if (
            not citation.reporter_pinpoint_evidence
            or citation.reporter_pinpoint_evidence[-1].outcome is not PinpointEvidenceOutcome.READY
        ):
            continue
        if service is None:
            service = IvrReporterPinpointReviewer.from_profile(MODEL_PROFILE)
        document = await review_citation(
            document, citation, stage=STAGE, scope=OpinionReviewScope.CITED_PAGES, reviewer=service
        )
    return document.complete(STAGE)
