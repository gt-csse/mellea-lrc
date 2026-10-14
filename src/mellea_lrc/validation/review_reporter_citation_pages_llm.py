"""Review only located pinpoint pages before reading the whole opinion."""

from mellea_lrc.llm.profiles import load_profile
from mellea_lrc.model.citations.reporter_pinpoint import OpinionReviewScope, PinpointEvidenceOutcome
from mellea_lrc.model.document import Document
from mellea_lrc.validation.reporter_pinpoint_review import review_citation
from mellea_lrc.validation.reporter_pinpoint_review.reviewer import (
    IvrReporterPinpointReviewer,
    ReporterPinpointReviewer,
)

SUBSTAGE = "validate_all_citation_content.review_citation_content.review_cited_pages_llm"
SOURCE_SUBSTAGE = "validate_all_citation_content.prepare_citation_evidence.prepare_evidence"


async def review_reporter_citation_pages_llm(
    document: Document, *, reviewer: ReporterPinpointReviewer | None = None
) -> Document:
    if SUBSTAGE in document.substage_runs:
        raise ValueError(f"Substage already completed: {SUBSTAGE}")
    if SOURCE_SUBSTAGE not in document.substage_runs:
        raise ValueError("Prepare reporter pinpoint evidence before page review")
    service = reviewer
    for citation in document.citations:
        if (
            citation.is_toa()
            or not citation.reporter_pinpoint_evidence
            or citation.reporter_pinpoint_evidence[-1].outcome is not PinpointEvidenceOutcome.READY
        ):
            continue
        if service is None:
            service = IvrReporterPinpointReviewer.from_profile(load_profile(SUBSTAGE))
        document = await review_citation(
            document, citation, substage=SUBSTAGE, scope=OpinionReviewScope.CITED_PAGES, reviewer=service
        )
    return document.complete_substage(SUBSTAGE)
