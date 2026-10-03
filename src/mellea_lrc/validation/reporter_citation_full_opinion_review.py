"""Review complete saved opinions when the selected pages do not settle support."""

from mellea_lrc.llm.profiles import NRP_QWEN
from mellea_lrc.model.citations.full_reporter import FullReporterCitation
from mellea_lrc.model.citations.reporter_pinpoint import (
    OpinionReviewScope,
    OpinionSupportResult,
    PinpointEvidenceOutcome,
    ReporterCitationSupportReview,
)
from mellea_lrc.model.document import Document
from mellea_lrc.validation.reporter_pinpoint_review import review_citation
from mellea_lrc.validation.reporter_pinpoint_review.reviewer import (
    IvrReporterPinpointReviewer,
    ReporterPinpointReviewer,
)

STAGE = "46_reporter_citation_full_opinion_review"
MODEL_PROFILE = NRP_QWEN
SOURCE_STAGE = "45_reporter_citation_pinpoint_page_review"


async def review_reporter_citation_full_opinions(
    document: Document, *, reviewer: ReporterPinpointReviewer | None = None
) -> Document:
    """Reuse a byte-identical root prefix across all its full-opinion reviews.

    A page-only negative is provisional. Missing or ambiguous pages enter this
    route directly. Source text is never retrieved again, and a failed call is
    saved without producing a false negative verdict.
    """
    if STAGE in document.stage_runs:
        raise ValueError(f"Stage already completed: {STAGE}")
    if SOURCE_STAGE not in document.stage_runs:
        raise ValueError("Complete page review before full-opinion review")
    service = reviewer
    # Reviews are occurrence-independent within this atomic stage. Grouping by
    # root keeps identical full-source prefixes adjacent for provider KV reuse.
    ordered = sorted(
        document.citations,
        key=lambda citation: (
            citation.reporter_pinpoint_evidence[-1].root_id if citation.reporter_pinpoint_evidence else "",
            citation.site_span.start,
        ),
    )
    for citation in ordered:
        if not citation.reporter_pinpoint_evidence:
            continue
        evidence = citation.reporter_pinpoint_evidence[-1]
        if evidence.outcome not in {PinpointEvidenceOutcome.READY, PinpointEvidenceOutcome.MISSING_PAGES}:
            continue
        prior = next(
            (
                item
                for item in reversed(citation.reporter_support_reviews)
                if item.evidence_index == len(citation.reporter_pinpoint_evidence) - 1
            ),
            None,
        )
        if (
            prior is not None
            and prior.decision is not None
            and prior.decision.result is OpinionSupportResult.SUPPORTED
            and prior.decision.correct_page is True
        ):
            continue
        root = next((item for item in document.roots if item.id == evidence.root_id), None)
        if (
            isinstance(root, FullReporterCitation)
            and root.reporter_root_opinion_page_index is not None
            and not any(opinion.text.strip() for opinion in root.reporter_root_opinion_page_index.opinions)
        ):
            recorded = citation.record(STAGE)
            recorded = recorded.with_reporter_support_review(
                ReporterCitationSupportReview(
                    node_id=recorded.nodes[-1].id,
                    evidence_index=len(citation.reporter_pinpoint_evidence) - 1,
                    scope=OpinionReviewScope.FULL_OPINION,
                    decision=None,
                    failure_reason="The saved root opinion bundle contains no nonempty text.",
                )
            )
            document = document.replace_citation(recorded)
            continue
        if service is None:
            service = IvrReporterPinpointReviewer.from_profile(MODEL_PROFILE)
        document = await review_citation(
            document, citation, stage=STAGE, scope=OpinionReviewScope.FULL_OPINION, reviewer=service
        )
    return document.complete(STAGE)
