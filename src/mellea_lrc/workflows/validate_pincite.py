"""Compose the user-defined pinpoint-validation workflow, one step at a time."""

from mellea_lrc.model.document import Document
from mellea_lrc.validation.reporter_citation_full_opinion_review import review_reporter_citation_full_opinions
from mellea_lrc.validation.reporter_citation_opinion_review import (
    STAGE as OPINION_REVIEW_STAGE,
)
from mellea_lrc.validation.reporter_citation_opinion_review import (
    review_reporter_citation_opinions,
)
from mellea_lrc.validation.reporter_citation_opinion_review.reviewer import ReporterCitationOpinionReviewer
from mellea_lrc.validation.reporter_citation_page_resolution import resolve_reporter_citation_pages
from mellea_lrc.validation.reporter_citation_pinpoint_evidence import (
    prepare_reporter_citation_pinpoint_evidence,
)
from mellea_lrc.validation.reporter_citation_pinpoint_judgment import judge_reporter_citation_pinpoints
from mellea_lrc.validation.reporter_citation_pinpoint_page_review import (
    review_reporter_citation_pinpoint_pages,
)
from mellea_lrc.validation.reporter_citation_propositions import read_reporter_citation_propositions
from mellea_lrc.validation.reporter_citation_propositions.reviewer import ReporterCitationPropositionReviewer
from mellea_lrc.validation.reporter_pinpoint_review.reviewer import ReporterPinpointReviewer
from mellea_lrc.validation.reporter_root_opinion_page_index import index_reporter_root_opinion_pages
from mellea_lrc.validation.reporter_root_opinion_retrieval import (
    OpinionClient,
    reporter_root_opinion_retrieval,
)


async def validate_pincite(
    document: Document,
    *,
    client: OpinionClient | None = None,
    review_opinions: bool = True,
    reviewer: ReporterCitationOpinionReviewer | None = None,
    proposition_reviewer: ReporterCitationPropositionReviewer | None = None,
    page_reviewer: ReporterPinpointReviewer | None = None,
    full_opinion_reviewer: ReporterPinpointReviewer | None = None,
) -> Document:
    """Validate reporter-root and leaf pinpoints through explicit checkpoints.

    Identity is an upstream eligibility policy. Source bodies and pagination
    remain root-owned; each occurrence reads its own proposition and pinpoint.
    Page review comes first, then full-opinion review only where needed. Full
    source prefixes are stable across occurrences to permit provider KV caching.
    No identity, root attachment, or extraction reading is overwritten here.
    """
    document = reporter_root_opinion_retrieval(document, client=client)
    document = index_reporter_root_opinion_pages(document)
    document = resolve_reporter_citation_pages(document)
    if review_opinions:
        document = await review_reporter_citation_opinions(document, reviewer=reviewer)
    else:
        document = document.complete(OPINION_REVIEW_STAGE)
    document = await read_reporter_citation_propositions(document, reviewer=proposition_reviewer)
    document = prepare_reporter_citation_pinpoint_evidence(document)
    document = await review_reporter_citation_pinpoint_pages(document, reviewer=page_reviewer)
    document = await review_reporter_citation_full_opinions(document, reviewer=full_opinion_reviewer)
    return judge_reporter_citation_pinpoints(document)
