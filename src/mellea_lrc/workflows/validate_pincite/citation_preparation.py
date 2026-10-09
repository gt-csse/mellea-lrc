"""Compose the citation preparation stage of validate_pincite."""

from __future__ import annotations

from mellea_lrc.model.document import Document
from mellea_lrc.validation.reporter_citation_opinion_review import SUBSTAGE as OPINION_REVIEW
from mellea_lrc.validation.reporter_citation_opinion_review import review_reporter_citation_opinions
from mellea_lrc.validation.reporter_citation_opinion_review.reviewer import ReporterCitationOpinionReviewer
from mellea_lrc.validation.reporter_citation_page_resolution import SUBSTAGE as PAGES
from mellea_lrc.validation.reporter_citation_page_resolution import resolve_reporter_citation_pages
from mellea_lrc.validation.reporter_citation_pinpoint_evidence import SUBSTAGE as EVIDENCE
from mellea_lrc.validation.reporter_citation_pinpoint_evidence import (
    prepare_reporter_citation_pinpoint_evidence,
)
from mellea_lrc.validation.reporter_citation_propositions import SUBSTAGE as PROPOSITIONS
from mellea_lrc.validation.reporter_citation_propositions import read_reporter_citation_propositions
from mellea_lrc.validation.reporter_citation_propositions.reviewer import ReporterCitationPropositionReviewer
from mellea_lrc.workflows import Checkpoint, _finish_stage, _require_stage_prefix, _save_checkpoint

STAGE = "validate_pincite.citation_preparation"


async def prepare_citation_evidence(
    document: Document,
    *,
    review_opinions: bool = True,
    reviewer: ReporterCitationOpinionReviewer | None = None,
    proposition_reviewer: ReporterCitationPropositionReviewer | None = None,
    checkpoint: Checkpoint | None = None,
) -> Document:
    _require_stage_prefix(document, STAGE, omitted_substages=())
    if PAGES not in document.substage_runs:
        document = resolve_reporter_citation_pages(document)
        _save_checkpoint(document, checkpoint)
    if OPINION_REVIEW not in document.substage_runs:
        if review_opinions:
            document = await review_reporter_citation_opinions(document, reviewer=reviewer)
        else:
            document = document.complete_substage(OPINION_REVIEW)
        _save_checkpoint(document, checkpoint)
    if PROPOSITIONS not in document.substage_runs:
        document = await read_reporter_citation_propositions(document, reviewer=proposition_reviewer)
        _save_checkpoint(document, checkpoint)
    if EVIDENCE not in document.substage_runs:
        document = prepare_reporter_citation_pinpoint_evidence(document)
        _save_checkpoint(document, checkpoint)

    return _finish_stage(document, STAGE, checkpoint)
