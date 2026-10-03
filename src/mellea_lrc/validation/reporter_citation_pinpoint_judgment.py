"""Derive support and location judgments from saved opinion reviews."""

from __future__ import annotations

from mellea_lrc.model.citations.citation import Citation
from mellea_lrc.model.citations.full_reporter import FullReporterCitation
from mellea_lrc.model.citations.reporter_opinion import OpinionRetrievalOutcome
from mellea_lrc.model.citations.reporter_pinpoint import (
    OpinionReviewScope,
    OpinionSupportResult,
    PinpointEvidenceOutcome,
    ReporterCitationSupportReview,
    ReporterOpinionEvidence,
    ReporterPinpointJudgment,
    ReporterPinpointVerdict,
)
from mellea_lrc.model.document import Document

STAGE = "47_reporter_citation_pinpoint_judgment"
SOURCE_STAGE = "46_reporter_citation_full_opinion_review"


def _selected_review(
    citation: Citation, evidence_index: int
) -> tuple[int, ReporterCitationSupportReview] | None:
    for scope in (OpinionReviewScope.FULL_OPINION, OpinionReviewScope.CITED_PAGES):
        for index in reversed(range(len(citation.reporter_support_reviews))):
            review = citation.reporter_support_reviews[index]
            if (
                review.evidence_index == evidence_index
                and review.scope is scope
                and review.decision is not None
            ):
                return index, review
    return None


def _complete_opinions(root: FullReporterCitation | None) -> bool:
    if (
        root is None
        or root.reporter_root_opinion_retrieval is None
        or root.reporter_root_opinion_page_index is None
    ):
        return False
    retrieval = root.reporter_root_opinion_retrieval
    indexed = {opinion.opinion_id: opinion for opinion in root.reporter_root_opinion_page_index.opinions}
    return bool(retrieval.opinions) and all(
        opinion.outcome is OpinionRetrievalOutcome.RETRIEVED
        and opinion.opinion_id in indexed
        and bool(indexed[opinion.opinion_id].text.strip())
        for opinion in retrieval.opinions
    )


def _grounded_quotes(
    citation: Citation, review: ReporterCitationSupportReview, root: FullReporterCitation | None
) -> tuple[ReporterOpinionEvidence, ...]:
    if not review.opinion_evidence_indices:
        return ()
    if root is None or root.reporter_root_opinion_page_index is None:
        raise ValueError("Opinion evidence requires its root's indexed opinion text")
    indexed = {opinion.opinion_id: opinion for opinion in root.reporter_root_opinion_page_index.opinions}
    quotes = tuple(citation.reporter_opinion_evidence[index] for index in review.opinion_evidence_indices)
    for quote in quotes:
        if quote.root_id != root.id or quote.opinion_id not in indexed:
            raise ValueError("Opinion evidence must belong to its reviewed root")
        quote.validate_source(indexed[quote.opinion_id].text)
    return quotes


def judge_reporter_citation_pinpoints(document: Document) -> Document:
    """A page-only negative or incomplete retrieval cannot establish wrong support."""
    if STAGE in document.stage_runs:
        raise ValueError(f"Stage already completed: {STAGE}")
    if SOURCE_STAGE not in document.stage_runs:
        raise ValueError("Complete full-opinion support review before judging reporter pinpoints")
    by_id = {citation.id: citation for citation in document.citations}
    for citation in document.citations:
        if not citation.reporter_pinpoint_evidence:
            continue
        evidence_index = len(citation.reporter_pinpoint_evidence) - 1
        evidence = citation.reporter_pinpoint_evidence[evidence_index]
        if evidence.outcome in {PinpointEvidenceOutcome.NO_PINCITE, PinpointEvidenceOutcome.NO_PROPOSITION}:
            continue
        candidate = by_id.get(evidence.root_id)
        root = candidate if isinstance(candidate, FullReporterCitation) else None
        selected = _selected_review(citation, evidence_index)
        review_index, review = selected if selected is not None else (None, None)
        verdict = ReporterPinpointVerdict.UNDETERMINED
        page_assessment = {"pagination_available": False, "correct_page": None, "found_pages": ()}
        reason = "No successful opinion support review is available."
        if review is not None:
            decision = review.decision
            assert decision is not None
            _grounded_quotes(citation, review, root)
            page_assessment = {
                "pagination_available": decision.pagination_available,
                "correct_page": decision.correct_page,
                "found_pages": decision.found_pages,
            }
            reason = decision.reason
            if decision.result is OpinionSupportResult.SUPPORTED:
                verdict = ReporterPinpointVerdict.CORRECT_PINCITE
                if decision.correct_page is False:
                    reason += " The opinion supports the attribution at another target."
                elif decision.correct_page is None:
                    reason += " Support is established, but its reporter target is unlocated."
            elif review.scope is OpinionReviewScope.FULL_OPINION:
                if decision.result is OpinionSupportResult.CONTRADICTED:
                    verdict = ReporterPinpointVerdict.WRONG_PINCITE
                elif decision.result is OpinionSupportResult.NOT_FOUND:
                    if _complete_opinions(root):
                        verdict = ReporterPinpointVerdict.WRONG_PINCITE
                    else:
                        reason += " Missing or empty opinion text prevents proving absence of support."
            elif decision.result in {OpinionSupportResult.CONTRADICTED, OpinionSupportResult.NOT_FOUND}:
                reason += " A negative review of cited pages alone does not establish absent opinion support."
        if evidence.outcome is not PinpointEvidenceOutcome.READY:
            reason += f" Evidence preparation: {evidence.reason}"
        recorded = citation.record(STAGE)
        recorded = recorded.with_reporter_pinpoint_judgment(
            ReporterPinpointJudgment(
                node_id=recorded.nodes[-1].id,
                evidence_index=evidence_index,
                review_index=review_index,
                verdict=verdict,
                **page_assessment,
                reason=reason,
            )
        )
        document = document.replace_citation(recorded)
    return document.complete(STAGE)
