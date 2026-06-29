"""Assess one found full-case citation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from mellea_lrc.assessment.deterministic.case_name import assess_case_name_exact_match
from mellea_lrc.assessment.deterministic.context import find_text_span_near_full_span, get_extended_span_text
from mellea_lrc.assessment.deterministic.year import assess_year_exact_match
from mellea_lrc.assessment.llm.case_name import assess_case_name_with_mellea
from mellea_lrc.assessment.types import (
    CaseNameReassessed,
    CaseNameReassessmentFailed,
    CaseNameReassessmentNotRequired,
    CaseNameReextractionFailed,
    CitationAssessmentResult,
    CitationReassessment,
    ModifiedExtractedCitation,
    ModifiedExtractedCitationProposal,
    ReassessedCitationReassessment,
    ReassessmentFailedCitationReassessment,
    ReassessmentSkipReason,
    ReextractionFailedCitationReassessment,
    SkippedCitationReassessment,
)

if TYPE_CHECKING:
    from mellea import MelleaSession

    from mellea_lrc.core.spans import Span


@dataclass(frozen=True, slots=True)
class CitationAssessmentBundle:
    """One citation assessment and its explicit reassessment state."""

    assessment: CitationAssessmentResult
    reassessment: CitationReassessment


async def assess_found_citation(
    *,
    citation_id: str,
    document_text: str,
    span: Span,
    extracted_case_name: str | None,
    courtlistener_case_name: str | None,
    extracted_year: str | None,
    courtlistener_year: str | None,
    session: MelleaSession | None = None,
) -> CitationAssessmentBundle:
    """Run deterministic checks, then Mellea when the case name still needs assessment."""
    year_assess = assess_year_exact_match(
        citation_id=citation_id,
        extracted_year=extracted_year,
        courtlistener_year=courtlistener_year,
    )
    exact = assess_case_name_exact_match(
        citation_id=citation_id,
        extracted_case_name=extracted_case_name,
        courtlistener_case_name=courtlistener_case_name,
    )
    if exact is not None:
        return CitationAssessmentBundle(
            assessment=CitationAssessmentResult(
                citation_id=citation_id,
                case_assess=exact,
                year_assess=year_assess,
            ),
            reassessment=SkippedCitationReassessment(
                citation_id=citation_id,
                reason=ReassessmentSkipReason.REEXTRACTION_NOT_REQUIRED,
                message="Primary assessment completed without re-extraction.",
            ),
        )
    if session is None:
        msg = "A Mellea session is required for a non-exact case-name assessment"
        raise RuntimeError(msg)

    document_context = get_extended_span_text(document_text, span)
    case_name_run = await assess_case_name_with_mellea(
        session,
        citation_id=citation_id,
        extracted_case_name=extracted_case_name,
        courtlistener_case_name=courtlistener_case_name,
        document_context=document_context,
    )
    reassessment = _build_citation_reassessment(
        case_name_run.reassessment,
        citation_id=citation_id,
        document_text=document_text,
        full_span=span,
    )
    return CitationAssessmentBundle(
        assessment=CitationAssessmentResult(
            citation_id=citation_id,
            case_assess=case_name_run.assessment,
            year_assess=year_assess,
        ),
        reassessment=reassessment,
    )


def _build_citation_reassessment(
    outcome: (
        CaseNameReassessmentNotRequired
        | CaseNameReextractionFailed
        | CaseNameReassessed
        | CaseNameReassessmentFailed
    ),
    *,
    citation_id: str,
    document_text: str,
    full_span: Span,
) -> CitationReassessment:
    if isinstance(outcome, CaseNameReassessmentNotRequired):
        return SkippedCitationReassessment(
            citation_id=citation_id,
            reason=ReassessmentSkipReason.REEXTRACTION_NOT_REQUIRED,
            message="Primary assessment completed without re-extraction.",
        )
    if isinstance(outcome, CaseNameReextractionFailed):
        return ReextractionFailedCitationReassessment(citation_id=citation_id, error=outcome.error)

    modified = bind_modified_citation(
        outcome.modified_citation,
        document_text=document_text,
        full_span=full_span,
        citation_id=citation_id,
    )
    if modified is None:
        msg = "Accepted re-extraction did not produce a modified citation"
        raise RuntimeError(msg)
    if isinstance(outcome, CaseNameReassessmentFailed):
        return ReassessmentFailedCitationReassessment(
            citation_id=citation_id,
            modified_citation=modified,
            error=outcome.error,
        )
    return ReassessedCitationReassessment(
        citation_id=citation_id,
        modified_citation=modified,
        result=outcome.reassessment,
    )


def bind_modified_citation(
    modified_citation: ModifiedExtractedCitationProposal | None,
    *,
    document_text: str,
    full_span: Span,
    citation_id: str,
) -> ModifiedExtractedCitation | None:
    """Bind a grounded re-extraction proposal to document-local citation identity."""
    if modified_citation is None or not modified_citation.case_name:
        return None
    modified_span = find_text_span_near_full_span(
        document_text,
        modified_citation.case_name,
        full_span,
    )
    return ModifiedExtractedCitation.from_proposal(
        modified_citation,
        citation_id=citation_id,
        span=modified_span,
    )
