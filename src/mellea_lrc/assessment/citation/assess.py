"""Assess one found full-case citation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from mellea_lrc.assessment.deterministic.case_name import assess_case_name_exact_match
from mellea_lrc.assessment.deterministic.context import find_text_span_near_full_span, get_extended_span_text
from mellea_lrc.assessment.deterministic.year import assess_year_exact_match
from mellea_lrc.assessment.llm.case_name import assess_case_name_with_mellea
from mellea_lrc.assessment.types import (
    CaseNameAssessmentStatus,
    CitationAssessmentResult,
    ModifiedExtractedCitation,
    ModifiedExtractedCitationProposal,
    YearAssessment,
)
from mellea_lrc.core.spans import Span

if TYPE_CHECKING:
    from mellea import MelleaSession


@dataclass(frozen=True, slots=True)
class CitationAssessmentBundle:
    """One citation assessment plus optional re-extraction history."""

    assessment: CitationAssessmentResult
    modified_citation: ModifiedExtractedCitation | None = None
    reassessment: CitationAssessmentResult | None = None


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
    if exact.status != CaseNameAssessmentStatus.NEEDS_ASSESSMENT or session is None:
        return CitationAssessmentBundle(
            assessment=CitationAssessmentResult(
                citation_id=citation_id,
                case_assess=exact,
                year_assess=year_assess,
            ),
        )

    document_context = get_extended_span_text(document_text, span)
    case_name_run = await assess_case_name_with_mellea(
        session,
        citation_id=citation_id,
        extracted_case_name=extracted_case_name,
        courtlistener_case_name=courtlistener_case_name,
        document_context=document_context,
    )
    reassessment = (
        CitationAssessmentResult(
            citation_id=citation_id,
            case_assess=case_name_run.reassessment,
            year_assess=year_assess,
        )
        if case_name_run.reassessment is not None
        else None
    )
    return CitationAssessmentBundle(
        assessment=CitationAssessmentResult(
            citation_id=citation_id,
            case_assess=case_name_run.assessment,
            year_assess=year_assess,
        ),
        modified_citation=bind_modified_citation(
            case_name_run.modified_citation,
            document_text=document_text,
            full_span=span,
            citation_id=citation_id,
        ),
        reassessment=reassessment,
    )


def bind_modified_citation(
    modified_citation: ModifiedExtractedCitationProposal | None,
    *,
    document_text: str,
    full_span: Span,
    citation_id: str,
) -> ModifiedExtractedCitation | None:
    """Bind a grounded re-extraction proposal to document-local citation identity."""
    if modified_citation is None or not modified_citation.extracted_case_name:
        return None
    modified_span = find_text_span_near_full_span(
        document_text,
        modified_citation.extracted_case_name,
        full_span,
    )
    return ModifiedExtractedCitation.from_proposal(
        modified_citation,
        citation_id=citation_id,
        span=modified_span,
    )
