"""Document-level assessment pipeline."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING

from mellea_lrc.assessment.citation.assess import CitationAssessmentBundle, assess_found_citation
from mellea_lrc.assessment.citation.clusters import first_cluster_case_name, first_cluster_year
from mellea_lrc.assessment.deterministic.case_name import assess_case_name_exact_match, build_extracted_case_name
from mellea_lrc.assessment.deterministic.context import get_extended_span_text
from mellea_lrc.assessment.types import (
    CaseNameAssessmentStatus,
    CitationAssessment,
    DocumentAssessment,
    ModifiedExtractedCitation,
)
from mellea_lrc.core.citations import FullCaseCitation
from mellea_lrc.extraction.types import ExtractedCitation
from mellea_lrc.llm import start_mellea_session_from_env
from mellea_lrc.validation.types import ValidationStatus

if TYPE_CHECKING:
    from collections.abc import Callable

    from mellea_lrc.validation.types import DocumentValidation


@dataclass(frozen=True, slots=True)
class MelleaCallContext:
    """Context emitted immediately before one semantic assessment call."""

    mellea_call: int
    citation_id: str
    matched_text: str
    extracted_case_name: str | None
    courtlistener_case_name: str | None
    context: str


@dataclass(frozen=True, slots=True)
class _PendingMelleaAssessment:
    citation: ExtractedCitation
    document_text: str
    extracted_case_name: str | None
    courtlistener_case_name: str | None
    extracted_year: str | None
    courtlistener_year: str | None
    context: str
    mellea_call: int


def run_assessment(
    validation: DocumentValidation,
    *,
    max_mellea: int | None = None,
    mellea_concurrency: int | None = None,
    on_mellea_call: Callable[[MelleaCallContext], None] | None = None,
    on_mellea_done: Callable[[MelleaCallContext, CitationAssessment], None] | None = None,
) -> DocumentAssessment:
    """Assess a validated document synchronously."""
    return asyncio.run(
        run_assessment_async(
            validation,
            max_mellea=max_mellea,
            mellea_concurrency=mellea_concurrency,
            on_mellea_call=on_mellea_call,
            on_mellea_done=on_mellea_done,
        )
    )


async def run_assessment_async(
    validation: DocumentValidation,
    *,
    max_mellea: int | None = None,
    mellea_concurrency: int | None = None,
    on_mellea_call: Callable[[MelleaCallContext], None] | None = None,
    on_mellea_done: Callable[[MelleaCallContext, CitationAssessment], None] | None = None,
) -> DocumentAssessment:
    """Assess a validated document with Mellea-backed case-name checks."""
    validations_by_id = {item.citation_id: item for item in validation.validations}
    assessments: list[CitationAssessment] = []
    modified_citations: list[ModifiedExtractedCitation] = []
    reassessments: list[CitationAssessment] = []
    pending: list[_PendingMelleaAssessment] = []

    for citation in validation.citations:
        citation_validation = validations_by_id.get(citation.citation_id)
        if citation_validation is None:
            continue
        if not isinstance(citation.citation, FullCaseCitation):
            continue
        if citation_validation.status != ValidationStatus.FOUND:
            continue

        extracted_case_name = build_extracted_case_name(citation.citation)
        courtlistener_case_name = first_cluster_case_name(citation_validation.clusters)
        courtlistener_year = first_cluster_year(citation_validation.clusters)
        exact = assess_case_name_exact_match(
            citation_id=citation.citation_id,
            extracted_case_name=extracted_case_name,
            courtlistener_case_name=courtlistener_case_name,
        )
        if exact.status != CaseNameAssessmentStatus.NEEDS_ASSESSMENT:
            bundle = await assess_found_citation(
                citation_id=citation.citation_id,
                document_text=validation.text,
                span=citation.span,
                extracted_case_name=extracted_case_name,
                courtlistener_case_name=courtlistener_case_name,
                extracted_year=citation.citation.year,
                courtlistener_year=courtlistener_year,
                session=None,
            )
            assessments.append(bundle.assessment)
            continue
        if max_mellea is not None and len(pending) >= max_mellea:
            continue

        pending.append(
            _PendingMelleaAssessment(
                citation=citation,
                document_text=validation.text,
                extracted_case_name=extracted_case_name,
                courtlistener_case_name=courtlistener_case_name,
                extracted_year=citation.citation.year,
                courtlistener_year=courtlistener_year,
                context=get_extended_span_text(validation.text, citation.span),
                mellea_call=len(pending) + 1,
            )
        )

    if pending:
        session = start_mellea_session_from_env()
        limit = mellea_concurrency if mellea_concurrency is not None else len(pending)
        semaphore = asyncio.Semaphore(max(1, limit))
        mellea_results = await asyncio.gather(
            *[
                _assess_pending_mellea_citation(
                    session,
                    job,
                    semaphore=semaphore,
                    on_mellea_call=on_mellea_call,
                    on_mellea_done=on_mellea_done,
                )
                for job in pending
            ]
        )
        for bundle in mellea_results:
            assessments.append(bundle.assessment)
            if bundle.modified_citation is not None:
                modified_citations.append(bundle.modified_citation)
            if bundle.reassessment is not None:
                reassessments.append(bundle.reassessment)

    return DocumentAssessment(
        preprocessed=validation.preprocessed,
        citations=validation.citations,
        validations=validation.validations,
        assessments=tuple(assessments),
        modified_citations=tuple(modified_citations),
        reassessments=tuple(reassessments),
    )


async def _assess_pending_mellea_citation(
    session,
    job: _PendingMelleaAssessment,
    *,
    semaphore: asyncio.Semaphore,
    on_mellea_call: Callable[[MelleaCallContext], None] | None,
    on_mellea_done: Callable[[MelleaCallContext, CitationAssessment], None] | None,
) -> CitationAssessmentBundle:
    call_context = MelleaCallContext(
        mellea_call=job.mellea_call,
        citation_id=job.citation.citation_id,
        matched_text=job.citation.matched_text,
        extracted_case_name=job.extracted_case_name,
        courtlistener_case_name=job.courtlistener_case_name,
        context=job.context,
    )
    async with semaphore:
        if on_mellea_call is not None:
            on_mellea_call(call_context)
        bundle = await assess_found_citation(
            citation_id=job.citation.citation_id,
            document_text=job.document_text,
            span=job.citation.span,
            extracted_case_name=job.extracted_case_name,
            courtlistener_case_name=job.courtlistener_case_name,
            extracted_year=job.extracted_year,
            courtlistener_year=job.courtlistener_year,
            session=session.clone(),
        )
    if on_mellea_done is not None:
        on_mellea_done(call_context, bundle.assessment)
    return bundle
