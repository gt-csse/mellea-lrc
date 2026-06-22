"""Document-level assessment pipeline."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING

from mellea_lrc.assessment.case_name import assess_case_name_exact_match, build_extracted_case_name
from mellea_lrc.assessment.citation import assess_year_exact_match
from mellea_lrc.assessment.context import find_text_span_near_full_span, get_extended_span_text
from mellea_lrc.assessment.types import (
    CaseNameAssessmentStatus,
    CitationAssessment,
    DocumentAssessment,
    ModifiedExtractedCitation,
    YearAssessment,
)
from mellea_lrc.core.citations import FullCaseCitation
from mellea_lrc.extraction.types import ExtractedCitation
from mellea_lrc.llm import start_mellea_session_from_env
from mellea_lrc.validation.types import ValidationStatus

if TYPE_CHECKING:
    from collections.abc import Callable

    from mellea_lrc.validation.types import CitationValidation, DocumentValidation


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
    year_assess: YearAssessment
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
    """Assess a validated document with Mellea-backed case-name checks.

    The interface boundary is intentionally simple:
    :class:`DocumentValidation` in, :class:`DocumentAssessment` out.
    Deterministic exact-match and year checks always run. Mellea-backed semantic
    assessment is called only for case names that need it. Each Mellea citation
    uses a cloned session. Use ``mellea_concurrency`` to cap parallel LLM calls.
    """
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
        courtlistener_case_name = _first_cluster_case_name(citation_validation)
        year_assess = assess_year_exact_match(
            citation_id=citation.citation_id,
            extracted_year=citation.citation.year,
            courtlistener_year=_first_cluster_year(citation_validation),
        )
        exact = assess_case_name_exact_match(
            citation_id=citation.citation_id,
            extracted_case_name=extracted_case_name,
            courtlistener_case_name=courtlistener_case_name,
        )
        if exact.status != CaseNameAssessmentStatus.NEEDS_ASSESSMENT:
            assessments.append(
                CitationAssessment(
                    citation_id=citation.citation_id,
                    case_assess=exact,
                    year_assess=year_assess,
                )
            )
            continue
        if max_mellea is not None and len(pending) >= max_mellea:
            continue

        pending.append(
            _PendingMelleaAssessment(
                citation=citation,
                document_text=validation.text,
                extracted_case_name=extracted_case_name,
                courtlistener_case_name=courtlistener_case_name,
                year_assess=year_assess,
                context=get_extended_span_text(validation.text, citation.span),
                mellea_call=len(pending) + 1,
            )
        )

    if pending:
        from mellea_lrc.assessment.mellea import assess_case_name_with_mellea  # noqa: PLC0415

        session = start_mellea_session_from_env()
        limit = mellea_concurrency if mellea_concurrency is not None else len(pending)
        semaphore = asyncio.Semaphore(max(1, limit))
        mellea_results = await asyncio.gather(
            *[
                _assess_pending_mellea_citation(
                    session,
                    job,
                    semaphore=semaphore,
                    assess_case_name_with_mellea=assess_case_name_with_mellea,
                    on_mellea_call=on_mellea_call,
                    on_mellea_done=on_mellea_done,
                )
                for job in pending
            ]
        )
        for citation_assessment, modified, reassessment in mellea_results:
            assessments.append(citation_assessment)
            if modified is not None:
                modified_citations.append(modified)
            if reassessment is not None:
                reassessments.append(reassessment)

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
    assess_case_name_with_mellea,
    on_mellea_call: Callable[[MelleaCallContext], None] | None,
    on_mellea_done: Callable[[MelleaCallContext, CitationAssessment], None] | None,
) -> tuple[CitationAssessment, ModifiedExtractedCitation | None, CitationAssessment | None]:
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
        run = await assess_case_name_with_mellea(
            session.clone(),
            citation_id=job.citation.citation_id,
            extracted_case_name=job.extracted_case_name,
            courtlistener_case_name=job.courtlistener_case_name,
            document_context=job.context,
        )
    modified = None
    if run.modified_citation is not None:
        modified = ModifiedExtractedCitation.from_proposal(
            run.modified_citation,
            citation_id=job.citation.citation_id,
            span=find_text_span_near_full_span(
                job.document_text,
                run.modified_citation.extracted_case_name or "",
                job.citation.span,
            ),
        )
    reassessment = None
    if run.reassessment is not None:
        reassessment = CitationAssessment(
            citation_id=job.citation.citation_id,
            case_assess=run.reassessment,
            year_assess=job.year_assess,
        )
    citation_assessment = CitationAssessment(
        citation_id=job.citation.citation_id,
        case_assess=run.assessment,
        year_assess=job.year_assess,
    )
    if on_mellea_done is not None:
        on_mellea_done(call_context, citation_assessment)
    return (
        citation_assessment,
        modified,
        reassessment,
    )


def _first_cluster_case_name(validation: CitationValidation) -> str | None:
    if not validation.clusters:
        return None
    case_name = validation.clusters[0].get("case_name") or validation.clusters[0].get("caseName")
    return str(case_name) if isinstance(case_name, str) and case_name else None


def _first_cluster_year(validation: CitationValidation) -> str | None:
    if not validation.clusters:
        return None
    date_filed = validation.clusters[0].get("date_filed") or validation.clusters[0].get("dateFiled")
    return str(date_filed)[:4] if isinstance(date_filed, str) and date_filed else None
