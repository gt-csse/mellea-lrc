"""Document-level assessment pipeline."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING

from mellea_lrc.assessment.citation import assess_found_citation
from mellea_lrc.assessment.context import DocumentTextWindow
from mellea_lrc.assessment.fields.case_name import (
    assess_case_name_exact_match,
    build_extracted_case_name,
)
from mellea_lrc.assessment.types import (
    AssessmentMetadata,
    AssessmentSkipReason,
    AssessedCitationAssessment,
    AssessedDocument,
    CitationAssessment,
    CitationAssessmentResult,
    FailedCitationAssessment,
    SkippedCitationAssessment,
    WaitingCitationAssessment,
)
from mellea_lrc.core.citations import FullCaseCitation
from mellea_lrc.llm import start_mellea_session_from_env
from mellea_lrc.validation.types import ValidationStatus

if TYPE_CHECKING:
    from collections.abc import Callable

    from mellea import MelleaSession

    from mellea_lrc.extraction.types import ExtractedCitation
    from mellea_lrc.validation.types import ValidatedDocument


@dataclass(frozen=True, slots=True)
class MelleaCallContext:
    """Document context emitted immediately before one semantic assessment call."""

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
    extracted_court: str | None
    courtlistener_court_id: str | None
    context: DocumentTextWindow


def run_assessment(
    validation: ValidatedDocument,
    *,
    mellea_concurrency: int | None = None,
    on_mellea_call: Callable[[MelleaCallContext], None] | None = None,
    on_mellea_done: Callable[[MelleaCallContext, CitationAssessmentResult], None] | None = None,
) -> AssessedDocument:
    """Assess a validated document synchronously."""
    return asyncio.run(
        run_assessment_async(
            validation,
            mellea_concurrency=mellea_concurrency,
            on_mellea_call=on_mellea_call,
            on_mellea_done=on_mellea_done,
        )
    )


async def run_assessment_async(
    validation: ValidatedDocument,
    *,
    mellea_concurrency: int | None = None,
    on_mellea_call: Callable[[MelleaCallContext], None] | None = None,
    on_mellea_done: Callable[[MelleaCallContext, CitationAssessmentResult], None] | None = None,
) -> AssessedDocument:
    """Assess a validated document with bounded Mellea concurrency."""
    initialized = initialize_assessment(validation)
    validations_by_id = {item.citation_id: item for item in validation.validations}
    assessments_by_id = {item.citation_id: item for item in initialized.assessments}
    pending: list[_PendingMelleaAssessment] = []
    effective_concurrency: int | None = None

    for citation in validation.citations:
        if not isinstance(assessments_by_id[citation.citation_id], WaitingCitationAssessment):
            continue
        citation_validation = validations_by_id[citation.citation_id]
        assert isinstance(citation.citation, FullCaseCitation)
        extracted_case_name = build_extracted_case_name(citation.citation)
        first_match = citation_validation.matches[0] if citation_validation.matches else None
        courtlistener_case_name = first_match.case_name if first_match is not None else None
        courtlistener_year = first_match.year if first_match is not None else None
        courtlistener_court_id = first_match.court_id if first_match is not None else None
        exact = assess_case_name_exact_match(
            extracted_case_name=extracted_case_name,
            courtlistener_case_name=courtlistener_case_name,
        )
        if exact is not None:
            try:
                result = await assess_found_citation(
                    document_text=validation.text,
                    span=citation.span,
                    extracted_case_name=extracted_case_name,
                    courtlistener_case_name=courtlistener_case_name,
                    extracted_year=citation.citation.year,
                    courtlistener_year=courtlistener_year,
                    extracted_court=citation.citation.court,
                    courtlistener_court_id=courtlistener_court_id,
                )
            except Exception as exc:
                assessments_by_id[citation.citation_id] = _failed_assessment(
                    citation.citation_id,
                    exc,
                )
            else:
                _record_result(citation.citation_id, result, assessments_by_id)
            continue

        pending.append(
            _PendingMelleaAssessment(
                citation=citation,
                document_text=validation.text,
                extracted_case_name=extracted_case_name,
                courtlistener_case_name=courtlistener_case_name,
                extracted_year=citation.citation.year,
                courtlistener_year=courtlistener_year,
                extracted_court=citation.citation.court,
                courtlistener_court_id=courtlistener_court_id,
                context=DocumentTextWindow.around(validation.text, citation.span),
            )
        )

    if pending:
        try:
            session = start_mellea_session_from_env()
        except Exception as exc:
            for job in pending:
                assessments_by_id[job.citation.citation_id] = _failed_assessment(
                    job.citation.citation_id,
                    exc,
                )
        else:
            limit = mellea_concurrency if mellea_concurrency is not None else len(pending)
            effective_concurrency = min(max(1, limit), len(pending))
            semaphore = asyncio.Semaphore(effective_concurrency)
            outcomes = await asyncio.gather(
                *[
                    _assess_pending_mellea_citation(
                        session,
                        job,
                        semaphore=semaphore,
                        on_mellea_call=on_mellea_call,
                        on_mellea_done=on_mellea_done,
                    )
                    for job in pending
                ],
                return_exceptions=True,
            )
            for job, outcome in zip(pending, outcomes, strict=True):
                citation_id = job.citation.citation_id
                if isinstance(outcome, BaseException):
                    assessments_by_id[citation_id] = _failed_assessment(citation_id, outcome)
                else:
                    _record_result(citation_id, outcome, assessments_by_id)

    return AssessedDocument(
        source_metadata=validation.source_metadata,
        text=validation.text,
        preprocessing_metadata=validation.preprocessing_metadata,
        citations=validation.citations,
        extraction_metadata=validation.extraction_metadata,
        validations=validation.validations,
        validation_metadata=validation.validation_metadata,
        assessments=tuple(assessments_by_id[item.citation_id] for item in validation.citations),
        assessment_metadata=AssessmentMetadata(mellea_concurrency=effective_concurrency),
    )


def initialize_assessment(validation: ValidatedDocument) -> AssessedDocument:
    """Create one waiting or skipped assessment record per citation."""
    validations_by_id = {item.citation_id: item for item in validation.validations}
    assessments: list[CitationAssessment] = []
    for citation in validation.citations:
        citation_validation = validations_by_id[citation.citation_id]
        if not isinstance(citation.citation, FullCaseCitation):
            assessments.append(
                SkippedCitationAssessment(
                    citation_id=citation.citation_id,
                    reason=AssessmentSkipReason.UNSUPPORTED_CITATION_KIND,
                    message=f"Citation kind {citation.citation.kind.value} is not assessed.",
                )
            )
        elif citation_validation.status != ValidationStatus.FOUND:
            assessments.append(
                SkippedCitationAssessment(
                    citation_id=citation.citation_id,
                    reason=AssessmentSkipReason.VALIDATION_NOT_ELIGIBLE,
                    message=f"Validation status {citation_validation.status.value} is not eligible.",
                )
            )
        else:
            assessments.append(WaitingCitationAssessment(citation_id=citation.citation_id))

    return AssessedDocument(
        source_metadata=validation.source_metadata,
        text=validation.text,
        preprocessing_metadata=validation.preprocessing_metadata,
        citations=validation.citations,
        extraction_metadata=validation.extraction_metadata,
        validations=validation.validations,
        validation_metadata=validation.validation_metadata,
        assessments=tuple(assessments),
        assessment_metadata=AssessmentMetadata(),
    )


def _record_result(
    citation_id: str,
    result: CitationAssessmentResult,
    assessments_by_id: dict[str, CitationAssessment],
) -> None:
    assessments_by_id[citation_id] = AssessedCitationAssessment(
        citation_id=citation_id,
        result=result,
    )


def _failed_assessment(citation_id: str, exc: BaseException) -> FailedCitationAssessment:
    return FailedCitationAssessment(
        citation_id=citation_id,
        error=f"{type(exc).__name__}: {exc}",
    )


async def _assess_pending_mellea_citation(
    session: MelleaSession,
    job: _PendingMelleaAssessment,
    *,
    semaphore: asyncio.Semaphore,
    on_mellea_call: Callable[[MelleaCallContext], None] | None,
    on_mellea_done: Callable[[MelleaCallContext, CitationAssessmentResult], None] | None,
) -> CitationAssessmentResult:
    call_context = MelleaCallContext(
        citation_id=job.citation.citation_id,
        matched_text=job.citation.matched_text,
        extracted_case_name=job.extracted_case_name,
        courtlistener_case_name=job.courtlistener_case_name,
        context=job.context.text,
    )
    async with semaphore:
        if on_mellea_call is not None:
            on_mellea_call(call_context)
        result = await assess_found_citation(
            document_text=job.document_text,
            span=job.citation.span,
            extracted_case_name=job.extracted_case_name,
            courtlistener_case_name=job.courtlistener_case_name,
            extracted_year=job.extracted_year,
            courtlistener_year=job.courtlistener_year,
            extracted_court=job.extracted_court,
            courtlistener_court_id=job.courtlistener_court_id,
            session=session.clone(),
        )
    if on_mellea_done is not None:
        on_mellea_done(call_context, result)
    return result
