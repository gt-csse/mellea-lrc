"""Document-level assessment pipeline."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING

from mellea_lrc.assessment.citation.assess import CitationAssessmentBundle, assess_found_citation
from mellea_lrc.assessment.deterministic.case_name import (
    assess_case_name_exact_match,
    build_extracted_case_name,
)
from mellea_lrc.assessment.deterministic.context import get_extended_span_text
from mellea_lrc.assessment.types import (
    AssessmentMetadata,
    AssessmentSkipReason,
    AssessedCitationAssessment,
    AssessedDocument,
    CaseNameAssessmentStatus,
    CitationAssessment,
    CitationAssessmentResult,
    CitationReassessment,
    FailedCitationAssessment,
    ReassessmentSkipReason,
    SkippedCitationAssessment,
    SkippedCitationReassessment,
    WaitingCitationAssessment,
    WaitingCitationReassessment,
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
    """Assess a validated document with Mellea-backed case-name checks."""
    initialized = initialize_assessment(validation)
    validations_by_id = {item.citation_id: item for item in validation.validations}
    assessments_by_id = {item.citation_id: item for item in initialized.assessments}
    reassessments_by_id = {item.citation_id: item for item in initialized.reassessments}
    pending: list[_PendingMelleaAssessment] = []
    mellea_calls = 0
    effective_concurrency: int | None = None

    for citation in validation.citations:
        if not isinstance(assessments_by_id[citation.citation_id], WaitingCitationAssessment):
            continue
        citation_validation = validations_by_id.get(citation.citation_id)
        assert citation_validation is not None
        assert isinstance(citation.citation, FullCaseCitation)

        extracted_case_name = build_extracted_case_name(citation.citation)
        first_match = citation_validation.matches[0] if citation_validation.matches else None
        courtlistener_case_name = first_match.case_name if first_match is not None else None
        courtlistener_year = first_match.year if first_match is not None else None
        exact = assess_case_name_exact_match(
            citation_id=citation.citation_id,
            extracted_case_name=extracted_case_name,
            courtlistener_case_name=courtlistener_case_name,
        )
        if exact.status != CaseNameAssessmentStatus.NEEDS_ASSESSMENT:
            try:
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
            except Exception as exc:
                _record_failure(
                    citation.citation_id,
                    exc,
                    assessments_by_id=assessments_by_id,
                    reassessments_by_id=reassessments_by_id,
                )
            else:
                _record_bundle(
                    citation.citation_id,
                    bundle,
                    assessments_by_id=assessments_by_id,
                    reassessments_by_id=reassessments_by_id,
                )
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
        try:
            session = start_mellea_session_from_env()
        except Exception as exc:
            for job in pending:
                citation_id = job.citation.citation_id
                _record_failure(
                    citation_id,
                    exc,
                    assessments_by_id=assessments_by_id,
                    reassessments_by_id=reassessments_by_id,
                )
        else:
            limit = mellea_concurrency if mellea_concurrency is not None else len(pending)
            effective_concurrency = min(max(1, limit), len(pending))
            mellea_calls = len(pending)
            semaphore = asyncio.Semaphore(effective_concurrency)
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
                ],
                return_exceptions=True,
            )
            for job, outcome in zip(pending, mellea_results, strict=True):
                citation_id = job.citation.citation_id
                if isinstance(outcome, BaseException):
                    _record_failure(
                        citation_id,
                        outcome,
                        assessments_by_id=assessments_by_id,
                        reassessments_by_id=reassessments_by_id,
                    )
                    continue
                _record_bundle(
                    citation_id,
                    outcome,
                    assessments_by_id=assessments_by_id,
                    reassessments_by_id=reassessments_by_id,
                )

    return AssessedDocument(
        source_metadata=validation.source_metadata,
        text=validation.text,
        preprocessing_metadata=validation.preprocessing_metadata,
        citations=validation.citations,
        extraction_metadata=validation.extraction_metadata,
        validations=validation.validations,
        validation_metadata=validation.validation_metadata,
        assessments=tuple(assessments_by_id[item.citation_id] for item in validation.citations),
        reassessments=tuple(reassessments_by_id[item.citation_id] for item in validation.citations),
        assessment_metadata=AssessmentMetadata(
            mellea_calls=mellea_calls,
            mellea_concurrency=effective_concurrency,
        ),
    )


def initialize_assessment(validation: ValidatedDocument) -> AssessedDocument:
    """Create one waiting or skipped assessment record per citation."""
    validations_by_id = {item.citation_id: item for item in validation.validations}
    assessments: list[CitationAssessment] = []
    reassessments: list[CitationReassessment] = []
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
            reassessments.append(
                SkippedCitationReassessment(
                    citation_id=citation.citation_id,
                    reason=ReassessmentSkipReason.ASSESSMENT_SKIPPED,
                    message="Primary assessment was skipped.",
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
            reassessments.append(
                SkippedCitationReassessment(
                    citation_id=citation.citation_id,
                    reason=ReassessmentSkipReason.ASSESSMENT_SKIPPED,
                    message="Primary assessment was skipped.",
                )
            )
        else:
            assessments.append(WaitingCitationAssessment(citation_id=citation.citation_id))
            reassessments.append(WaitingCitationReassessment(citation_id=citation.citation_id))

    return AssessedDocument(
        source_metadata=validation.source_metadata,
        text=validation.text,
        preprocessing_metadata=validation.preprocessing_metadata,
        citations=validation.citations,
        extraction_metadata=validation.extraction_metadata,
        validations=validation.validations,
        validation_metadata=validation.validation_metadata,
        assessments=tuple(assessments),
        reassessments=tuple(reassessments),
        assessment_metadata=AssessmentMetadata(),
    )


def _record_bundle(
    citation_id: str,
    bundle: CitationAssessmentBundle,
    *,
    assessments_by_id: dict[str, CitationAssessment],
    reassessments_by_id: dict[str, CitationReassessment],
) -> None:
    assessments_by_id[citation_id] = AssessedCitationAssessment(
        citation_id=citation_id,
        result=bundle.assessment,
    )
    reassessments_by_id[citation_id] = bundle.reassessment


def _failed_assessment(citation_id: str, exc: BaseException) -> FailedCitationAssessment:
    return FailedCitationAssessment(
        citation_id=citation_id,
        error=f"{type(exc).__name__}: {exc}",
    )


def _record_failure(
    citation_id: str,
    exc: BaseException,
    *,
    assessments_by_id: dict[str, CitationAssessment],
    reassessments_by_id: dict[str, CitationReassessment],
) -> None:
    assessments_by_id[citation_id] = _failed_assessment(citation_id, exc)
    reassessments_by_id[citation_id] = SkippedCitationReassessment(
        citation_id=citation_id,
        reason=ReassessmentSkipReason.ASSESSMENT_FAILED,
        message="Primary assessment failed before reassessment could be completed.",
    )


async def _assess_pending_mellea_citation(
    session: MelleaSession,
    job: _PendingMelleaAssessment,
    *,
    semaphore: asyncio.Semaphore,
    on_mellea_call: Callable[[MelleaCallContext], None] | None,
    on_mellea_done: Callable[[MelleaCallContext, CitationAssessmentResult], None] | None,
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
