"""Document-level assessment pipeline.

Found and ambiguous citations share one per-candidate assessment path: a "found"
citation has a single candidate (its match), an "ambiguous" citation has one per
returned cluster. Each candidate is delegated to :func:`assess_found_citation`;
the results are then wrapped — a single ``AssessedCitationAssessment`` for found,
or an ``AmbiguousCitationAssessment`` collecting one ``CandidateAssessment`` per
cluster. Ambiguous lookups with more than ``MAX_AMBIGUOUS_CANDIDATES`` clusters
fail fast (gated) rather than fan out an unbounded number of Mellea calls.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

from mellea_lrc.assessment.citation import assess_found_citation
from mellea_lrc.assessment.context import DocumentTextWindow
from mellea_lrc.assessment.fields.case_name import (
    assess_case_name_exact_match,
    build_extracted_case_name,
)
from mellea_lrc.assessment.types import (
    AmbiguousCitationAssessment,
    AssessmentMetadata,
    AssessmentSkipReason,
    AssessedCitationAssessment,
    AssessedDocument,
    CandidateAssessment,
    CitationAssessment,
    CitationAssessmentResult,
    FailedCitationAssessment,
    SkippedCitationAssessment,
    WaitingCitationAssessment,
)
from mellea_lrc.core.citations import FullCaseCitation
from mellea_lrc.llm import start_mellea_session_from_env
from mellea_lrc.retrieval.types import (
    AmbiguousCitationRetrieval,
    FoundCitationRetrieval,
    RetrievalStatus,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from mellea import MelleaSession

    from mellea_lrc.extraction.types import ExtractedCitation
    from mellea_lrc.retrieval.types import RetrievedDocument

# Ambiguous lookups with more clusters than this skip per-candidate enumeration.
MAX_AMBIGUOUS_CANDIDATES = 5

_ELIGIBLE_RETRIEVAL_STATUSES = frozenset({RetrievalStatus.FOUND, RetrievalStatus.AMBIGUOUS})


@dataclass(frozen=True, slots=True)
class MelleaCallContext:
    """Document context emitted immediately before one semantic assessment call."""

    citation_id: str
    matched_text: str
    extracted_case_name: str | None
    courtlistener_case_name: str | None
    context: str


@dataclass(frozen=True, slots=True)
class _AssessmentJob:
    """One (citation, candidate) unit assessed via the found-branch path.

    ``candidate_index`` is ``None`` for a found citation's single candidate and
    ``0..n`` for each ambiguous candidate. ``candidate_id`` links the result
    back to retrieval without duplicating the retrieved record.
    """

    citation: ExtractedCitation
    candidate_id: str
    candidate_index: int | None
    document_text: str
    extracted_case_name: str | None
    courtlistener_case_name: str | None
    courtlistener_year: str | None
    courtlistener_court_id: str | None
    context: DocumentTextWindow

    @property
    def key(self) -> tuple[str, int | None]:
        return (self.citation.citation_id, self.candidate_index)


def run_assessment(
    retrieval: RetrievedDocument,
    *,
    mellea_concurrency: int | None = None,
    on_mellea_call: Callable[[MelleaCallContext], None] | None = None,
    on_mellea_done: Callable[[MelleaCallContext, CitationAssessmentResult], None] | None = None,
) -> AssessedDocument:
    """Assess a retrieved document synchronously."""
    return asyncio.run(
        run_assessment_async(
            retrieval,
            mellea_concurrency=mellea_concurrency,
            on_mellea_call=on_mellea_call,
            on_mellea_done=on_mellea_done,
        )
    )


async def run_assessment_async(
    retrieval: RetrievedDocument,
    *,
    mellea_concurrency: int | None = None,
    on_mellea_call: Callable[[MelleaCallContext], None] | None = None,
    on_mellea_done: Callable[[MelleaCallContext, CitationAssessmentResult], None] | None = None,
) -> AssessedDocument:
    """Assess a retrieved document with bounded Mellea concurrency."""
    initialized = initialize_assessment(retrieval)
    retrievals_by_id = {item.citation_id: item for item in retrieval.retrievals}
    assessments_by_id = {item.citation_id: item for item in initialized.assessments}
    outcomes: dict[tuple[str, int | None], CitationAssessmentResult | BaseException] = {}
    jobs: list[_AssessmentJob] = []

    for citation in retrieval.citations:
        if not isinstance(assessments_by_id[citation.citation_id], WaitingCitationAssessment):
            continue
        citation_retrieval = retrievals_by_id[citation.citation_id]
        assert isinstance(citation.citation, FullCaseCitation)
        gate = _candidate_jobs(citation, citation_retrieval, retrieval.text)
        if gate.direct is not None:
            assessments_by_id[citation.citation_id] = gate.direct
            continue
        jobs.extend(gate.jobs)

    # Exact case-name candidates need no Mellea session; run them immediately.
    pending: list[_AssessmentJob] = []
    for job in jobs:
        if _is_exact_case_name(job):
            outcomes[job.key] = await _run_job(job, session=None)
        else:
            pending.append(job)

    effective_concurrency = await _run_pending(
        pending,
        outcomes,
        mellea_concurrency=mellea_concurrency,
        on_mellea_call=on_mellea_call,
        on_mellea_done=on_mellea_done,
    )

    _assemble(retrieval, assessments_by_id, retrievals_by_id, jobs, outcomes)

    return AssessedDocument(
        source_metadata=retrieval.source_metadata,
        text=retrieval.text,
        preprocessing_metadata=retrieval.preprocessing_metadata,
        citations=retrieval.citations,
        extraction_metadata=retrieval.extraction_metadata,
        retrievals=retrieval.retrievals,
        retrieval_metadata=retrieval.retrieval_metadata,
        assessments=tuple(assessments_by_id[item.citation_id] for item in retrieval.citations),
        assessment_metadata=AssessmentMetadata(mellea_concurrency=effective_concurrency),
    )


@dataclass(frozen=True, slots=True)
class _CandidateJobs:
    """Per-citation plan: candidate jobs to run, or a ``direct`` result to emit.

    ``direct`` short-circuits enumeration — the >5 gate and the (malformed)
    empty-candidate case both resolve to an ``AmbiguousCitationAssessment``
    without fanning out any per-candidate work.
    """

    jobs: tuple[_AssessmentJob, ...]
    direct: AmbiguousCitationAssessment | None = None


def _candidate_jobs(
    citation: ExtractedCitation,
    citation_retrieval: object,
    document_text: str,
) -> _CandidateJobs:
    assert isinstance(citation.citation, FullCaseCitation)
    extracted_case_name = build_extracted_case_name(citation.citation)
    context = DocumentTextWindow.around(document_text, citation.span)

    def job(
        candidate_id: str,
        candidate_index: int | None,
        cl_case_name: str | None,
        cl_year: str | None,
        cl_court_id: str | None,
    ) -> _AssessmentJob:
        return _AssessmentJob(
            citation=citation,
            candidate_id=candidate_id,
            candidate_index=candidate_index,
            document_text=document_text,
            extracted_case_name=extracted_case_name,
            courtlistener_case_name=cl_case_name,
            courtlistener_year=cl_year,
            courtlistener_court_id=cl_court_id,
            context=context,
        )

    if isinstance(citation_retrieval, FoundCitationRetrieval):
        candidate = citation_retrieval.candidate
        record = candidate.record
        return _CandidateJobs(
            jobs=(
                job(
                    candidate.candidate_id,
                    None,
                    record.case_name,
                    record.year,
                    candidate.court_resolution.courtlistener_court_id,
                ),
            )
        )

    if isinstance(citation_retrieval, AmbiguousCitationRetrieval):
        candidates = citation_retrieval.candidates
        if not candidates:
            return _CandidateJobs(
                jobs=(),
                direct=AmbiguousCitationAssessment(
                    citation_id=citation.citation_id,
                    candidates=(),
                    message="Ambiguous lookup returned no candidates.",
                ),
            )
        if len(candidates) > MAX_AMBIGUOUS_CANDIDATES:
            return _CandidateJobs(
                jobs=(),
                direct=AmbiguousCitationAssessment(
                    citation_id=citation.citation_id,
                    candidates=(),
                    gated=True,
                    message=(
                        f"{len(candidates)} candidates exceed the "
                        f"{MAX_AMBIGUOUS_CANDIDATES}-candidate enumeration limit."
                    ),
                ),
            )
        return _CandidateJobs(
            jobs=tuple(
                job(
                    candidate.candidate_id,
                    index,
                    candidate.record.case_name,
                    candidate.record.year,
                    candidate.court_resolution.courtlistener_court_id,
                )
                for index, candidate in enumerate(candidates)
            )
        )

    return _CandidateJobs(jobs=())


def _is_exact_case_name(job: _AssessmentJob) -> bool:
    return (
        assess_case_name_exact_match(
            extracted_case_name=job.extracted_case_name,
            courtlistener_case_name=job.courtlistener_case_name,
        )
        is not None
    )


async def _run_job(
    job: _AssessmentJob,
    *,
    session: MelleaSession | None,
) -> CitationAssessmentResult | BaseException:
    assert isinstance(job.citation.citation, FullCaseCitation)
    try:
        return await assess_found_citation(
            document_text=job.document_text,
            span=job.citation.span,
            extracted_case_name=job.extracted_case_name,
            courtlistener_case_name=job.courtlistener_case_name,
            extracted_year=job.citation.citation.year,
            courtlistener_year=job.courtlistener_year,
            extracted_court=job.citation.citation.court,
            reporter=job.citation.citation.reporter,
            courtlistener_court_id=job.courtlistener_court_id,
            session=session.clone() if session is not None else None,
        )
    except Exception as exc:  # noqa: BLE001 - surfaced per candidate/citation downstream
        return exc


async def _run_pending(
    pending: list[_AssessmentJob],
    outcomes: dict[tuple[str, int | None], CitationAssessmentResult | BaseException],
    *,
    mellea_concurrency: int | None,
    on_mellea_call: Callable[[MelleaCallContext], None] | None,
    on_mellea_done: Callable[[MelleaCallContext, CitationAssessmentResult], None] | None,
) -> int | None:
    if not pending:
        return None
    try:
        session = start_mellea_session_from_env()
    except Exception as exc:  # noqa: BLE001 - session failure fails each pending candidate
        for job in pending:
            outcomes[job.key] = exc
        return None

    limit = mellea_concurrency if mellea_concurrency is not None else len(pending)
    effective_concurrency = min(max(1, limit), len(pending))
    semaphore = asyncio.Semaphore(effective_concurrency)
    results = await asyncio.gather(
        *[
            _assess_pending_job(
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
    for job, outcome in zip(pending, results, strict=True):
        outcomes[job.key] = outcome
    return effective_concurrency


def _assemble(
    retrieval: RetrievedDocument,
    assessments_by_id: dict[str, CitationAssessment],
    retrievals_by_id: dict[str, object],
    jobs: list[_AssessmentJob],
    outcomes: dict[tuple[str, int | None], CitationAssessmentResult | BaseException],
) -> None:
    by_citation: dict[str, list[_AssessmentJob]] = defaultdict(list)
    for job in jobs:
        by_citation[job.citation.citation_id].append(job)

    for citation_id, citation_jobs in by_citation.items():
        is_found = isinstance(retrievals_by_id.get(citation_id), FoundCitationRetrieval)
        errors = [outcomes[job.key] for job in citation_jobs if isinstance(outcomes[job.key], BaseException)]
        if errors:
            assessments_by_id[citation_id] = _failed_assessment(citation_id, errors[0])
        elif is_found:
            outcome = outcomes[citation_jobs[0].key]
            assert not isinstance(outcome, BaseException)
            assessments_by_id[citation_id] = AssessedCitationAssessment(
                citation_id=citation_id,
                candidate_id=citation_jobs[0].candidate_id,
                result=outcome,
            )
        else:
            ordered = sorted(citation_jobs, key=lambda job: job.candidate_index or 0)
            candidates = tuple(
                CandidateAssessment(
                    candidate_id=job.candidate_id,
                    result=outcomes[job.key],  # type: ignore[arg-type]
                )
                for job in ordered
            )
            assessments_by_id[citation_id] = AmbiguousCitationAssessment(
                citation_id=citation_id,
                candidates=candidates,
            )


def initialize_assessment(retrieval: RetrievedDocument) -> AssessedDocument:
    """Create one waiting or skipped assessment record per citation."""
    retrievals_by_id = {item.citation_id: item for item in retrieval.retrievals}
    assessments: list[CitationAssessment] = []
    for citation in retrieval.citations:
        citation_retrieval = retrievals_by_id[citation.citation_id]
        if not isinstance(citation.citation, FullCaseCitation):
            assessments.append(
                SkippedCitationAssessment(
                    citation_id=citation.citation_id,
                    reason=AssessmentSkipReason.UNSUPPORTED_CITATION_KIND,
                    message=f"Citation kind {citation.citation.kind.value} is not assessed.",
                )
            )
        elif citation_retrieval.status not in _ELIGIBLE_RETRIEVAL_STATUSES:
            assessments.append(
                SkippedCitationAssessment(
                    citation_id=citation.citation_id,
                    reason=AssessmentSkipReason.RETRIEVAL_NOT_ELIGIBLE,
                    message=f"Retrieval status {citation_retrieval.status.value} is not eligible.",
                )
            )
        else:
            assessments.append(WaitingCitationAssessment(citation_id=citation.citation_id))

    return AssessedDocument(
        source_metadata=retrieval.source_metadata,
        text=retrieval.text,
        preprocessing_metadata=retrieval.preprocessing_metadata,
        citations=retrieval.citations,
        extraction_metadata=retrieval.extraction_metadata,
        retrievals=retrieval.retrievals,
        retrieval_metadata=retrieval.retrieval_metadata,
        assessments=tuple(assessments),
        assessment_metadata=AssessmentMetadata(),
    )


def _failed_assessment(citation_id: str, exc: BaseException) -> FailedCitationAssessment:
    return FailedCitationAssessment(
        citation_id=citation_id,
        error=f"{type(exc).__name__}: {exc}",
    )


async def _assess_pending_job(
    session: MelleaSession,
    job: _AssessmentJob,
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
    assert isinstance(job.citation.citation, FullCaseCitation)
    async with semaphore:
        if on_mellea_call is not None:
            on_mellea_call(call_context)
        result = await assess_found_citation(
            document_text=job.document_text,
            span=job.citation.span,
            extracted_case_name=job.extracted_case_name,
            courtlistener_case_name=job.courtlistener_case_name,
            extracted_year=job.citation.citation.year,
            courtlistener_year=job.courtlistener_year,
            extracted_court=job.citation.citation.court,
            reporter=job.citation.citation.reporter,
            courtlistener_court_id=job.courtlistener_court_id,
            session=session.clone(),
        )
    if on_mellea_done is not None:
        on_mellea_done(call_context, result)
    return result
