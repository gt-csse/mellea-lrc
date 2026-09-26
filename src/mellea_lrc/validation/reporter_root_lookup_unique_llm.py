"""Review unresolved single-candidate reporter lookups in one model call each."""

from __future__ import annotations

from mellea_lrc.model.citations import FullReporterCitation
from mellea_lrc.model.citations.judgments import IdentityVerdict, MatchResult
from mellea_lrc.model.citations.reporter_lookup import (
    ReporterExactLookupOutcome,
    ReporterUniqueReview,
    ReporterUniqueReviewDecision,
)
from mellea_lrc.model.document import Document
from mellea_lrc.model.span import Span
from mellea_lrc.validation._support.reporter_unique_llm import (
    IvrReporterUniqueReviewer,
    ReporterUniqueReviewContext,
    ReporterUniqueReviewer,
    ReporterUniqueReviewOutcome,
)
from mellea_lrc.validation.stage_names import (
    REPORTER_ROOT_LOOKUP,
    REPORTER_ROOT_LOOKUP_UNIQUE_LLM,
    REPORTER_ROOT_SEARCH,
)

STAGE = REPORTER_ROOT_LOOKUP_UNIQUE_LLM


def _append_corrections(
    root: FullReporterCitation,
    source: str,
    corrections: dict[str, Span],
) -> FullReporterCitation:
    """Only append a reading when the model found different source bytes."""
    for field, span in corrections.items():
        prior = getattr(root, field)
        if prior and prior[-1].span == span:
            continue
        if field == "case_name":
            root = root.with_case_name(source, span)
        elif field == "court":
            root = root.with_court(source, span)
        else:
            root = root.with_date(source, span)
    return root


def _append_judgments(
    root: FullReporterCitation,
    decision: ReporterUniqueReviewDecision,
) -> FullReporterCitation:
    """Compare the newest reading of each field with candidate zero."""
    root = root.with_case_name_judgment(
        len(root.case_name) - 1 if root.case_name else None, 0, decision.case_name.result
    )
    root = root.with_court_judgment(len(root.court) - 1 if root.court else None, 0, decision.court.result)
    root = root.with_date_judgment(len(root.date) - 1 if root.date else None, 0, decision.date.result)
    return root


def _identity_verdict(root: FullReporterCitation, decision: ReporterUniqueReviewDecision) -> IdentityVerdict:
    """Compute identity from the three field decisions; never ask for a second model verdict."""
    if any(
        assessment.result is MatchResult.MISMATCH
        for assessment in (decision.case_name, decision.court, decision.date)
    ):
        return IdentityVerdict.WRONG_IDENTITY
    candidate = root.reporter_exact_lookup.response.clusters[0]
    if (
        not root.case_name
        or not (candidate.case_name_full or candidate.case_name)
        or decision.case_name.result is not MatchResult.MATCH
    ):
        return IdentityVerdict.DEFERRED
    # Missing court/date evidence cannot contradict identity. When both sides
    # have evidence but the model remains unsure, send the root to search.
    docket = root.reporter_exact_docket.response if root.reporter_exact_docket else None
    has_candidate_court = bool(
        candidate.court_id or candidate.court or (docket and (docket.court_id or docket.court))
    )
    if root.court and has_candidate_court and decision.court.result is MatchResult.UNDETERMINED:
        return IdentityVerdict.DEFERRED
    if root.date and candidate.date_filed and decision.date.result is MatchResult.UNDETERMINED:
        return IdentityVerdict.DEFERRED
    return IdentityVerdict.CORRECT_IDENTITY


async def reporter_root_lookup_unique_llm(
    document: Document,
    *,
    reviewer: ReporterUniqueReviewer | None = None,
) -> Document:
    """Reread, correct, and judge each routed unique reporter lookup once.

    The saved candidate is reused. The model's correction quotes must ground
    in the filing, and the program derives identity from its field judgments.
    """
    if STAGE in document.stage_runs:
        raise ValueError(f"Stage already completed: {STAGE}")
    if REPORTER_ROOT_LOOKUP not in document.stage_runs:
        raise ValueError("Complete reporter lookup before its unique model review")
    service = reviewer
    for root in tuple(item for item in document.roots if isinstance(item, FullReporterCitation)):
        if not root.identity_judgments or root.identity_judgments[-1].next_stage != STAGE:
            continue
        lookup = root.reporter_exact_lookup
        if (
            lookup is None
            or lookup.outcome is not ReporterExactLookupOutcome.UNIQUE
            or lookup.response is None
            or len(lookup.response.clusters) != 1
        ):
            raise ValueError("Unique model route requires one saved reporter lookup candidate")
        context = ReporterUniqueReviewContext.from_document(document, root)
        if service is None:
            service = IvrReporterUniqueReviewer.from_env()
        result = await service(context)
        outcome = (
            result
            if isinstance(result, ReporterUniqueReviewOutcome)
            else ReporterUniqueReviewOutcome(decision=result)
        )
        recorded = root.record(STAGE)
        decision = outcome.decision
        corrections = context.grounded_corrections(decision) if decision is not None else None
        failure = outcome.failure_reason
        if decision is not None and corrections is None:
            failure = "Proposed field correction could not be grounded in the filing window"
        if decision is not None and (assessment_error := context.assessment_error(decision)) is not None:
            failure = assessment_error
        if decision is None or corrections is None or failure is not None:
            recorded = recorded.with_reporter_unique_review(
                ReporterUniqueReview(
                    node_id=recorded.nodes[-1].id,
                    ivr=outcome.run,
                    failure_reason=failure or "Model review produced no decision",
                )
            )
            recorded = recorded.with_identity_judgment(IdentityVerdict.DEFERRED, REPORTER_ROOT_SEARCH)
        else:
            recorded = recorded.with_reporter_unique_review(
                ReporterUniqueReview(node_id=recorded.nodes[-1].id, decision=decision, ivr=outcome.run)
            )
            recorded = _append_corrections(recorded, document.text, corrections)
            recorded = _append_judgments(recorded, decision)
            verdict = _identity_verdict(recorded, decision)
            recorded = recorded.with_identity_judgment(
                verdict,
                REPORTER_ROOT_SEARCH if verdict is IdentityVerdict.DEFERRED else None,
            )
        document = document.replace_citation(recorded)
    return document.complete(STAGE)
