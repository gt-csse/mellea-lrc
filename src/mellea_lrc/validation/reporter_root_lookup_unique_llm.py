"""Review unresolved single-candidate reporter lookups in one model call each."""

from __future__ import annotations

from mellea_lrc.model.citations import FullReporterCitation
from mellea_lrc.model.citations.judgments import IdentityVerdict
from mellea_lrc.model.citations.reporter_lookup import (
    ReporterExactLookupOutcome,
    ReporterUniqueReview,
)
from mellea_lrc.model.document import Document
from mellea_lrc.validation._support.reporter_review_fields import (
    append_corrections,
    append_field_judgments,
    identity_verdict,
)
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
            failure = "Replacement intent or source quote is inconsistent or ungrounded"
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
            recorded = append_corrections(recorded, document.text, corrections)
            recorded = append_field_judgments(recorded, decision, 0)
            verdict = identity_verdict(
                recorded,
                decision,
                0,
                recorded.reporter_exact_docket.response if recorded.reporter_exact_docket else None,
            )
            recorded = recorded.with_identity_judgment(
                verdict,
                REPORTER_ROOT_SEARCH if verdict is IdentityVerdict.DEFERRED else None,
            )
        document = document.replace_citation(recorded)
    return document.complete(STAGE)
