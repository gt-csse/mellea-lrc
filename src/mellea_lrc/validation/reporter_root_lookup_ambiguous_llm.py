"""Select and judge unresolved bounded reporter lookup candidates in one model call."""

from __future__ import annotations

from mellea_lrc.model.citations import FullReporterCitation
from mellea_lrc.model.citations.judgments import IdentityVerdict
from mellea_lrc.model.citations.reporter_lookup import (
    ReporterAmbiguousReview,
    ReporterExactAmbiguityOutcome,
    ReporterExactLookupOutcome,
)
from mellea_lrc.model.document import Document
from mellea_lrc.validation._support.reporter_ambiguous_llm import (
    IvrReporterAmbiguousReviewer,
    ReporterAmbiguousReviewContext,
    ReporterAmbiguousReviewer,
    ReporterAmbiguousReviewOutcome,
)
from mellea_lrc.validation._support.reporter_review_fields import (
    append_corrections,
    append_field_judgments,
    identity_verdict,
)
from mellea_lrc.validation.stage_names import (
    REPORTER_ROOT_LOOKUP_AMBIGUOUS,
    REPORTER_ROOT_LOOKUP_AMBIGUOUS_LLM,
    REPORTER_ROOT_SEARCH,
)

STAGE = REPORTER_ROOT_LOOKUP_AMBIGUOUS_LLM


async def reporter_root_lookup_ambiguous_llm(
    document: Document,
    *,
    reviewer: ReporterAmbiguousReviewer | None = None,
) -> Document:
    """Review every exact-lookup ambiguity routed here and retain all candidates.

    One selected representative receives independent field judgments. A null
    selection or failed review can still be searched later. Grounded filing
    corrections from a valid null selection remain useful to that later stage.
    """
    if STAGE in document.stage_runs:
        raise ValueError(f"Stage already completed: {STAGE}")
    if REPORTER_ROOT_LOOKUP_AMBIGUOUS not in document.stage_runs:
        raise ValueError("Complete rule-only reporter ambiguity review before model choice")
    service = reviewer
    for root in tuple(item for item in document.roots if isinstance(item, FullReporterCitation)):
        if not root.identity_judgments or root.identity_judgments[-1].next_stage != STAGE:
            continue
        lookup = root.reporter_exact_lookup
        resolution = root.reporter_exact_ambiguity_resolution
        if (
            lookup is None
            or lookup.outcome is not ReporterExactLookupOutcome.AMBIGUOUS
            or lookup.response is None
            or not 2 <= len(lookup.response.clusters) < 20
            or resolution is None
            or resolution.outcome is not ReporterExactAmbiguityOutcome.NO_UNIQUE_RULE_MATCH
        ):
            raise ValueError("Ambiguous model route requires an unresolved bounded candidate list")
        context = ReporterAmbiguousReviewContext.from_document(document, root)
        if service is None:
            service = IvrReporterAmbiguousReviewer.from_env()
        result = await service(context)
        outcome = (
            result
            if isinstance(result, ReporterAmbiguousReviewOutcome)
            else ReporterAmbiguousReviewOutcome(decision=result)
        )
        recorded = root.record(STAGE)
        decision = outcome.decision
        corrections = context.grounded_corrections(decision) if decision is not None else None
        failure = outcome.failure_reason
        if decision is not None and corrections is None:
            failure = "Replacement intent or source quote is inconsistent or ungrounded"
        if decision is not None and (assessment_error := context.choice_error(decision)) is not None:
            failure = assessment_error
        if decision is None or corrections is None or failure is not None:
            recorded = recorded.with_reporter_ambiguous_review(
                ReporterAmbiguousReview(
                    node_id=recorded.nodes[-1].id,
                    ivr=outcome.run,
                    failure_reason=failure or "Model review produced no decision",
                )
            )
            recorded = recorded.with_identity_judgment(IdentityVerdict.DEFERRED, REPORTER_ROOT_SEARCH)
        else:
            recorded = recorded.with_reporter_ambiguous_review(
                ReporterAmbiguousReview(node_id=recorded.nodes[-1].id, decision=decision, ivr=outcome.run)
            )
            recorded = append_corrections(recorded, document.text, corrections, decision)
            selected = decision.selected_candidate_index
            if selected is None:
                recorded = recorded.with_identity_judgment(IdentityVerdict.DEFERRED, REPORTER_ROOT_SEARCH)
            else:
                recorded = append_field_judgments(recorded, decision, selected)
                verdict = identity_verdict(recorded, decision, selected, context.selected_docket(selected))
                recorded = recorded.with_identity_judgment(
                    verdict,
                    REPORTER_ROOT_SEARCH if verdict is IdentityVerdict.DEFERRED else None,
                )
        document = document.replace_citation(recorded)
    return document.complete(STAGE)
