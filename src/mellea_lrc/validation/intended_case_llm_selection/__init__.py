"""Review field-discovered citations without admitting the source locator."""

from __future__ import annotations

from mellea_lrc.llm.profiles import OPENROUTER_LUNA
from mellea_lrc.model.citations.field_body_evidence import IntendedCaseDecision, IntendedCaseReview
from mellea_lrc.model.document import Document
from mellea_lrc.model.span import Span
from mellea_lrc.validation.intended_case_llm_selection.reviewer import (
    IntendedCaseContext,
    IntendedCaseOutcome,
    IntendedCaseReviewer,
    IvrIntendedCaseReviewer,
)

STAGE = "27_intended_case_llm_selection"
MODEL_PROFILE = OPENROUTER_LUNA
NEXT_STAGE_WITH_CANDIDATE = "intended_case_resolution"
NEXT_STAGE_WITHOUT_CANDIDATE = "open_web_search"
NEXT_STAGE_ON_FAILURE = "intended_case_review_retry"


async def intended_case_llm_selection(
    document: Document, *, reviewer: IntendedCaseReviewer | None = None
) -> Document:
    """Save a possible intended authority, leaving identity judgments unchanged."""
    if STAGE in document.stage_runs:
        raise ValueError(f"Stage already completed: {STAGE}")
    if "23_locator_body_llm_judgment" not in document.stage_runs:
        raise ValueError("Complete locator-body review before field-body review")
    roots = tuple(root for root in document.roots if root.next_stage == "case_name_body_discovery")
    if roots and not any(root.field_body_searches for root in roots):
        raise ValueError("Complete a field-body provider search before review")
    service = reviewer
    for root in roots:
        context = IntendedCaseContext.from_document(document, root)
        recorded = root.record(STAGE)
        if not context.evidence:
            outcome = IntendedCaseOutcome(
                decision=IntendedCaseDecision(
                    source=None,
                    evidence_index=None,
                    citation_quote=None,
                    case_name=None,
                    locator=None,
                    court=None,
                    date=None,
                    confidence=None,
                    reason="No independent case-name-anchored citation was found.",
                )
            )
        else:
            if service is None:
                service = IvrIntendedCaseReviewer.from_profile(MODEL_PROFILE)
            result = await service(context)
            outcome = result if isinstance(result, IntendedCaseOutcome) else IntendedCaseOutcome(result)
        decision = outcome.decision
        failure = outcome.failure_reason
        if outcome.run is not None and not outcome.run.success:
            failure = failure or outcome.run.failure_reason or "Intended-case IVR failed"
        if decision is not None and (error := context.validation_error(decision)) is not None:
            failure = error
        if failure is not None or decision is None:
            recorded = recorded.with_intended_case_review(
                IntendedCaseReview(
                    node_id=recorded.nodes[-1].id,
                    ivr=outcome.run,
                    failure_reason=failure or "Model review produced no decision",
                )
            )
            recorded = recorded.with_route(NEXT_STAGE_ON_FAILURE)
        elif decision.source is None:
            recorded = recorded.with_intended_case_review(
                IntendedCaseReview(node_id=recorded.nodes[-1].id, decision=decision, ivr=outcome.run)
            )
            recorded = recorded.with_route(NEXT_STAGE_WITHOUT_CANDIDATE)
        else:
            grounded = context.grounded_quote(decision)
            if grounded is None:
                raise ValueError("Validated intended-case quote lost its grounded evidence")
            recorded = recorded.with_intended_case_review(
                IntendedCaseReview(
                    node_id=recorded.nodes[-1].id,
                    decision=decision,
                    grounded_quote=grounded.text,
                    quote_span=Span(grounded.start, grounded.end),
                    quote_similarity=grounded.similarity_percent,
                    ivr=outcome.run,
                )
            )
            recorded = recorded.with_route(NEXT_STAGE_WITH_CANDIDATE)
        document = document.replace_citation(recorded)
    return document.complete(STAGE)
