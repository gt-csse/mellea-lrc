"""Review field-discovered citations without admitting the source locator."""

from __future__ import annotations

from mellea_lrc.llm.profiles import load_profile
from mellea_lrc.model.citations.field_body_evidence import IntendedCaseDecision, IntendedCaseReview
from mellea_lrc.model.document import Document
from mellea_lrc.parsing.span import Span
from mellea_lrc.validation.review_intended_cases_llm.reviewer import (
    IntendedCaseContext,
    IntendedCaseOutcome,
    IntendedCaseReviewer,
    IvrIntendedCaseReviewer,
)

SUBSTAGE = "validate_roots_citation_identity.discover_intended_cases.review_intended_cases_llm"
NEXT_STAGE_WITH_CANDIDATE = "resolve_intended_cases"
NEXT_STAGE_WITHOUT_CANDIDATE = "search_open_web_roots"
NEXT_STAGE_ON_FAILURE = "retry_intended_case_review"


async def review_intended_cases_llm(
    document: Document, *, reviewer: IntendedCaseReviewer | None = None
) -> Document:
    """Save a possible intended authority, leaving identity judgments unchanged."""
    if SUBSTAGE in document.substage_runs:
        raise ValueError(f"Substage already completed: {SUBSTAGE}")
    if (
        "validate_roots_citation_identity.corroborate_locator_bodies.review_locator_bodies_llm"
        not in document.substage_runs
    ):
        raise ValueError("Complete locator-body review before field-body review")
    roots = tuple(
        root
        for root in document.roots
        if root.next_substage
        == "validate_roots_citation_identity.discover_intended_cases.retrieve_courtlistener_opinions"
    )
    if roots and not any(root.field_body_searches for root in roots):
        raise ValueError("Complete a field-body provider search before review")
    service = reviewer
    for root in roots:
        context = IntendedCaseContext.from_document(document, root)
        recorded = root.record(SUBSTAGE)
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
                service = IvrIntendedCaseReviewer.from_profile(load_profile(SUBSTAGE))
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
    return document.complete_substage(SUBSTAGE)
