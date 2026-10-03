"""Judge locator-anchored citations found in independent document bodies."""

from __future__ import annotations

from mellea_lrc.llm.profiles import OPENROUTER_LUNA
from mellea_lrc.model.citations import FullCitationVariant, FullDocketCitation
from mellea_lrc.model.citations.body_evidence import (
    BodyCorroborationDecision,
    BodyCorroborationReview,
)
from mellea_lrc.model.citations.judgments import IdentityBasis, IdentityVerdict
from mellea_lrc.model.document import Document
from mellea_lrc.model.span import Span
from mellea_lrc.validation.body_search.common import roots_for_body_search
from mellea_lrc.validation.locator_body_llm_judgment.reviewer import (
    BodyCorroborationContext,
    BodyCorroborationOutcome,
    BodyCorroborationReviewer,
    IvrBodyCorroborationReviewer,
)

STAGE = "23_locator_body_llm_judgment"
MODEL_PROFILE = OPENROUTER_LUNA
NEXT_STAGE = "case_name_body_discovery"


def _append_corrections(
    root: FullCitationVariant,
    source: str,
    spans: dict[str, Span],
    decision: BodyCorroborationDecision,
) -> FullCitationVariant:
    """Append only actual source reread differences at the review node."""
    filing = decision.filing
    assert filing is not None
    if isinstance(root, FullDocketCitation) and (span := spans.get("locator")) is not None:
        if span != root.locator[-1].number_span:
            root = root.with_docket_number(source, span)
    name_span = spans.get("case_name")
    if name_span is None and root.case_name and filing.case_name is not None:
        name_span = root.case_name[-1].span
    if name_span is not None and filing.normalized_case_name is not None:
        prior = root.case_name[-1] if root.case_name else None
        if (
            prior is None
            or prior.span != name_span
            or not prior.normalizable
            or prior.get_normalized() != filing.normalized_case_name
        ):
            root = root.with_case_name(source, name_span, normalized=filing.normalized_case_name)
    for field in ("court", "date"):
        span = spans.get(field)
        if span is None:
            continue
        prior = getattr(root, field)
        if prior and prior[-1].span == span:
            continue
        root = root.with_court(source, span) if field == "court" else root.with_date(source, span)
    return root


async def locator_body_llm_judgment(
    document: Document,
    *,
    reviewer: BodyCorroborationReviewer | None = None,
) -> Document:
    """Judge a grounded locator occurrence across provider excerpts.

    Provider search stages are independently resumable. Their records remain
    attached to each citation; this stage owns the sole cross-provider verdict.
    """
    if STAGE in document.stage_runs:
        raise ValueError(f"Stage already completed: {STAGE}")
    roots = roots_for_body_search(document)
    if not roots:
        return document.complete(STAGE)
    if not any(root.body_searches for root in roots):
        raise ValueError("Complete at least one body search before corroboration review")
    service = reviewer
    for root in roots:
        context = BodyCorroborationContext.from_document(document, root)
        recorded = root.record(STAGE)
        if not context.evidence:
            outcome = BodyCorroborationOutcome(
                decision=BodyCorroborationDecision(
                    source=None,
                    evidence_index=None,
                    citation_quote=None,
                    context_quote=None,
                    treatment=None,
                    filing=None,
                    third_party=None,
                    comparisons=None,
                    identity_verdict=None,
                    reason="No fetched third-party body citation is available for comparison.",
                )
            )
        else:
            if service is None:
                service = IvrBodyCorroborationReviewer.from_profile(MODEL_PROFILE)
            result = await service(context)
            outcome = (
                result if isinstance(result, BodyCorroborationOutcome) else BodyCorroborationOutcome(result)
            )
        decision = outcome.decision
        failure = outcome.failure_reason
        if outcome.run is not None and not outcome.run.success:
            failure = failure or outcome.run.failure_reason or "Body corroboration IVR failed"
        if decision is not None and (error := context.validation_error(decision)) is not None:
            failure = error
        if failure is not None or decision is None:
            recorded = recorded.with_body_review(
                BodyCorroborationReview(
                    node_id=recorded.nodes[-1].id,
                    ivr=outcome.run,
                    failure_reason=failure or "Model review produced no decision",
                )
            )
            recorded = recorded.with_route(NEXT_STAGE)
        elif decision.source is None:
            recorded = recorded.with_body_review(
                BodyCorroborationReview(node_id=recorded.nodes[-1].id, decision=decision, ivr=outcome.run)
            )
            recorded = recorded.with_route(NEXT_STAGE)
        else:
            grounded = context.grounded_quote(decision)
            grounded_context = context.grounded_context(decision)
            spans = context.corrected_spans(decision)
            if grounded is None or spans is None:
                raise ValueError("Validated body citation lost its grounded evidence")
            recorded = recorded.with_body_review(
                BodyCorroborationReview(
                    node_id=recorded.nodes[-1].id,
                    decision=decision,
                    grounded_quote=grounded.text,
                    quote_span=Span(grounded.start, grounded.end),
                    quote_similarity=grounded.similarity_percent,
                    grounded_context=grounded_context.text if grounded_context is not None else None,
                    context_span=(
                        Span(grounded_context.start, grounded_context.end)
                        if grounded_context is not None
                        else None
                    ),
                    context_similarity=(
                        grounded_context.similarity_percent if grounded_context is not None else None
                    ),
                    ivr=outcome.run,
                )
            )
            recorded = _append_corrections(recorded, document.text, spans, decision)
            verdict = decision.identity_verdict
            if verdict is None:
                raise ValueError("A selected body citation must issue an identity opinion")
            recorded = recorded.with_identity_judgment(verdict, basis=IdentityBasis.THIRD_PARTY)
            recorded = recorded.with_route(
                NEXT_STAGE
                if verdict in {IdentityVerdict.UNDETERMINED, IdentityVerdict.PARTIALLY_CORROBORATED}
                else None
            )
        document = document.replace_citation(recorded)
    return document.complete(STAGE)
