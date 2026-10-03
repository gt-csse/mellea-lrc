"""Read occurrence-specific filing propositions before opinion support review."""

from __future__ import annotations

from mellea_lrc.llm.profiles import OPENROUTER_LUNA
from mellea_lrc.model.citations.reporter_page_resolution import ReporterPageResolutionOutcome
from mellea_lrc.model.citations.reporter_pinpoint import PropositionDecision, ReporterCitationProposition
from mellea_lrc.model.document import Document
from mellea_lrc.model.span import is_within
from mellea_lrc.validation.reporter_citation_propositions.reviewer import (
    IvrReporterCitationPropositionReviewer,
    ReporterCitationPropositionContext,
    ReporterCitationPropositionOutcome,
    ReporterCitationPropositionReviewer,
)

STAGE = "43_reporter_citation_propositions"
MODEL_PROFILE = OPENROUTER_LUNA
SOURCE_STAGE = "42_reporter_citation_opinion_review"


async def read_reporter_citation_propositions(
    document: Document, *, reviewer: ReporterCitationPropositionReviewer | None = None
) -> Document:
    """Keep the filing's attributed use distinct from opinion and page judgments."""
    if STAGE in document.stage_runs:
        raise ValueError(f"Stage already completed: {STAGE}")
    if SOURCE_STAGE not in document.stage_runs:
        raise ValueError("Complete reporter opinion selection before reading citation propositions")
    service = reviewer
    for citation in document.citations:
        if (
            not citation.reporter_page_resolutions
            or citation.reporter_page_resolutions[-1].outcome is ReporterPageResolutionOutcome.NO_PIN
        ):
            continue
        context = ReporterCitationPropositionContext.from_document(document, citation)
        # TODO: Consume a citation-level TOA tag propagated from typed serialized
        # preprocessing components. Until then, known index_spans short-circuit
        # reading here; an untagged index can still yield an empty proposition.
        # Both paths become NO_PROPOSITION and skip later pinpoint review.
        if is_within(citation.site_span, document.index_spans):
            outcome = ReporterCitationPropositionOutcome(
                PropositionDecision(
                    quotes=(),
                    reason="This occurrence is in the table of authorities and supplies no proposition.",
                )
            )
        else:
            if service is None:
                service = IvrReporterCitationPropositionReviewer.from_profile(MODEL_PROFILE)
            try:
                result = await service(context)
            except Exception as error:
                result = ReporterCitationPropositionOutcome(
                    None, failure_reason=f"{type(error).__name__}: {error}"
                )
            outcome = (
                result
                if isinstance(result, ReporterCitationPropositionOutcome)
                else ReporterCitationPropositionOutcome(result)
            )
        passages = ()
        if outcome.decision is not None:
            if outcome.failure_reason is not None:
                raise ValueError("Proposition reviewer returned both a decision and a failure")
            if (error := context.decision_error(outcome.decision)) is not None:
                raise ValueError(error)
            passages = context.grounded_passages(outcome.decision)
            assert passages is not None
        recorded = citation.record(STAGE)
        record = ReporterCitationProposition(
            node_id=recorded.nodes[-1].id,
            resolution_index=len(citation.reporter_page_resolutions) - 1,
            decision=outcome.decision,
            passages=passages,
            ivr=outcome.run,
            failure_reason=outcome.failure_reason
            if outcome.decision is None and outcome.failure_reason is not None
            else "Proposition review produced no decision"
            if outcome.decision is None
            else None,
        )
        document = document.replace_citation(recorded.with_reporter_proposition(record))
    return document.complete(STAGE)
