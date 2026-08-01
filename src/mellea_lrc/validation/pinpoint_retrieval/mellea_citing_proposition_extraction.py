"""Mellea extraction of the proposition attributed to one citation."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Literal

from mellea.core import ValidationResult
from mellea.stdlib.requirements import req
from mellea.stdlib.sampling import MultiTurnStrategy
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from mellea_lrc.core.spans import Span
from mellea_lrc.llm import (
    InstructIvrSpec,
    llm_api_config_from_env,
    run_instruct_ivr,
    start_mellea_session_from_env,
)
from mellea_lrc.validation.pinpoint_retrieval.evidence_quote import (
    ResolvedEvidenceQuote,
    resolve_evidence_quote,
)
from mellea_lrc.validation.types import (
    MelleaCitingPropositionExtractionNode,
    MelleaCitingPropositionExtractionOutcome,
    ReporterPageRetrievalNode,
    ReporterPageRetrievalOutcome,
    ValidationNodeStatus,
)

if TYPE_CHECKING:
    from mellea import MelleaSession
    from mellea.core.base import Context

    from mellea_lrc.validation.types import CitationValidation

CONTEXT_WINDOW_CHARS = 1_200
CONTEXT_BEFORE_CITATION_CHARS = 900
MAX_TOKENS = 256
MAX_REPAIR_TURNS = 2
INSTRUCTION = """
Identify the single proposition that citing_context attributes to target_citation.
The proposition may be a clause in the citation's sentence, a nearby explanatory
parenthetical, or the immediately preceding sentence. Select only text genuinely
tied to target_citation, not a neighboring citation.

Prefer, in order: an explanatory parenthetical attached to target_citation; a
proposition in the same clause or sentence; then the immediately preceding
sentence. This ordering prevents a nearby citation from donating its proposition.

Return "identified" and copy one contiguous proposition_quote from
citing_context when a proposition is discernible. Exclude the citation text and
citation signal from the quote when possible. Return "inconclusive" with a null
quote when the context does not state a proposition clearly enough to test.

The program locates proposition_quote in the original document. Copy it exactly:
do not paraphrase, shorten, normalize, or change any word. If you cannot copy a
distinctive contiguous quote, return inconclusive. Give a concise explanation of
why the selected text is the proposition attributed to the citation; do not
provide hidden chain-of-thought.

Example:
citing_context: A reviewing court considers the issue anew. See Sample v. Case,
10 F.4th 20, 25 (1st Cir. 2024) (applying de novo review).
target_citation: Sample v. Case, 10 F.4th 20, 25 (1st Cir. 2024)
classification: identified
proposition_quote: applying de novo review

target_citation:
{{target_citation}}
""".strip()


class _CitingPropositionProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    classification: Literal["identified", "inconclusive"]
    reasoning: str = Field(min_length=1)
    proposition_quote: str | None = None


async def run_mellea_citing_proposition_extraction(
    validation: CitationValidation,
    *,
    trigger: ReporterPageRetrievalNode,
    document_text: str,
    session: MelleaSession | None = None,
) -> MelleaCitingPropositionExtractionNode:
    """Extract one grounded citing proposition inside a bounded source window."""
    context_span = citing_context_span(document_text, validation.citation.span)
    if trigger.outcome is not ReporterPageRetrievalOutcome.FOUND:
        return _node(
            trigger,
            context_span=context_span,
            status=ValidationNodeStatus.SKIPPED,
            outcome=MelleaCitingPropositionExtractionOutcome.UNAVAILABLE,
            status_message=(
                "Skipped citing-proposition extraction because reporter-page evidence is unavailable."
            ),
            outcome_message="No pinpoint inference will run without retrieved reporter-page evidence.",
        )

    citing_context = document_text[context_span.start : context_span.end]
    citation_span = validation.citation.span
    target_citation = document_text[citation_span.start : citation_span.end]
    try:
        spec = InstructIvrSpec(
            description=INSTRUCTION,
            grounding_context={"citing_context": citing_context},
            user_variables={"target_citation": target_citation},
            output_format=_CitingPropositionProposal,
            requirements=[
                req("Return a valid citing-proposition object.", validation_fn=_validate_schema),
                req(
                    "An identified proposition must resolve uniquely inside citing_context.",
                    validation_fn=lambda ctx: _validate_grounding(ctx, citing_context),
                ),
            ],
        )
        result = await run_instruct_ivr(
            session or start_mellea_session_from_env(),
            spec,
            strategy=MultiTurnStrategy(loop_budget=MAX_REPAIR_TURNS),
            model_options=llm_api_config_from_env(os.environ).mellea_call_options(max_tokens=MAX_TOKENS),
        )
        if not result.success:
            return _node(
                trigger,
                context_span=context_span,
                status=ValidationNodeStatus.FAILED,
                outcome=MelleaCitingPropositionExtractionOutcome.FAILED,
                status_message="Citing-proposition extraction exhausted its repair attempts.",
                outcome_message="No grounded citing proposition is available.",
                error="Mellea citing-proposition extraction exhausted its repair budget",
            )
        proposal = _parse(result.result.value)
        resolved = (
            resolve_evidence_quote(citing_context, proposal.proposition_quote)
            if proposal.proposition_quote is not None
            else None
        )
    except Exception as exc:
        return _node(
            trigger,
            context_span=context_span,
            status=ValidationNodeStatus.FAILED,
            outcome=MelleaCitingPropositionExtractionOutcome.FAILED,
            status_message="Citing-proposition extraction failed during execution.",
            outcome_message="No grounded citing proposition is available.",
            error=f"{type(exc).__name__}: {exc}",
        )

    proposition_span = (
        Span(
            context_span.start + resolved.span.start,
            context_span.start + resolved.span.end,
        )
        if resolved is not None
        else None
    )
    return _node(
        trigger,
        context_span=context_span,
        status=ValidationNodeStatus.SUCCEEDED,
        outcome=(
            MelleaCitingPropositionExtractionOutcome.IDENTIFIED
            if proposal.classification == "identified"
            else MelleaCitingPropositionExtractionOutcome.INCONCLUSIVE
        ),
        proposal=proposal,
        resolved=resolved,
        proposition_span=proposition_span,
        status_message="Citing-proposition extraction completed.",
        outcome_message=(
            "Identified a grounded proposition attributed to the citation."
            if proposal.classification == "identified"
            else "The bounded citing context does not state a sufficiently clear proposition."
        ),
    )


def citing_context_span(document_text: str, citation_span: Span) -> Span:
    """Return a bounded 1,200-character window biased before the citation."""
    start = max(0, citation_span.start - CONTEXT_BEFORE_CITATION_CHARS)
    end = min(len(document_text), start + CONTEXT_WINDOW_CHARS)
    if end < citation_span.end:
        end = citation_span.end
        start = max(0, end - CONTEXT_WINDOW_CHARS)
    if end - start < CONTEXT_WINDOW_CHARS:
        start = max(0, end - CONTEXT_WINDOW_CHARS)
    return Span(start, end)


def _parse(value: object) -> _CitingPropositionProposal:
    try:
        return _CitingPropositionProposal.model_validate_json(value)
    except ValidationError as exc:
        msg = f"Invalid Mellea citing-proposition output: {exc}"
        raise ValueError(msg) from exc


def _validate_schema(ctx: Context) -> ValidationResult:
    try:
        _parse(ctx.last_output().value)
    except ValueError as exc:
        return ValidationResult(result=False, reason=str(exc))
    return ValidationResult(result=True)


def _validate_grounding(ctx: Context, citing_context: str) -> ValidationResult:
    proposal = _parse(ctx.last_output().value)
    quote = proposal.proposition_quote
    if proposal.classification == "identified" and (quote is None or not quote.strip()):
        return ValidationResult(
            result=False,
            reason="An identified proposition requires a proposition_quote.",
        )
    if proposal.classification == "inconclusive" and quote is not None:
        return ValidationResult(
            result=False,
            reason="An inconclusive proposition must use a null proposition_quote.",
        )
    if quote is None:
        return ValidationResult(result=True)
    resolved = resolve_evidence_quote(citing_context, quote)
    return ValidationResult(
        result=resolved is not None,
        reason=(
            None
            if resolved is not None
            else "proposition_quote does not resolve uniquely inside citing_context"
        ),
    )


def _node(
    trigger: ReporterPageRetrievalNode,
    *,
    context_span: Span,
    status: ValidationNodeStatus,
    outcome: MelleaCitingPropositionExtractionOutcome,
    proposal: _CitingPropositionProposal | None = None,
    resolved: ResolvedEvidenceQuote | None = None,
    proposition_span: Span | None = None,
    status_message: str | None = None,
    outcome_message: str | None = None,
    error: str | None = None,
) -> MelleaCitingPropositionExtractionNode:
    return MelleaCitingPropositionExtractionNode(
        node_id=f"{trigger.node_id}:mellea_citing_proposition_extraction",
        status=status,
        outcome=outcome,
        context_span=context_span,
        reasoning=proposal.reasoning.strip() if proposal is not None else None,
        proposition=resolved.text if resolved is not None else None,
        proposition_span=proposition_span,
        proposition_match_method=resolved.method if resolved is not None else None,
        proposition_match_score=resolved.score if resolved is not None else None,
        depends_on=(trigger.node_id,),
        status_message=status_message,
        outcome_message=outcome_message,
        error=error,
    )
