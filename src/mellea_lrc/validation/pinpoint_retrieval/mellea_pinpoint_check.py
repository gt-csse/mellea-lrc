"""Mellea semantic inference over retrieved reporter-page evidence."""

from __future__ import annotations

import os
import re
from typing import TYPE_CHECKING, Literal

from mellea.core import ValidationResult
from mellea.stdlib.requirements import req
from mellea.stdlib.sampling import MultiTurnStrategy
from pydantic import BaseModel, ConfigDict, Field, ValidationError

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
    MelleaPinpointCheckNode,
    MelleaPinpointCheckOutcome,
    ReporterPageRetrievalNode,
    ReporterPageRetrievalOutcome,
    ValidationNodeStatus,
)

if TYPE_CHECKING:
    from mellea import MelleaSession
    from mellea.core.base import Context

    from mellea_lrc.validation.types import CitationValidation

_ELLIPSIS = re.compile(r"\.\s*\.\s*\.|\u2026")
_QUOTE_MARKS = frozenset("\"'\u201c\u201d\u2018\u2019")

MAX_TOKENS = 384
MAX_REPAIR_TURNS = 2
INSTRUCTION = """
Decide whether cited_reporter_page substantively supports citing_proposition.
The proposition was separately extracted from the citing document. Use only
cited_reporter_page; do not rely on outside knowledge or another part of the
opinion.

Return "supports" only when the page affirmatively contains or substantively
supports the attributed proposition. Return "inconclusive" otherwise, including
when the page is unrelated, incomplete, ambiguous, or appears inconsistent with
the proposition. One retrieved page is not sufficient evidence for this operation
to make a negative judgment about the citation.

For a supports judgment, copy one short, sufficiently distinctive evidence_quote
from cited_reporter_page. It must be a single unbroken run of characters copied
exactly as it appears there: never stand an ellipsis in for omitted words, never
join text from two places, and never add quotation marks around it. A shorter
quote that is exact is worth more than a longer one that is abridged, so when a
passage is too long to copy whole, quote the one sentence that carries it. Give
only a concise evidence-based explanation, not hidden chain-of-thought. For
inconclusive, evidence_quote may be null.

citing_proposition:
{{citing_proposition}}
""".strip()


class _PinpointProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verdict: Literal["supports", "inconclusive"]
    reasoning: str = Field(min_length=1)
    evidence_quote: str | None = None


async def run_mellea_pinpoint_check(
    validation: CitationValidation,
    *,
    retrieval: ReporterPageRetrievalNode,
    proposition: MelleaCitingPropositionExtractionNode,
    session: MelleaSession | None = None,
) -> MelleaPinpointCheckNode:
    """Assess the citing proposition against explicitly retrieved page evidence."""
    evidence = retrieval.evidence
    if (
        retrieval.outcome is not ReporterPageRetrievalOutcome.FOUND
        or evidence is None
        or proposition.outcome is not MelleaCitingPropositionExtractionOutcome.IDENTIFIED
        or proposition.proposition is None
    ):
        return _node(
            retrieval,
            proposition,
            ValidationNodeStatus.SKIPPED,
            MelleaPinpointCheckOutcome.UNAVAILABLE,
            status_message=(
                "Skipped Mellea pinpoint inference because required grounded evidence is unavailable."
            ),
            outcome_message="A citing proposition and retrieved reporter page are both required.",
        )

    try:
        spec = InstructIvrSpec(
            description=INSTRUCTION,
            grounding_context={
                "cited_reporter_page": evidence.text,
            },
            user_variables={"citing_proposition": proposition.proposition},
            output_format=_PinpointProposal,
            requirements=[
                req("Return a valid pinpoint-inference object.", validation_fn=_validate_schema),
                req(
                    "A conclusive judgment must identify a uniquely grounded evidence quote.",
                    validation_fn=lambda ctx: _validate_grounding(ctx, evidence.text),
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
                retrieval,
                proposition,
                ValidationNodeStatus.FAILED,
                MelleaPinpointCheckOutcome.FAILED,
                status_message="Mellea pinpoint inference exhausted its repair attempts.",
                outcome_message="No grounded pinpoint inference is available.",
                error="Mellea pinpoint inference exhausted its repair budget",
            )
        proposal = _parse(result.result.value)
        resolved = (
            resolve_evidence_quote(evidence.text, proposal.evidence_quote)
            if proposal.evidence_quote is not None
            else None
        )
    except Exception as exc:
        return _node(
            retrieval,
            proposition,
            ValidationNodeStatus.FAILED,
            MelleaPinpointCheckOutcome.FAILED,
            status_message="Mellea pinpoint inference failed during execution.",
            outcome_message="No grounded pinpoint inference is available.",
            error=f"{type(exc).__name__}: {exc}",
        )

    return _node(
        retrieval,
        proposition,
        ValidationNodeStatus.SUCCEEDED,
        MelleaPinpointCheckOutcome(proposal.verdict),
        proposal=proposal,
        resolved=resolved,
        status_message="Mellea pinpoint inference completed.",
        outcome_message={
            "supports": "The cited reporter page supports the proposition attributed by the citing text.",
            "inconclusive": "The available text does not permit a reliable pinpoint judgment.",
        }[proposal.verdict],
    )


def _parse(value: object) -> _PinpointProposal:
    try:
        return _PinpointProposal.model_validate_json(value)
    except ValidationError as exc:
        msg = f"Invalid Mellea pinpoint output: {exc}"
        raise ValueError(msg) from exc


def _validate_schema(ctx: Context) -> ValidationResult:
    try:
        _parse(ctx.last_output().value)
    except ValueError as exc:
        return ValidationResult(result=False, reason=str(exc))
    return ValidationResult(result=True)


def _validate_grounding(ctx: Context, page_text: str) -> ValidationResult:
    proposal = _parse(ctx.last_output().value)
    quote = proposal.evidence_quote
    if proposal.verdict != "inconclusive" and (quote is None or not quote.strip()):
        return ValidationResult(
            result=False,
            reason="A supports judgment requires an evidence_quote.",
        )
    if quote is None:
        return ValidationResult(result=True)
    resolved = resolve_evidence_quote(page_text, quote)
    return ValidationResult(
        result=resolved is not None,
        reason=None if resolved is not None else _grounding_failure(quote),
    )


def _grounding_failure(quote: str) -> str:
    """Say why a quote did not ground, so a repair turn has something to act on.

    A bare "does not resolve" tells the model nothing, and it re-sends the same
    quote. The two ways it goes wrong -- eliding with an ellipsis, and wrapping
    the passage in quotation marks the page does not have -- are both visible in
    the string itself, so name them.
    """
    if _ELLIPSIS.search(quote):
        return (
            "evidence_quote stands an ellipsis in for omitted words, so it is not one "
            "run of characters from cited_reporter_page. Quote a single unbroken "
            "passage exactly as it appears, even if that means quoting less of it."
        )
    stripped = quote.strip()
    if stripped[:1] in _QUOTE_MARKS or stripped[-1:] in _QUOTE_MARKS:
        return (
            "evidence_quote is wrapped in quotation marks that are not around it in "
            "cited_reporter_page. Copy only the characters that appear there."
        )
    return (
        "evidence_quote does not appear in cited_reporter_page exactly once. Copy a "
        "distinctive passage from it verbatim."
    )


def _node(
    retrieval: ReporterPageRetrievalNode,
    proposition: MelleaCitingPropositionExtractionNode,
    status: ValidationNodeStatus,
    outcome: MelleaPinpointCheckOutcome,
    *,
    proposal: _PinpointProposal | None = None,
    resolved: ResolvedEvidenceQuote | None = None,
    status_message: str | None = None,
    outcome_message: str | None = None,
    error: str | None = None,
) -> MelleaPinpointCheckNode:
    return MelleaPinpointCheckNode(
        node_id=f"{retrieval.node_id}:mellea_pinpoint_check",
        status=status,
        outcome=outcome,
        reasoning=proposal.reasoning.strip() if proposal is not None else None,
        evidence_quote=resolved.text if resolved is not None else None,
        evidence_span=resolved.span if resolved is not None else None,
        evidence_match_method=resolved.method if resolved is not None else None,
        evidence_match_score=resolved.score if resolved is not None else None,
        depends_on=(retrieval.node_id, proposition.node_id),
        status_message=status_message,
        outcome_message=outcome_message,
        error=error,
    )
