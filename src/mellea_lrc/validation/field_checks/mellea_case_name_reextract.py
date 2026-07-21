"""Mellea plaintiff/defendant re-extraction from local citation context."""

from __future__ import annotations

import os
import re
from typing import TYPE_CHECKING, Literal

from mellea.core import ValidationResult
from mellea.stdlib.requirements import check, req
from mellea.stdlib.sampling import MultiTurnStrategy
from pydantic import BaseModel, ConfigDict, ValidationError

from mellea_lrc.llm import (
    InstructIvrSpec,
    llm_api_config_from_env,
    run_instruct_ivr,
    start_mellea_session_from_env,
)
from mellea_lrc.validation.types import (
    CaseNameCheckOutcome,
    CaseNameReextractionNode,
    CaseNameReextractionOutcome,
    ExactLocatorLookupNode,
    LocatorLookupOutcome,
    ValidationNodeStatus,
)

if TYPE_CHECKING:
    from mellea import MelleaSession
    from mellea.core.base import Context

    from mellea_lrc.validation.types import CitationValidation

CONTEXT_BEFORE_CHARS = 320
CONTEXT_AFTER_CHARS = 160
REEXTRACTION_MAX_TOKENS = 256
REEXTRACTION_MAX_REPAIR_TURNS = 2

REEXTRACTION_INSTRUCTION = """
Extract the plaintiff and defendant copied in local_context for the citation
marked by locator. Treat locator as the boundary marker. Prefer the nearest
copied "plaintiff v. defendant" name before locator, even when a docket number,
parallel citation, or other citation metadata occurs between the name and
locator. If local_context contains multiple citations, do not borrow parties
from another citation.

Copy the party text as written. Minor spacing or punctuation damage does not
erase otherwise literal parties. Do not use outside knowledge, expand legal
abbreviations, normalize a party toward a known case, or return a combined case
name. Return plaintiff and defendant as separate fields.

Return classification "complete_case_name" when both parties are present,
"partial_case_name" when exactly one is present, and "no_case_name" only when no
party bound to locator appears in local_context.

locator:
{{locator}}
""".strip()


class _PartyProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    classification: Literal["complete_case_name", "partial_case_name", "no_case_name"]
    plaintiff: str | None = None
    defendant: str | None = None


async def run_mellea_case_name_reextract(
    validation: CitationValidation,
    *,
    document_text: str,
    session: MelleaSession | None = None,
) -> CaseNameReextractionNode:
    """Extract grounded parties from local context as one appendable operation."""
    dependency = next(
        (
            node
            for node in reversed(validation.nodes)
            if (
                getattr(node, "outcome", None) is CaseNameCheckOutcome.NOT_SEMANTIC_MATCH
                or (
                    isinstance(node, ExactLocatorLookupNode)
                    and node.outcome is LocatorLookupOutcome.NOT_FOUND
                )
            )
        ),
        validation.nodes[-1],
    )
    lookup = next(
        (node for node in reversed(validation.nodes) if isinstance(node, ExactLocatorLookupNode)),
        None,
    )
    locator = lookup.locator if lookup is not None else None
    if not locator:
        return _node(
            validation,
            dependency.node_id,
            ValidationNodeStatus.SKIPPED,
            CaseNameReextractionOutcome.EMPTY,
        )

    span = validation.citation.span
    start = max(0, span.start - CONTEXT_BEFORE_CHARS)
    end = min(len(document_text), span.end + CONTEXT_AFTER_CHARS)
    local_context = document_text[start:end]
    before_locator = document_text[start:span.start]
    try:
        resolved_session = session or start_mellea_session_from_env()
        options = llm_api_config_from_env(os.environ).mellea_call_options(
            max_tokens=REEXTRACTION_MAX_TOKENS
        )
        spec = InstructIvrSpec(
            description=REEXTRACTION_INSTRUCTION,
            grounding_context={"local_context": local_context},
            user_variables={"locator": locator},
            requirements=[
                req("Return a valid plaintiff/defendant JSON object.", validation_fn=_validate_schema),
                check(
                    "classification must match party availability",
                    validation_fn=_validate_classification,
                ),
                req(
                    "parties must be copied before locator in local_context",
                    validation_fn=lambda ctx: _validate_grounding(ctx, before_locator),
                ),
            ],
        )
        result = await run_instruct_ivr(
            resolved_session,
            spec,
            strategy=MultiTurnStrategy(loop_budget=REEXTRACTION_MAX_REPAIR_TURNS),
            model_options=options,
        )
        if not result.success:
            return _node(
                validation,
                dependency.node_id,
                ValidationNodeStatus.FAILED,
                CaseNameReextractionOutcome.FAILED,
                error="Case-name re-extraction exhausted its repair budget",
            )
        proposal = _proposal(result.result.value)
    except Exception as exc:
        return _node(
            validation,
            dependency.node_id,
            ValidationNodeStatus.FAILED,
            CaseNameReextractionOutcome.FAILED,
            error=f"{type(exc).__name__}: {exc}",
        )

    if proposal.plaintiff is None and proposal.defendant is None:
        return _node(
            validation,
            dependency.node_id,
            ValidationNodeStatus.SKIPPED,
            CaseNameReextractionOutcome.EMPTY,
        )
    return _node(
        validation,
        dependency.node_id,
        ValidationNodeStatus.SUCCEEDED,
        CaseNameReextractionOutcome.ACCEPTED,
        plaintiff=proposal.plaintiff,
        defendant=proposal.defendant,
    )


def _node(
    validation: CitationValidation,
    dependency_id: str,
    status: ValidationNodeStatus,
    outcome: CaseNameReextractionOutcome,
    *,
    plaintiff: str | None = None,
    defendant: str | None = None,
    error: str | None = None,
) -> CaseNameReextractionNode:
    return CaseNameReextractionNode(
        node_id=f"{validation.citation_id}:case_name_reextract",
        status=status,
        outcome=outcome,
        plaintiff=plaintiff,
        defendant=defendant,
        depends_on=(dependency_id,),
        error=error,
    )


def _proposal(output: object) -> _PartyProposal:
    try:
        return _PartyProposal.model_validate_json(output)
    except ValidationError as exc:
        msg = f"Invalid case-name re-extraction output: {exc}"
        raise ValueError(msg) from exc


def _validate_schema(ctx: Context) -> ValidationResult:
    try:
        _proposal(ctx.last_output().value)
    except ValueError as exc:
        return ValidationResult(result=False, reason=str(exc))
    return ValidationResult(result=True)


def _validate_classification(ctx: Context) -> ValidationResult:
    proposal = _proposal(ctx.last_output().value)
    count = sum(value is not None for value in (proposal.plaintiff, proposal.defendant))
    expected = ("no_case_name", "partial_case_name", "complete_case_name")[count]
    return ValidationResult(
        result=proposal.classification == expected,
        reason=None if proposal.classification == expected else f"expected {expected}",
    )


def _validate_grounding(ctx: Context, before_locator: str) -> ValidationResult:
    proposal = _proposal(ctx.last_output().value)
    missing = [
        label
        for label, value in (("plaintiff", proposal.plaintiff), ("defendant", proposal.defendant))
        if value is not None and not _is_grounded(value, before_locator)
    ]
    return ValidationResult(
        result=not missing,
        reason=None if not missing else f"not copied before locator: {', '.join(missing)}",
    )


def _is_grounded(value: str, context: str) -> bool:
    """Return whether copied text occurs with exact tokens and flexible whitespace."""
    pattern = r"\s+".join(re.escape(piece) for piece in value.split())
    return re.search(pattern, context) is not None
