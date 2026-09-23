"""One grounded reread and case-name comparison after a literal mismatch."""

from __future__ import annotations

import os
import re
from typing import TYPE_CHECKING, Annotated, Literal

from mellea.core import ValidationResult
from mellea.stdlib.requirements import req
from mellea.stdlib.sampling import MultiTurnStrategy
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from mellea_lrc.llm import (
    InstructIvrSpec,
    IvrRun,
    llm_api_config_from_env,
    run_instruct_ivr,
    start_mellea_session_from_env,
)
from mellea_lrc.validation.field_checks.source_case_name import ground_source_case_name
from mellea_lrc.validation.types import (
    MelleaCaseNameCheckOutcome,
    MelleaCaseNameReviewNode,
    ValidationNodeStatus,
)

if TYPE_CHECKING:
    from mellea import MelleaSession
    from mellea.core.base import Context

    from mellea_lrc.validation.types import (
        CandidateEvaluationNode,
        CitationValidation,
        ExactCaseNameCheckNode,
        ExactLocatorLookupNode,
    )


CONTEXT_BEFORE_CHARS = 320
CONTEXT_AFTER_CHARS = 160
MAX_TOKENS = 768

PREFIX = """
Review one complete citation in the filing after a literal case-name comparison
did not settle it. First reread the filing itself. Copy only the case name
belonging to locator from local_context, before that locator. Do not borrow a
name from another citation or copy the retrieved record's name into the filing.
The entire written name can be a one-party proceeding such as In re or Ex parte.
Return null when the filing does not state a name that belongs to this locator.

Then compare that reread name with retrieved_case_name. Judge whether they name
the same case, allowing ordinary legal abbreviation, party shortening, and
legal-entity suffix variation. Different parties are a mismatch. A misspelled
party or title is also a mismatch, even when the intended case is recognizable.
Do not use outside knowledge or change the reporter locator. Explain the
comparison briefly in reason.

For a complete copied name, return equivalent true or false. For a fragment or
no name, return equivalent null. Plaintiff and defendant must be copied from
the filing when present; use defendant alone for a one-party proceeding.
""".strip()

INSTRUCTION = """
locator:
{{locator}}

retrieved_case_name:
{{retrieved_case_name}}
""".strip()


class _ReviewProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    classification: Literal["complete_case_name", "partial_case_name", "no_case_name"]
    case_name_quote: str | None
    plaintiff: str | None
    defendant: str | None
    equivalent: bool | None
    reason: Annotated[str, Field(min_length=1)]


async def run_mellea_case_name_review(
    validation: CitationValidation,
    *,
    trigger: ExactCaseNameCheckNode,
    locator_lookup: ExactLocatorLookupNode,
    candidate: CandidateEvaluationNode,
    document_text: str,
    session: MelleaSession | None = None,
) -> MelleaCaseNameReviewNode:
    """Reread and compare in one model attempt; persist the full IVR trace."""
    retrieved_name = candidate.case_name
    if not locator_lookup.locator or retrieved_name is None:
        return _node(
            trigger,
            retrieved_name,
            status=ValidationNodeStatus.SKIPPED,
            outcome=MelleaCaseNameCheckOutcome.UNAVAILABLE,
            outcome_message="The locator or retrieved case name is unavailable for comparison.",
        )

    span = validation.citation.locator_span
    start = max(0, span.start - CONTEXT_BEFORE_CHARS)
    end = min(len(document_text), span.end + CONTEXT_AFTER_CHARS)
    local_context = document_text[start:end]
    before_locator = document_text[start : span.start]
    try:
        spec = InstructIvrSpec(
            description=INSTRUCTION,
            prefix=PREFIX,
            grounding_context={"local_context": local_context},
            user_variables={
                "locator": locator_lookup.locator,
                "retrieved_case_name": retrieved_name,
            },
            output_format=_ReviewProposal,
            requirements=[
                req(
                    "The copied local name and parties must precede the locator; equivalence requires a complete name",
                    validation_fn=lambda ctx: _validate_review(ctx, before_locator),
                ),
            ],
        )
        result = await run_instruct_ivr(
            session or start_mellea_session_from_env(),
            spec,
            strategy=MultiTurnStrategy(loop_budget=1),
            model_options=llm_api_config_from_env(os.environ).mellea_call_options(max_tokens=MAX_TOKENS),
        )
        if not result.success:
            return _node(
                trigger,
                retrieved_name,
                status=ValidationNodeStatus.FAILED,
                outcome=MelleaCaseNameCheckOutcome.FAILED,
                outcome_message="The combined case-name review did not pass validation.",
                error=result.failure_reason or "Combined case-name review failed",
                run=result,
            )
        proposal = _ReviewProposal.model_validate_json(result.output)
    except Exception as exc:
        return _node(
            trigger,
            retrieved_name,
            status=ValidationNodeStatus.FAILED,
            outcome=MelleaCaseNameCheckOutcome.FAILED,
            outcome_message="The combined case-name review failed.",
            error=f"{type(exc).__name__}: {exc}",
        )

    if not _valid_proposal(proposal, before_locator):
        return _node(
            trigger,
            retrieved_name,
            status=ValidationNodeStatus.FAILED,
            outcome=MelleaCaseNameCheckOutcome.FAILED,
            outcome_message="The combined case-name review was inconsistent or ungrounded.",
            error="Review classification, equivalence, or source grounding failed",
            run=result,
        )
    grounded = ground_source_case_name(
        proposal.case_name_quote,
        before_locator=before_locator,
        source_start=start,
        plaintiff=proposal.plaintiff,
        defendant=proposal.defendant,
    )
    outcome = (
        MelleaCaseNameCheckOutcome.UNAVAILABLE
        if proposal.classification != "complete_case_name"
        else MelleaCaseNameCheckOutcome.MATCH
        if proposal.equivalent
        else MelleaCaseNameCheckOutcome.MISMATCH
    )
    return _node(
        trigger,
        retrieved_name,
        status=ValidationNodeStatus.SUCCEEDED,
        outcome=outcome,
        case_name=grounded if proposal.classification == "complete_case_name" else None,
        plaintiff=proposal.plaintiff,
        defendant=proposal.defendant,
        rationale=proposal.reason,
        outcome_message=f"Combined source reread and name comparison: {outcome.value}.",
        run=result,
    )


def _node(
    trigger: ExactCaseNameCheckNode,
    retrieved_name: str | None,
    *,
    status: ValidationNodeStatus,
    outcome: MelleaCaseNameCheckOutcome,
    case_name=None,
    plaintiff: str | None = None,
    defendant: str | None = None,
    rationale: str | None = None,
    outcome_message: str | None = None,
    error: str | None = None,
    run: IvrRun | None = None,
) -> MelleaCaseNameReviewNode:
    return MelleaCaseNameReviewNode(
        node_id=f"{trigger.node_id}:mellea_case_name_review",
        status=status,
        outcome=outcome,
        retrieved_case_name=retrieved_name,
        case_name=case_name,
        plaintiff=plaintiff,
        defendant=defendant,
        rationale=rationale,
        depends_on=(trigger.node_id,),
        status_message="Combined case-name review completed."
        if status is ValidationNodeStatus.SUCCEEDED
        else None,
        outcome_message=outcome_message,
        error=error,
        run=run,
    )


def _validate_review(ctx: Context, before_locator: str) -> ValidationResult:
    try:
        proposal = _ReviewProposal.model_validate_json(ctx.last_output().value)
    except ValidationError as exc:
        return ValidationResult(result=False, reason=str(exc))
    valid = _valid_proposal(proposal, before_locator)
    return ValidationResult(
        result=valid,
        reason=None
        if valid
        else "Copy the local name and parties before the locator; return equivalence only for a complete name.",
    )


def _valid_proposal(proposal: _ReviewProposal, before_locator: str) -> bool:
    quoted = bool(proposal.case_name_quote and proposal.case_name_quote.strip())
    if proposal.classification == "complete_case_name":
        if not quoted or proposal.equivalent is None:
            return False
    elif proposal.classification == "partial_case_name":
        if not quoted or proposal.equivalent is not None:
            return False
    elif quoted or proposal.equivalent is not None or proposal.plaintiff or proposal.defendant:
        return False
    if (
        quoted
        and ground_source_case_name(
            proposal.case_name_quote,
            before_locator=before_locator,
            source_start=0,
        )
        is None
    ):
        return False
    return all(
        value is None or re.search(r"\s+".join(re.escape(piece) for piece in value.split()), before_locator)
        for value in (proposal.plaintiff, proposal.defendant)
    )
