"""Mellea case-name re-extraction."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import Enum
from typing import TYPE_CHECKING

from mellea.core import ValidationResult
from mellea.core.base import Context, ModelOutputThunk
from mellea.core.requirement import Requirement
from mellea.stdlib.components import Message
from mellea.stdlib.requirements import check, req
from mellea.stdlib.sampling import MultiTurnStrategy
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

from mellea_lrc.assessment.context import is_text_in_context
from mellea_lrc.assessment.model_options import structured_model_options
from mellea_lrc.assessment.types.case_name import CaseNameProposal
from mellea_lrc.assessment.types.common import ChatTurn
from mellea_lrc.llm import InstructIvrSpec, run_instruct_ivr

if TYPE_CHECKING:
    from mellea import MelleaSession

MISSING_EXTRACTED_CASE_NAME_PROMPT = "<NO_EXTRACTED_CASE_NAME>"
REEXTRACTION_MAX_TOKENS = 512
PROPOSAL_ADAPTER = TypeAdapter(CaseNameProposal)


class ReextractionStatus(str, Enum):
    """Outcome of the re-extraction workflow."""

    ACCEPTED = "accepted"
    EMPTY = "empty"
    INVALID = "invalid"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class ReextractionResult:
    """Complete case-name proposal outcome."""

    status: ReextractionStatus
    proposal: CaseNameProposal | None
    error_message: str | None = None
    chat_history: tuple[ChatTurn, ...] | None = None

    def to_json(self) -> dict[str, object]:
        """Return a JSON-ready representation for diagnostics."""
        return {
            "status": self.status.value,
            "proposal": asdict(self.proposal) if self.proposal is not None else None,
            "error_message": self.error_message,
            "chat_history": (
                [
                    {
                        "role": turn.role,
                        "content": turn.content,
                        "extra_data": turn.extra_data.to_dict(),
                    }
                    for turn in self.chat_history
                ]
                if self.chat_history is not None
                else None
            ),
        }


class _ReextractionProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    available: bool
    case_name: str | None = None


JSON_OUTPUT_REQUIREMENT = (
    'Return exactly one JSON object with shape '
    '{"available":true_or_false,"case_name":"... or null"}.'
)
REEXTRACTION_INSTRUCTION = """
Extract the case name that actually appears in local_context.

Your task is faithful extraction: copy whatever case name is present in
local_context, even if it differs from retrieved_case_name or extracted_case_name.
Do not correct toward retrieved_case_name. The downstream system handles whether
names match.

Set available to false only when local_context contains no identifiable case
name at all. If extracted_case_name is <NO_EXTRACTED_CASE_NAME>, no prior
extraction exists. Do not treat locator strings, such as citation numbers, as
case names.

extracted_case_name:
{{extracted_case_name}}

retrieved_case_name:
{{retrieved_case_name}}
""".strip()


async def _propose_case_name_reextraction(
    session: MelleaSession,
    *,
    local_context: str,
    extracted_case_name: str,
    retrieved_case_name: str,
    requirements: list[Requirement],
    strategy: MultiTurnStrategy,
    model_options: dict[str, object],
) -> tuple[_ReextractionProposal, Context, bool]:
    """Propose a grounded case name through direct Mellea instruct IVR."""
    spec = InstructIvrSpec(
        description=REEXTRACTION_INSTRUCTION,
        grounding_context={"local_context": local_context},
        user_variables={
            "extracted_case_name": extracted_case_name,
            "retrieved_case_name": retrieved_case_name,
        },
        requirements=requirements,
    )
    result = await run_instruct_ivr(session, spec, strategy=strategy, model_options=model_options)
    proposal = _reextraction_proposal_from_output(result.result.value)
    return proposal, result.result_ctx, result.success


def _chat_history_from_context(ctx: Context) -> tuple[ChatTurn, ...]:
    turns: list[ChatTurn] = []
    for item in ctx.as_list():
        if isinstance(item, Message):
            turns.append(ChatTurn(role=item.role, content=item.content))
        elif isinstance(item, ModelOutputThunk) and item.value:
            turns.append(ChatTurn(role="assistant", content=item.value))
    return tuple(turns)


def _validate_availability_consistency(ctx: Context) -> ValidationResult:
    try:
        proposal = _reextraction_proposal_from_context(ctx)
    except ValueError as exc:
        return ValidationResult(result=False, reason=str(exc))
    if proposal.available == (proposal.case_name is not None):
        return ValidationResult(result=True)
    reason = (
        "available is true but case_name is null"
        if proposal.available
        else "available is false but case_name is not null"
    )
    return ValidationResult(result=False, reason=reason)


def _validate_grounding(ctx: Context, document_context: str) -> ValidationResult:
    try:
        proposal = _reextraction_proposal_from_context(ctx)
    except ValueError as exc:
        return ValidationResult(result=False, reason=str(exc))
    if not proposal.available:
        return ValidationResult(result=True)
    if proposal.case_name is not None and is_text_in_context(proposal.case_name, document_context):
        return ValidationResult(result=True)
    return ValidationResult(
        result=False,
        reason=f"case_name={proposal.case_name!r} was not copied exactly from local_context",
    )


def _validate_output_schema(ctx: Context) -> ValidationResult:
    try:
        _reextraction_proposal_from_context(ctx)
    except ValueError as exc:
        return ValidationResult(result=False, reason=str(exc))
    return ValidationResult(result=True)


def _reextraction_proposal_from_output(output: str | object) -> _ReextractionProposal:
    if not isinstance(output, str):
        msg = f"LLM output was not text: {type(output).__name__}"
        raise ValueError(msg)
    try:
        payload = json.loads(output)
    except json.JSONDecodeError as exc:
        msg = f"LLM output was not valid JSON: {exc}"
        raise ValueError(msg) from exc
    if not isinstance(payload, dict):
        msg = "LLM output JSON was not an object"
        raise ValueError(msg)
    try:
        return _ReextractionProposal.model_validate(payload)
    except ValidationError as exc:
        msg = f"LLM output did not match re-extraction schema: {exc}"
        raise ValueError(msg) from exc


def _reextraction_proposal_from_context(ctx: Context) -> _ReextractionProposal:
    return _reextraction_proposal_from_output(ctx.last_output().value)


def _reextraction_requirements(document_context: str) -> list[Requirement]:
    return [
        req(JSON_OUTPUT_REQUIREMENT, validation_fn=_validate_output_schema),
        check(
            "available must be true when case_name is provided, and false when case_name is null",
            validation_fn=_validate_availability_consistency,
        ),
        req(
            "case_name must be a string copied exactly from local_context",
            validation_fn=lambda ctx: _validate_grounding(ctx, document_context),
        ),
    ]


async def reextract_case_name(
    session: MelleaSession,
    *,
    document_context: str,
    extracted_case_name: str | None,
    courtlistener_case_name: str,
) -> ReextractionResult:
    """Propose a case name with a deterministic local-grounding requirement."""
    try:
        proposal, final_ctx, success = await _propose_case_name_reextraction(
            session,
            local_context=document_context,
            extracted_case_name=case_name_for_prompt(extracted_case_name),
            retrieved_case_name=courtlistener_case_name,
            requirements=_reextraction_requirements(document_context),
            strategy=MultiTurnStrategy(loop_budget=3),
            model_options=structured_model_options(max_tokens=REEXTRACTION_MAX_TOKENS),
        )
    except Exception as exc:
        return ReextractionResult(ReextractionStatus.FAILED, None, error_message=str(exc))

    history = _chat_history_from_context(final_ctx)
    if not success:
        return ReextractionResult(
            ReextractionStatus.FAILED,
            None,
            error_message="Re-extraction exhausted retries without satisfying requirements.",
            chat_history=history,
        )
    if not proposal.available or proposal.case_name is None:
        return ReextractionResult(ReextractionStatus.EMPTY, None, chat_history=history)
    return ReextractionResult(
        ReextractionStatus.ACCEPTED,
        CaseNameProposal(case_name=proposal.case_name),
        chat_history=history,
    )


def proposal_from_output(output: str) -> CaseNameProposal | None:
    """Parse a case-name proposal from JSON output."""
    try:
        payload = json.loads(output)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    try:
        return PROPOSAL_ADAPTER.validate_python(payload)
    except (ValidationError, ValueError):
        return None


def case_name_for_prompt(extracted_case_name: str | None) -> str:
    """Represent a missing extracted case name explicitly in model prompts."""
    return extracted_case_name or MISSING_EXTRACTED_CASE_NAME_PROMPT


def validate_proposal(
    proposal: CaseNameProposal | None,
    document_context: str,
) -> tuple[ReextractionStatus, str | None]:
    """Validate proposal grounding and return retryable diagnostics."""
    if proposal is None:
        return ReextractionStatus.INVALID, "Output could not be parsed as a proposal."
    if not is_text_in_context(proposal.case_name, document_context):
        return (
            ReextractionStatus.INVALID,
            f"case_name={proposal.case_name!r} was not copied exactly from local_context.",
        )
    return ReextractionStatus.ACCEPTED, None
