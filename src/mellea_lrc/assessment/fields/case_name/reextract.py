"""Mellea case-name re-extraction."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from enum import Enum
from typing import TYPE_CHECKING

from mellea import generative
from mellea.core import ValidationResult
from mellea.core.base import Context, ModelOutputThunk
from mellea.stdlib.components import Message
from mellea.stdlib.context import ChatContext
from mellea.stdlib.requirements import check, req
from mellea.stdlib.sampling import MultiTurnStrategy
from pydantic import BaseModel, TypeAdapter, ValidationError

from mellea_lrc.assessment.context import is_text_in_context
from mellea_lrc.assessment.model_options import structured_model_options
from mellea_lrc.assessment.types.case_name import CaseNameProposal
from mellea_lrc.assessment.types.common import ChatTurn
from mellea_lrc.llm import LlmResponseFormat, llm_api_config_from_env

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
    available: bool
    case_name: str | None = None


# Passed as output_format_hint when the env selects json_object mode (the schema is
# then stripped by mellea_call_options, so the prompt must describe the JSON shape and
# the {"result": ...} wrapper Mellea expects). Under json_schema mode "none" is passed
# and the schema carries the contract.
JSON_OBJECT_HINT = (
    'Return exactly one JSON object with key "result" containing two fields: '
    '"available" (true when a case name is identifiable in local_context, false otherwise) '
    'and "case_name" (a string copied exactly from local_context when available is true, '
    "null otherwise). "
    'Example - case name found: {"result": {"available": true, "case_name": "Smith v. Jones"}}. '
    'Example - no case name: {"result": {"available": false, "case_name": null}}.'
)


@generative
async def _propose_case_name_reextraction(
    local_context: str,
    extracted_case_name: str,
    retrieved_case_name: str,
    output_format_hint: str,
) -> _ReextractionProposal:
    """Extract the case name that actually appears in local_context.

    Your task is faithful extraction - copy whatever case name is present in local_context,
    even if it differs from retrieved_case_name or extracted_case_name. Do not try to
    correct toward retrieved_case_name. The downstream system handles whether names match.

    Set available to false only when local_context contains no identifiable case name
    at all. If extracted_case_name is <NO_EXTRACTED_CASE_NAME>, no prior extraction
    exists. Do not treat locator strings (e.g. citation numbers) as case names.

    If output_format_hint is "none", follow the JSON schema passed to you.
    Otherwise, return JSON exactly matching the shape described by output_format_hint.
    """


def _chat_history_from_context(ctx: Context) -> tuple[ChatTurn, ...]:
    turns: list[ChatTurn] = []
    for item in ctx.as_list():
        if isinstance(item, Message):
            turns.append(ChatTurn(role=item.role, content=item.content))
        elif isinstance(item, ModelOutputThunk) and item.value:
            turns.append(ChatTurn(role="assistant", content=item.value))
    return tuple(turns)


def _validate_availability_consistency(ctx: Context) -> ValidationResult:
    proposal: _ReextractionProposal = ctx.last_output().parsed_repr
    if proposal.available == (proposal.case_name is not None):
        return ValidationResult(result=True)
    reason = (
        "available is true but case_name is null"
        if proposal.available
        else "available is false but case_name is not null"
    )
    return ValidationResult(result=False, reason=reason)


def _validate_grounding(ctx: Context, document_context: str) -> ValidationResult:
    proposal: _ReextractionProposal = ctx.last_output().parsed_repr
    if not proposal.available:
        return ValidationResult(result=True)
    if proposal.case_name is not None and is_text_in_context(proposal.case_name, document_context):
        return ValidationResult(result=True)
    return ValidationResult(
        result=False,
        reason=f"case_name={proposal.case_name!r} was not copied exactly from local_context",
    )


def _output_format_hint() -> str:
    """Return the JSON-shape hint to pass into the prompt, or ``"none"`` if schema-enforced."""
    fmt = llm_api_config_from_env(os.environ).response_format
    return JSON_OBJECT_HINT if fmt is LlmResponseFormat.JSON_OBJECT else "none"


async def reextract_case_name(
    session: MelleaSession,
    *,
    document_context: str,
    extracted_case_name: str | None,
    courtlistener_case_name: str,
) -> ReextractionResult:
    """Propose a case name with a deterministic local-grounding requirement."""
    try:
        proposal, final_ctx = await _propose_case_name_reextraction(
            ChatContext(),
            session.backend,
            local_context=document_context,
            extracted_case_name=case_name_for_prompt(extracted_case_name),
            retrieved_case_name=courtlistener_case_name,
            output_format_hint=_output_format_hint(),
            requirements=[
                check(
                    "available must be true when case_name is provided, and false when case_name is null",
                    validation_fn=_validate_availability_consistency,
                ),
                req(
                    "case_name must be a string copied exactly from local_context",
                    validation_fn=lambda ctx: _validate_grounding(ctx, document_context),
                ),
            ],
            strategy=MultiTurnStrategy(loop_budget=3),
            model_options=structured_model_options(max_tokens=REEXTRACTION_MAX_TOKENS),
        )
    except Exception as exc:
        return ReextractionResult(ReextractionStatus.FAILED, None, error_message=str(exc))

    history = _chat_history_from_context(final_ctx)
    if not proposal.available or proposal.case_name is None:
        return ReextractionResult(ReextractionStatus.EMPTY, None, chat_history=history)
    if not is_text_in_context(proposal.case_name, document_context):
        return ReextractionResult(
            ReextractionStatus.FAILED,
            None,
            error_message=f"Re-extraction exhausted retries: {proposal.case_name!r} not grounded",
            chat_history=history,
        )
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
    result = payload.get("result", payload)
    try:
        return PROPOSAL_ADAPTER.validate_python(result)
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
