"""Mellea case-name re-extraction."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import Enum

from mellea import MelleaSession, generative
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


@generative
async def _propose_case_name_reextraction(
    local_context: str,
    extracted_case_name: str,
    retrieved_case_name: str,
) -> _ReextractionProposal:
    """Extract the case name that actually appears in local_context.

    Copy the local case name faithfully even when it differs from the retrieved or
    previously extracted name. Return it only when an identifiable case name exists.
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


async def reextract_case_name(
    session: MelleaSession,
    *,
    document_context: str,
    extracted_case_name: str | None,
    courtlistener_case_name: str,
) -> ReextractionResult:
    """Propose a case name with a deterministic local-grounding requirement."""
    final_ctx: Context | None = None
    try:
        proposal, final_ctx = await _propose_case_name_reextraction(
            ChatContext(),
            session.backend,
            local_context=document_context,
            extracted_case_name=case_name_for_prompt(extracted_case_name),
            retrieved_case_name=courtlistener_case_name,
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
