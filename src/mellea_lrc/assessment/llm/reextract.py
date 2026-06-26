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

from mellea_lrc.assessment.grounding.proposal import is_in_context
from mellea_lrc.assessment.llm.options import structured_model_options
from mellea_lrc.assessment.types import ModifiedExtractedCitationProposal

MISSING_EXTRACTED_CASE_NAME_PROMPT = "<NO_EXTRACTED_CASE_NAME>"
REEXTRACTION_MAX_TOKENS = 512
PROPOSAL_ADAPTER = TypeAdapter(ModifiedExtractedCitationProposal)


class ReextractionStatus(str, Enum):
    """Outcome of the re-extraction workflow."""

    ACCEPTED = "accepted"
    EMPTY = "empty"
    INVALID = "invalid"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class ReextractionResult:
    """Complete re-extraction outcome."""

    status: ReextractionStatus
    proposal: ModifiedExtractedCitationProposal | None
    error_message: str | None = None
    chat_history: list[dict[str, str]] | None = None

    def to_json(self) -> dict[str, object]:
        """Return a JSON-ready representation for scripts and diagnostics."""
        return {
            "status": self.status.value,
            "proposal": asdict(self.proposal) if self.proposal is not None else None,
            "error_message": self.error_message,
            "chat_history": self.chat_history,
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

    Your task is faithful extraction — copy whatever case name is present in local_context,
    even if it differs from retrieved_case_name or extracted_case_name. Do not try to
    correct toward retrieved_case_name. The downstream system handles whether names match.

    Return exactly one JSON object with key "result" containing two fields:
      - "available": true if a case name can be identified in local_context, false otherwise
      - "case_name": a string copied exactly from local_context when available is true,
                     null when available is false

    Set available to false only when local_context contains no identifiable case name at all.
    If extracted_case_name is <NO_EXTRACTED_CASE_NAME>, no prior extraction exists.
    Do not treat locator strings (e.g. citation numbers) as case names.

    Example — case name found:   {"result": {"available": true,  "case_name": "Smith v. Jones"}}
    Example — no case name:      {"result": {"available": false, "case_name": null}}
    """


def _chat_history_from_context(ctx: Context) -> list[dict[str, str]]:
    turns = []
    for item in ctx.as_list():
        if isinstance(item, Message):
            turns.append({"role": item.role, "content": item.content})
        elif isinstance(item, ModelOutputThunk) and item.value:
            turns.append({"role": "assistant", "content": item.value})
    return turns


def _validate_availability_consistency(ctx: Context) -> ValidationResult:
    proposal: _ReextractionProposal = ctx.last_output().parsed_repr
    if proposal.available == (proposal.case_name is not None):
        return ValidationResult(True)
    if proposal.available:
        return ValidationResult(False, reason="available is true but case_name is null")
    return ValidationResult(False, reason="available is false but case_name is not null")


def _validate_grounding(ctx: Context, document_context: str) -> ValidationResult:
    proposal: _ReextractionProposal = ctx.last_output().parsed_repr
    if not proposal.available:
        return ValidationResult(True)
    if is_in_context(proposal.case_name, document_context):
        return ValidationResult(True)
    return ValidationResult(
        False,
        reason=f"case_name={proposal.case_name!r} was not copied exactly from local_context",
    )


async def reextract_case_name(
    session: MelleaSession,
    *,
    document_context: str,
    extracted_case_name: str | None,
    courtlistener_case_name: str,
) -> ReextractionResult:
    """Run case-name re-extraction through Mellea with programmatic grounding check.

    Uses MultiTurnStrategy so that on requirement failure the model receives a new
    user turn with the failure reason — unlike simple rejection sampling, this gives
    the model different input on each retry even at temperature=0.
    """
    final_ctx: Context | None = None
    try:
        # Use (ChatContext, backend) form so MultiTurnStrategy can build a conversation.
        # A fresh ChatContext is created per call so there is no state bleed-over.
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

    if not proposal.available:
        return ReextractionResult(ReextractionStatus.EMPTY, None, chat_history=history)

    # Guard against exhausted retries: strategy returns the last attempt even on failure.
    if not is_in_context(proposal.case_name, document_context):
        return ReextractionResult(
            ReextractionStatus.FAILED,
            None,
            error_message=f"Re-extraction exhausted retries: {proposal.case_name!r} not grounded",
            chat_history=history,
        )

    return ReextractionResult(
        ReextractionStatus.ACCEPTED,
        ModifiedExtractedCitationProposal(case_name=proposal.case_name),
    )


def proposal_from_output(output: str) -> ModifiedExtractedCitationProposal | None:
    """Parse a JSON proposal from Mellea output."""
    try:
        payload = json.loads(output)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return _parse_proposal_payload(payload)


def case_name_for_prompt(extracted_case_name: str | None) -> str:
    """Represent a missing extracted case name explicitly in model prompts."""
    return extracted_case_name if extracted_case_name else MISSING_EXTRACTED_CASE_NAME_PROMPT


def validate_proposal(
    proposal: ModifiedExtractedCitationProposal | None,
    document_context: str,
) -> tuple[ReextractionStatus, str | None]:
    """Validate proposal grounding and return retryable diagnostics."""
    if proposal is None:
        return ReextractionStatus.INVALID, "Output could not be parsed as a proposal."
    if not proposal.case_name:
        return ReextractionStatus.EMPTY, None
    if not is_in_context(proposal.case_name, document_context):
        return (
            ReextractionStatus.INVALID,
            f"case_name={proposal.case_name!r} was not copied exactly from local_context.",
        )
    return ReextractionStatus.ACCEPTED, None


def _parse_proposal_payload(
    payload: dict[str, object],
) -> ModifiedExtractedCitationProposal | None:
    result = payload.get("result", payload)
    try:
        return PROPOSAL_ADAPTER.validate_python(result)
    except ValidationError:
        return None
