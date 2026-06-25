"""Mellea case-name re-extraction."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import Enum

from mellea import MelleaSession, generative
from mellea.stdlib.sampling import RejectionSamplingStrategy
from pydantic import TypeAdapter, ValidationError

from mellea_lrc.assessment.grounding.proposal import is_in_context
from mellea_lrc.assessment.llm.options import structured_model_options
from mellea_lrc.assessment.types import ModifiedExtractedCitationProposal

MISSING_EXTRACTED_CASE_NAME_PROMPT = "<NO_EXTRACTED_CASE_NAME>"
REEXTRACTION_MAX_TOKENS = 512
REEXTRACTION_ATTEMPTS = 3
REEXTRACTION_STRATEGY = RejectionSamplingStrategy(loop_budget=1)
PROPOSAL_ADAPTER = TypeAdapter(ModifiedExtractedCitationProposal)


class ReextractionStatus(str, Enum):
    """Outcome of the re-extraction workflow."""

    ACCEPTED = "accepted"
    EMPTY = "empty"
    INVALID = "invalid"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class ReextractionAttempt:
    """One model proposal and the result of programmatic checking."""

    attempt: int
    proposal: ModifiedExtractedCitationProposal | None
    raw_output: str | None
    error_message: str | None = None


@dataclass(frozen=True, slots=True)
class ReextractionResult:
    """Complete re-extraction outcome, including failed attempts."""

    status: ReextractionStatus
    proposal: ModifiedExtractedCitationProposal | None
    attempts: tuple[ReextractionAttempt, ...]
    error_message: str | None = None

    def to_json(self) -> dict[str, object]:
        """Return a JSON-ready representation for scripts and diagnostics."""
        return {
            "status": self.status.value,
            "proposal": asdict(self.proposal) if self.proposal is not None else None,
            "attempts": [
                {
                    "attempt": item.attempt,
                    "proposal": asdict(item.proposal) if item.proposal is not None else None,
                    "raw_output": item.raw_output,
                    "error_message": item.error_message,
                }
                for item in self.attempts
            ],
            "error_message": self.error_message,
        }


@generative
async def propose_case_name_reextraction_json(
    local_context: str,
    extracted_case_name: str,
    retrieved_case_name: str,
    previous_output: str | None = None,
    validation_error: str | None = None,
) -> str:
    """Re-extract a corrected case name from local_context as compact JSON.

    Return exactly one JSON object with this key: case_name. The value must be a
    string copied exactly from local_context or null. Do not add markdown,
    comments, or explanatory text.

    Propose a correction ONLY when local_context shows a more complete or more
    correct case name than extracted_case_name. If the current extraction is
    already the best grounded reading, return: {"case_name": null}

    If extracted_case_name is <NO_EXTRACTED_CASE_NAME>, no case name text was
    extracted. Do not treat the locator string as a case name.

    If previous_output and validation_error are provided, revise the previous
    proposal to fix validation_error. The corrected case_name must still be
    copied exactly from local_context.
    """


async def reextract_case_name(
    session: MelleaSession,
    *,
    document_context: str,
    extracted_case_name: str | None,
    courtlistener_case_name: str,
    attempts: int = REEXTRACTION_ATTEMPTS,
) -> ReextractionResult:
    """Run multi-attempt case-name re-extraction through Mellea."""
    prompt_case_name = case_name_for_prompt(extracted_case_name)
    previous_output: str | None = None
    validation_error: str | None = None
    attempt_results: list[ReextractionAttempt] = []

    for attempt in range(1, attempts + 1):
        try:
            raw = await propose_case_name_reextraction_json(
                session,
                local_context=document_context,
                extracted_case_name=prompt_case_name,
                retrieved_case_name=courtlistener_case_name,
                previous_output=previous_output,
                validation_error=validation_error,
                strategy=REEXTRACTION_STRATEGY,
                model_options=structured_model_options(max_tokens=REEXTRACTION_MAX_TOKENS),
            )
        except Exception as exc:
            validation_error = f"model call failed: {exc}"
            attempt_results.append(
                ReextractionAttempt(
                    attempt=attempt,
                    proposal=None,
                    raw_output=None,
                    error_message=validation_error,
                )
            )
            continue

        raw_output = str(raw)
        proposal = proposal_from_output(raw_output)
        status, validation_error = validate_proposal(proposal, document_context)
        attempt_results.append(
            ReextractionAttempt(
                attempt=attempt,
                proposal=proposal,
                raw_output=raw_output,
                error_message=validation_error,
            )
        )
        if status == ReextractionStatus.ACCEPTED:
            return ReextractionResult(status, proposal, tuple(attempt_results))
        if status == ReextractionStatus.EMPTY:
            return ReextractionResult(status, None, tuple(attempt_results))
        previous_output = raw_output

    return ReextractionResult(
        ReextractionStatus.FAILED,
        None,
        tuple(attempt_results),
        validation_error or "Re-extraction failed after retries.",
    )


def proposal_from_output(output: str) -> ModifiedExtractedCitationProposal | None:
    """Parse a JSON proposal from Mellea output."""
    try:
        payload = json.loads(output)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    result = payload.get("result", payload)
    try:
        return PROPOSAL_ADAPTER.validate_python(result)
    except ValidationError:
        return None


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
