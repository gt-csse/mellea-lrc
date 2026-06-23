"""Standalone Mellea workflow for case-name re-extraction."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from enum import Enum
from typing import cast

from mellea import MelleaSession, generative
from mellea.stdlib.sampling import RejectionSamplingStrategy
from pydantic import TypeAdapter, ValidationError

from mellea_lrc.assessment.types import ModifiedExtractedCitationProposal, is_in_context
from mellea_lrc.llm import LlmProvider, LlmProviderConfig, llm_provider_config_from_env

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

    Return exactly one JSON object with these keys: plaintiff, defendant,
    case_name. Each value must be a string copied exactly from local_context
    or null. Do not add markdown, comments, or explanatory text.

    Use case_name when the corrected full case name appears as one visible string.
    Use plaintiff and defendant when the corrected parties appear separately.
    Propose a correction ONLY when local_context shows a more complete or more
    correct case name than extracted_case_name. If the current extraction is
    already the best grounded reading, return:
    {"plaintiff": null, "defendant": null, "case_name": null}

    If extracted_case_name is <NO_EXTRACTED_CASE_NAME>, no plaintiff/defendant text
    was extracted. Do not treat the locator string as a case name.

    If previous_output and validation_error are provided, revise the previous
    proposal to fix validation_error. The corrected proposal must still copy every
    non-null string exactly from local_context.
    """


async def reextract_case_name_with_mellea(
    session: MelleaSession,
    *,
    document_context: str,
    extracted_case_name: str | None,
    courtlistener_case_name: str,
    attempts: int = REEXTRACTION_ATTEMPTS,
) -> ReextractionResult:
    """Run explicit multi-attempt re-extraction with validation feedback."""
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
                model_options=_structured_model_options(max_tokens=REEXTRACTION_MAX_TOKENS),
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


async def reextract_case_name_with_deepseek(
    config: LlmProviderConfig,
    *,
    document_context: str,
    extracted_case_name: str | None,
    courtlistener_case_name: str,
    attempts: int = REEXTRACTION_ATTEMPTS,
) -> ReextractionResult:
    """Run the same retry contract using DeepSeek's JSON-object API."""
    prompt_case_name = case_name_for_prompt(extracted_case_name)
    messages = _deepseek_reextraction_messages(
        document_context=document_context,
        extracted_case_name=prompt_case_name,
        courtlistener_case_name=courtlistener_case_name,
    )
    attempt_results: list[ReextractionAttempt] = []
    validation_error: str | None = None

    for attempt in range(1, attempts + 1):
        request_messages = messages
        if attempt_results:
            previous = attempt_results[-1]
            request_messages = [
                *messages,
                {"role": "assistant", "content": previous.raw_output or "{}"},
                {
                    "role": "user",
                    "content": (
                        "The previous proposal failed programmatic validation: "
                        f"{previous.error_message}. Return corrected JSON only."
                    ),
                },
            ]
        try:
            payload = await _deepseek_json_chat(
                config,
                messages=request_messages,
                max_tokens=REEXTRACTION_MAX_TOKENS,
            )
            raw_output = json.dumps(payload)
            proposal = PROPOSAL_ADAPTER.validate_python(payload)
            status, validation_error = validate_proposal(proposal, document_context)
        except Exception as exc:
            raw_output = None
            proposal = None
            status = ReextractionStatus.INVALID
            validation_error = f"model output could not be parsed or validated: {exc}"

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

    return ReextractionResult(
        ReextractionStatus.FAILED,
        None,
        tuple(attempt_results),
        validation_error or "Re-extraction failed after retries.",
    )


async def reextract_case_name(
    session: MelleaSession,
    *,
    document_context: str,
    extracted_case_name: str | None,
    courtlistener_case_name: str,
    attempts: int = REEXTRACTION_ATTEMPTS,
) -> ReextractionResult:
    """Provider-aware re-extraction workflow used by assessment and scripts."""
    config = llm_provider_config_from_env(os.environ)
    if config.provider == LlmProvider.DEEPSEEK:
        return await reextract_case_name_with_deepseek(
            config,
            document_context=document_context,
            extracted_case_name=extracted_case_name,
            courtlistener_case_name=courtlistener_case_name,
            attempts=attempts,
        )
    return await reextract_case_name_with_mellea(
        session,
        document_context=document_context,
        extracted_case_name=extracted_case_name,
        courtlistener_case_name=courtlistener_case_name,
        attempts=attempts,
    )


def validate_proposal(
    proposal: ModifiedExtractedCitationProposal | None,
    document_context: str,
) -> tuple[ReextractionStatus, str | None]:
    """Validate proposal grounding and return retryable diagnostics."""
    if proposal is None:
        return ReextractionStatus.INVALID, "Output could not be parsed as a proposal."
    values = {
        "plaintiff": proposal.plaintiff,
        "defendant": proposal.defendant,
        "case_name": proposal.case_name,
    }
    non_empty = {field: value for field, value in values.items() if value}
    if not non_empty:
        return ReextractionStatus.EMPTY, None
    missing = [
        f"{field}={value!r}"
        for field, value in non_empty.items()
        if not is_in_context(value, document_context)
    ]
    if missing:
        return (
            ReextractionStatus.INVALID,
            "The following proposed fields were not copied exactly from local_context: "
            + "; ".join(missing),
        )
    return ReextractionStatus.ACCEPTED, None


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


def _structured_model_options(*, max_tokens: int) -> dict[str, object]:
    return llm_provider_config_from_env(os.environ).mellea_call_options(max_tokens=max_tokens)


def _deepseek_reextraction_messages(
    *,
    document_context: str,
    extracted_case_name: str,
    courtlistener_case_name: str,
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Re-extract a corrected case name from local_context. Return JSON "
                "only, for example "
                '{"plaintiff":null,"defendant":null,"case_name":null}. '
                "Copy any non-null string exactly from local_context. Do not use "
                "the retrieved case name to invent text. If extracted_case_name "
                f"is {MISSING_EXTRACTED_CASE_NAME_PROMPT}, no plaintiff/defendant "
                "text was extracted; do not treat the locator string as a case name."
            ),
        },
        {
            "role": "user",
            "content": (
                f"local_context:\n{document_context}\n\n"
                f"extracted_case_name: {extracted_case_name}\n"
                f"retrieved_case_name: {courtlistener_case_name}\n\n"
                "Return JSON with exactly these keys: plaintiff, defendant, "
                "case_name. Use null when no correction is warranted."
            ),
        },
    ]


async def _deepseek_json_chat(
    config: LlmProviderConfig,
    *,
    messages: list[dict[str, str]],
    max_tokens: int,
) -> dict[str, object]:
    try:
        from openai import AsyncOpenAI  # noqa: PLC0415
    except ImportError as exc:
        msg = "DeepSeek LLM calls require the OpenAI client. Run with: uv sync --group llm"
        raise RuntimeError(msg) from exc

    client = AsyncOpenAI(api_key=config.api_key, base_url=config.chat_completions_base_url())
    response = await client.chat.completions.create(
        model=config.model,
        messages=messages,  # type: ignore[arg-type]
        temperature=0,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
        extra_body={"thinking": {"type": "disabled"}},
    )
    content = response.choices[0].message.content or ""
    if not content.strip():
        msg = "DeepSeek JSON Output returned empty content."
        raise RuntimeError(msg)
    payload = json.loads(content)
    if not isinstance(payload, dict):
        msg = "DeepSeek JSON Output returned a non-object JSON value."
        raise RuntimeError(msg)
    return cast("dict[str, object]", payload)
