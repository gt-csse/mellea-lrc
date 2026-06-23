"""Mellea-backed semantic assessment for citations."""

from __future__ import annotations

import asyncio
import json
import os
from typing import Literal, cast

from mellea import MelleaSession, generative
from mellea.stdlib.sampling import RejectionSamplingStrategy

from mellea_lrc.assessment.case_name import assess_case_name_exact_match
from mellea_lrc.assessment.reextraction import (
    ReextractionResult,
    ReextractionStatus,
    reextract_case_name_with_deepseek,
    reextract_case_name_with_mellea,
)
from mellea_lrc.assessment.types import (
    CaseNameAssessment,
    CaseNameAssessmentRun,
    CaseNameAssessmentStatus,
)
from mellea_lrc.llm import LlmProvider, LlmProviderConfig, llm_provider_config_from_env

CaseNameVerdict = Literal["match", "different_case", "irregular_form"]
# Reasoning-style providers can spend completion budget before emitting the tiny
# final JSON object that Mellea parses. Keep this comfortably above the final
# schema size to avoid empty final content.
CASE_NAME_VERDICT_MAX_TOKENS = 2048
CASE_NAME_CLASSIFICATION_ATTEMPTS = 3
CASE_NAME_CLASSIFICATION_STRATEGY = RejectionSamplingStrategy(loop_budget=3)


@generative
async def classify_case_name(
    local_context: str,
    extracted_case_name: str,
    retrieved_case_name: str,
) -> CaseNameVerdict:
    """Classify how an extracted legal case name relates to the retrieved record.

    You are given the case name pulled from a citation (extracted_case_name), the
    case name of the record retrieved for that citation (retrieved_case_name), and
    the surrounding document text (local_context). Judge ONLY the case name. Do not
    consider volume, reporter, page, pin cite, court, or year.
    If extracted_case_name is <NO_EXTRACTED_CASE_NAME>, no plaintiff/defendant text
    was extracted. Do not treat the locator string as a case name.

    Return one of:
    - "match": the two names denote the SAME case and the extracted form is a
      normal way to cite it. Treat these as acceptable (still "match"): using only
      party surnames and dropping given or middle names (e.g. "United States v.
      Golden" for "United States v. Bobby Ray Golden"), abbreviation, "et al.",
      dropped institutional suffixes (such as "Inc." or "Co."), and ordinary
      citation style. As long as BOTH sides of the "v." are represented by a
      recognizable party, prefer "match".
    - "different_case": the extracted name denotes a DIFFERENT, unrelated case than
      the retrieved record. A differing retrieved name is NOT automatically the
      extractor's fault; report "different_case" and do not assume the extraction
      is wrong.
    - "irregular_form": the names denote the SAME case, but the extracted name is
      genuinely incomplete or garbled BEYOND normal shortening — for example a
      whole party is missing (only one side of the "v." is present), the parties
      are in the wrong order, or the text is broken by stray characters or line
      breaks.
    """


async def assess_case_name_with_mellea(
    session: MelleaSession,
    *,
    citation_id: str,
    extracted_case_name: str | None,
    courtlistener_case_name: str | None,
    document_context: str,
) -> CaseNameAssessmentRun:
    """Assess one case name, using Mellea only when exact equality is not enough."""
    exact_result = assess_case_name_exact_match(
        citation_id=citation_id,
        extracted_case_name=extracted_case_name,
        courtlistener_case_name=courtlistener_case_name,
    )
    if exact_result.status == CaseNameAssessmentStatus.EXACT_MATCH:
        return CaseNameAssessmentRun(assessment=exact_result)
    if not courtlistener_case_name:
        return CaseNameAssessmentRun(assessment=exact_result)

    try:
        return await _assess_case_name_with_mellea_after_exact(
            session,
            citation_id=citation_id,
            extracted_case_name=extracted_case_name,
            courtlistener_case_name=courtlistener_case_name,
            document_context=document_context,
        )
    except RuntimeError:
        raise
    except Exception as exc:
        msg = f"Mellea case-name assessment failed: {exc}"
        raise RuntimeError(msg) from exc


def assess_case_name_with_mellea_sync(
    session: MelleaSession,
    *,
    citation_id: str,
    extracted_case_name: str | None,
    courtlistener_case_name: str | None,
    document_context: str,
) -> CaseNameAssessmentRun:
    """Run :func:`assess_case_name_with_mellea` from synchronous callers."""
    return _run_coroutine(
        assess_case_name_with_mellea(
            session,
            citation_id=citation_id,
            extracted_case_name=extracted_case_name,
            courtlistener_case_name=courtlistener_case_name,
            document_context=document_context,
        )
    )


async def _assess_case_name_with_mellea_after_exact(
    session: MelleaSession,
    *,
    citation_id: str,
    extracted_case_name: str | None,
    courtlistener_case_name: str,
    document_context: str,
) -> CaseNameAssessmentRun:
    """Classify the case name, then re-extract from local context when warranted."""
    deepseek_config = _deepseek_official_config()
    if deepseek_config is not None:
        return await _assess_case_name_with_deepseek_after_exact(
            deepseek_config,
            citation_id=citation_id,
            extracted_case_name=extracted_case_name,
            courtlistener_case_name=courtlistener_case_name,
            document_context=document_context,
        )

    if extracted_case_name:
        status = await _classify_case_name_with_mellea(
            session,
            local_context=document_context,
            extracted_case_name=extracted_case_name,
            retrieved_case_name=courtlistener_case_name,
        )
        if status == CaseNameAssessmentStatus.MATCH:
            return CaseNameAssessmentRun(
                assessment=_case_name_assessment(
                    citation_id, status, extracted_case_name, courtlistener_case_name
                ),
            )
    else:
        status = CaseNameAssessmentStatus.IRREGULAR_FORM

    reextraction = await reextract_case_name_with_mellea(
        session,
        document_context=document_context,
        extracted_case_name=extracted_case_name,
        courtlistener_case_name=courtlistener_case_name,
    )
    first_pass = _case_name_assessment(
        citation_id, status, extracted_case_name, courtlistener_case_name
    )
    return await _run_reassessment_after_reextraction(
        session,
        citation_id=citation_id,
        first_pass=first_pass,
        reextraction=reextraction,
        courtlistener_case_name=courtlistener_case_name,
        document_context=document_context,
    )


async def _run_reassessment_after_reextraction(
    session: MelleaSession,
    *,
    citation_id: str,
    first_pass: CaseNameAssessment,
    reextraction: ReextractionResult,
    courtlistener_case_name: str,
    document_context: str,
) -> CaseNameAssessmentRun:
    if reextraction.status == ReextractionStatus.EMPTY:
        return CaseNameAssessmentRun(assessment=first_pass)
    if reextraction.status != ReextractionStatus.ACCEPTED or reextraction.proposal is None:
        return CaseNameAssessmentRun(
            assessment=_case_name_assessment(
                citation_id,
                CaseNameAssessmentStatus.REEXTRACTION_ERROR,
                first_pass.extracted_case_name,
                courtlistener_case_name,
                message=(
                    "Case-name re-extraction failed programmatic validation after retries: "
                    f"{reextraction.error_message or 'unknown error'}"
                ),
            )
        )

    reassessment = await _assess_corrected_case_name(
        session,
        citation_id=citation_id,
        corrected_case_name=cast("str", reextraction.proposal.extracted_case_name),
        courtlistener_case_name=courtlistener_case_name,
        document_context=document_context,
    )
    return CaseNameAssessmentRun(
        assessment=first_pass,
        modified_citation=reextraction.proposal,
        reassessment=reassessment,
    )


async def _assess_corrected_case_name(
    session: MelleaSession,
    *,
    citation_id: str,
    corrected_case_name: str,
    courtlistener_case_name: str,
    document_context: str,
) -> CaseNameAssessment:
    """Assess a re-extracted case name, exact-first then model-backed."""
    exact = assess_case_name_exact_match(
        citation_id=citation_id,
        extracted_case_name=corrected_case_name,
        courtlistener_case_name=courtlistener_case_name,
    )
    if exact.status == CaseNameAssessmentStatus.EXACT_MATCH:
        return exact
    deepseek_config = _deepseek_official_config()
    if deepseek_config is not None:
        status = await _classify_case_name_with_deepseek(
            deepseek_config,
            document_context=document_context,
            extracted_case_name=corrected_case_name,
            courtlistener_case_name=courtlistener_case_name,
        )
        return _case_name_assessment(
            citation_id, status, corrected_case_name, courtlistener_case_name
        )
    status = await _classify_case_name_with_mellea(
        session,
        local_context=document_context,
        extracted_case_name=corrected_case_name,
        retrieved_case_name=courtlistener_case_name,
    )
    return _case_name_assessment(
        citation_id, status, corrected_case_name, courtlistener_case_name
    )


def _case_name_assessment(
    citation_id: str,
    status: CaseNameAssessmentStatus,
    extracted_case_name: str | None,
    courtlistener_case_name: str | None,
    *,
    message: str | None = None,
) -> CaseNameAssessment:
    return CaseNameAssessment(
        citation_id=citation_id,
        status=status,
        extracted_case_name=extracted_case_name,
        courtlistener_case_name=courtlistener_case_name,
        message=message or _message_for_status(status),
    )


async def _classify_case_name_with_mellea(
    session: MelleaSession,
    *,
    local_context: str,
    extracted_case_name: str,
    retrieved_case_name: str,
) -> CaseNameAssessmentStatus:
    """Classify a case name with Mellea resampling and JSON-parse retries."""
    last_error: Exception | None = None
    for _ in range(CASE_NAME_CLASSIFICATION_ATTEMPTS):
        try:
            verdict = await classify_case_name(
                session,
                local_context=local_context,
                extracted_case_name=extracted_case_name,
                retrieved_case_name=retrieved_case_name,
                strategy=CASE_NAME_CLASSIFICATION_STRATEGY,
                model_options=_structured_model_options(max_tokens=CASE_NAME_VERDICT_MAX_TOKENS),
            )
            return CaseNameAssessmentStatus(cast("str", verdict))
        except Exception as exc:
            if not _is_retryable_structured_output_error(exc):
                raise
            last_error = exc
    assert last_error is not None
    config = llm_provider_config_from_env(os.environ)
    msg = (
        "Mellea case-name classification failed after resampling. "
        "The model returned empty or invalid structured JSON before Mellea could parse "
        "ClassifyCaseNameResponse. This is commonly caused by the provider spending "
        "the completion budget on reasoning tokens and emitting no final JSON. "
        f"provider={config.provider.value}; model={config.model}; "
        f"api_base={config.api_base}; max_tokens={CASE_NAME_VERDICT_MAX_TOKENS}; "
        f"attempts={CASE_NAME_CLASSIFICATION_ATTEMPTS}; "
        f"mellea_loop_budget={CASE_NAME_CLASSIFICATION_STRATEGY.loop_budget}; "
        f"last_error={last_error}"
    )
    raise RuntimeError(msg) from last_error


def _is_retryable_structured_output_error(exc: Exception) -> bool:
    message = str(exc)
    return "Invalid JSON" in message or "EOF while parsing" in message or "input_value=''" in message


def _structured_model_options(*, max_tokens: int) -> dict[str, object]:
    return llm_provider_config_from_env(os.environ).mellea_call_options(max_tokens=max_tokens)


def _deepseek_official_config() -> LlmProviderConfig | None:
    config = llm_provider_config_from_env(os.environ)
    if config.provider == LlmProvider.DEEPSEEK:
        return config
    return None


async def _assess_case_name_with_deepseek_after_exact(
    config: LlmProviderConfig,
    *,
    citation_id: str,
    extracted_case_name: str | None,
    courtlistener_case_name: str,
    document_context: str,
) -> CaseNameAssessmentRun:
    if extracted_case_name:
        status = await _classify_case_name_with_deepseek(
            config,
            document_context=document_context,
            extracted_case_name=extracted_case_name,
            courtlistener_case_name=courtlistener_case_name,
        )
        if status == CaseNameAssessmentStatus.MATCH:
            return CaseNameAssessmentRun(
                assessment=_case_name_assessment(
                    citation_id, status, extracted_case_name, courtlistener_case_name
                ),
            )
    else:
        status = CaseNameAssessmentStatus.IRREGULAR_FORM

    reextraction = await reextract_case_name_with_deepseek(
        config,
        document_context=document_context,
        extracted_case_name=extracted_case_name,
        courtlistener_case_name=courtlistener_case_name,
    )
    first_pass = _case_name_assessment(
        citation_id, status, extracted_case_name, courtlistener_case_name
    )
    if reextraction.status == ReextractionStatus.EMPTY:
        return CaseNameAssessmentRun(assessment=first_pass)
    if reextraction.status != ReextractionStatus.ACCEPTED or reextraction.proposal is None:
        return CaseNameAssessmentRun(
            assessment=_case_name_assessment(
                citation_id,
                CaseNameAssessmentStatus.REEXTRACTION_ERROR,
                extracted_case_name,
                courtlistener_case_name,
                message=(
                    "Case-name re-extraction failed programmatic validation after retries: "
                    f"{reextraction.error_message or 'unknown error'}"
                ),
            )
        )

    reassessment = await _assess_corrected_case_name_with_deepseek(
        config,
        citation_id=citation_id,
        corrected_case_name=cast("str", reextraction.proposal.extracted_case_name),
        courtlistener_case_name=courtlistener_case_name,
        document_context=document_context,
    )
    return CaseNameAssessmentRun(
        assessment=first_pass,
        modified_citation=reextraction.proposal,
        reassessment=reassessment,
    )


async def _classify_case_name_with_deepseek(
    config: LlmProviderConfig,
    *,
    document_context: str,
    extracted_case_name: str,
    courtlistener_case_name: str,
) -> CaseNameAssessmentStatus:
    payload = await _deepseek_json_chat(
        config,
        messages=[
            {
                "role": "system",
                "content": (
                    "Classify how an extracted legal case name relates to a retrieved "
                    "CourtListener record. Return JSON only, for example "
                    '{"verdict":"match"}. The verdict must be one of: match, '
                    "different_case, irregular_form. Judge only the case name, not "
                    "volume, reporter, page, pin cite, court, or year."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"local_context:\n{document_context}\n\n"
                    f"extracted_case_name: {extracted_case_name}\n"
                    f"retrieved_case_name: {courtlistener_case_name}\n\n"
                    "Return JSON with exactly this shape: "
                    '{"verdict":"match|different_case|irregular_form"}'
                ),
            },
        ],
        max_tokens=CASE_NAME_VERDICT_MAX_TOKENS,
    )
    verdict = payload.get("verdict")
    if not isinstance(verdict, str):
        msg = "DeepSeek case-name classification did not return a string verdict."
        raise RuntimeError(msg)
    try:
        return CaseNameAssessmentStatus(verdict)
    except ValueError as exc:
        msg = f"DeepSeek case-name classification returned unsupported verdict: {verdict}"
        raise RuntimeError(msg) from exc


async def _assess_corrected_case_name_with_deepseek(
    config: LlmProviderConfig,
    *,
    citation_id: str,
    corrected_case_name: str,
    courtlistener_case_name: str,
    document_context: str,
) -> CaseNameAssessment:
    exact = assess_case_name_exact_match(
        citation_id=citation_id,
        extracted_case_name=corrected_case_name,
        courtlistener_case_name=courtlistener_case_name,
    )
    if exact.status == CaseNameAssessmentStatus.EXACT_MATCH:
        return exact
    status = await _classify_case_name_with_deepseek(
        config,
        document_context=document_context,
        extracted_case_name=corrected_case_name,
        courtlistener_case_name=courtlistener_case_name,
    )
    return _case_name_assessment(
        citation_id, status, corrected_case_name, courtlistener_case_name
    )


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
    last_content = ""
    for attempt in range(3):
        request_messages = messages
        if attempt:
            request_messages = [
                *messages,
                {
                    "role": "user",
                    "content": "Retry and return one non-empty JSON object only. Do not include prose.",
                },
            ]
        response = await client.chat.completions.create(
            model=config.model,
            messages=request_messages,  # type: ignore[arg-type]
            temperature=0,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
            extra_body={"thinking": {"type": "disabled"}},
        )
        content = response.choices[0].message.content or ""
        last_content = content
        if not content.strip():
            continue
        try:
            payload = json.loads(content)
        except json.JSONDecodeError as exc:
            msg = "DeepSeek JSON Output returned invalid JSON."
            raise RuntimeError(msg) from exc
        if not isinstance(payload, dict):
            msg = "DeepSeek JSON Output returned a non-object JSON value."
            raise RuntimeError(msg)
        return cast("dict[str, object]", payload)
    msg = f"DeepSeek JSON Output returned empty content after retries: {last_content!r}"
    raise RuntimeError(msg)


def _run_coroutine(coroutine):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)
    msg = "assess_case_name_with_mellea_sync cannot run inside an active event loop; use assess_case_name_with_mellea"
    raise RuntimeError(msg)


_STATUS_MESSAGES = {
    CaseNameAssessmentStatus.EXACT_MATCH: "Extracted case name exactly matches CourtListener.",
    CaseNameAssessmentStatus.MATCH: "Extracted case name matches the retrieved case.",
    CaseNameAssessmentStatus.DIFFERENT_CASE: (
        "Extracted case name refers to a different case than the retrieved record."
    ),
    CaseNameAssessmentStatus.IRREGULAR_FORM: (
        "Extracted case name uses an unusual or incomplete form for this case."
    ),
    CaseNameAssessmentStatus.REEXTRACTION_ERROR: "Case-name re-extraction failed.",
    CaseNameAssessmentStatus.NEEDS_ASSESSMENT: "Case name has not been assessed.",
}


def _message_for_status(status: CaseNameAssessmentStatus) -> str:
    return _STATUS_MESSAGES.get(status, "Case name has not been assessed.")
