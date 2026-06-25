"""Mellea case-name classification."""

from __future__ import annotations

import os
from typing import Literal, cast

from mellea import MelleaSession, generative
from mellea.stdlib.sampling import RejectionSamplingStrategy

from mellea_lrc.assessment.types import CaseNameAssessmentStatus
from mellea_lrc.assessment.llm.options import structured_model_options
from mellea_lrc.llm import llm_provider_config_from_env

SemanticMatchVerdict = Literal["semantic_match", "not_semantic_match"]
NonSemanticVerdict = Literal["different_case", "irregular_form"]
CASE_NAME_VERDICT_MAX_TOKENS = 2048
CASE_NAME_CLASSIFICATION_ATTEMPTS = 3
CASE_NAME_CLASSIFICATION_STRATEGY = RejectionSamplingStrategy(loop_budget=3)


@generative
async def semantic_match_case_name(
    local_context: str,
    extracted_case_name: str,
    retrieved_case_name: str,
) -> SemanticMatchVerdict:
    """Decide whether an extracted case name is an acceptable citation of the retrieved case.

    Judge ONLY the case name using extracted_case_name, retrieved_case_name, and
    local_context. Do not consider volume, reporter, page, pin cite, court, or year.
    If extracted_case_name is <NO_EXTRACTED_CASE_NAME>, return "not_semantic_match".

    Return exactly one JSON object with key "result". The value must be one of:
    - "semantic_match": the two names denote the SAME case and the extracted form is a
      normal way to cite it. Accept abbreviation, party shortening, dropped suffixes,
      and ordinary citation style when BOTH sides of the "v." are represented.
    - "not_semantic_match": the names are not equivalent under that standard, including
      when the extracted name is missing, incomplete, garbled, or refers to a different case.

    Example: {"result": "semantic_match"}
    """


@generative
async def classify_non_semantic_case_name(
    local_context: str,
    extracted_case_name: str,
    retrieved_case_name: str,
) -> NonSemanticVerdict:
    """Classify a non-exact, non-semantic case name after re-extraction.

    Judge ONLY the case name. Return exactly one JSON object with key "result". The
    value must be one of:
    - "different_case": the extracted name denotes a DIFFERENT, unrelated case than the
      retrieved record.
    - "irregular_form": the names denote the SAME case, but the extracted name is
      genuinely incomplete or garbled beyond normal shortening.

    Example: {"result": "different_case"}
    """


async def is_semantic_match_with_mellea(
    session: MelleaSession,
    *,
    local_context: str,
    extracted_case_name: str,
    retrieved_case_name: str,
) -> bool:
    """Return whether Mellea accepts the extracted case name as a semantic match."""
    verdict = await _call_with_retries(
        session,
        semantic_match_case_name,
        local_context=local_context,
        extracted_case_name=extracted_case_name,
        retrieved_case_name=retrieved_case_name,
        failure_label="semantic match",
    )
    return verdict == "semantic_match"


async def classify_non_semantic_with_mellea(
    session: MelleaSession,
    *,
    local_context: str,
    extracted_case_name: str,
    retrieved_case_name: str,
) -> CaseNameAssessmentStatus:
    """Return ``different_case`` or ``irregular_form`` for a post-re-extraction name."""
    verdict = await _call_with_retries(
        session,
        classify_non_semantic_case_name,
        local_context=local_context,
        extracted_case_name=extracted_case_name,
        retrieved_case_name=retrieved_case_name,
        failure_label="non-semantic classification",
    )
    return CaseNameAssessmentStatus(cast("str", verdict))


async def _call_with_retries(
    session: MelleaSession,
    fn,
    *,
    local_context: str,
    extracted_case_name: str,
    retrieved_case_name: str,
    failure_label: str,
) -> str:
    last_error: Exception | None = None
    for _ in range(CASE_NAME_CLASSIFICATION_ATTEMPTS):
        try:
            verdict = await fn(
                session,
                local_context=local_context,
                extracted_case_name=extracted_case_name,
                retrieved_case_name=retrieved_case_name,
                strategy=CASE_NAME_CLASSIFICATION_STRATEGY,
                model_options=structured_model_options(max_tokens=CASE_NAME_VERDICT_MAX_TOKENS),
            )
            return cast("str", verdict)
        except Exception as exc:
            if not _is_retryable_structured_output_error(exc):
                raise
            last_error = exc
    assert last_error is not None
    config = llm_provider_config_from_env(os.environ)
    msg = (
        f"Mellea case-name {failure_label} failed after resampling. "
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
