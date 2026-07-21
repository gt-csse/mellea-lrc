"""Mellea-backed semantic case-name check."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Literal

from mellea.core import ValidationResult
from mellea.stdlib.requirements import req
from mellea.stdlib.sampling import MultiTurnStrategy
from pydantic import BaseModel, ConfigDict, ValidationError

from mellea_lrc.llm import (
    InstructIvrSpec,
    llm_api_config_from_env,
    run_instruct_ivr,
    start_mellea_session_from_env,
)

if TYPE_CHECKING:
    from mellea import MelleaSession
    from mellea.core.base import Context

SemanticMatchVerdict = Literal["semantic_match", "not_semantic_match"]
CASE_NAME_VERDICT_MAX_TOKENS = 128
VERDICT_MAX_REPAIR_TURNS = 2

SEMANTIC_MATCH_INSTRUCTION = """
Classify whether extracted_case_name is a normal citation of retrieved_case_name.

Consider only the case names. A semantic match denotes the same case using
normal legal-citation abbreviation or party shortening, with both sides of
"v." represented. Missing, garbled, materially incomplete, or different-case
names are not semantic matches.

extracted_case_name:
{{extracted_case_name}}

retrieved_case_name:
{{retrieved_case_name}}
""".strip()


class _SemanticVerdictOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verdict: SemanticMatchVerdict


async def mellea_case_names_match(
    extracted_case_name: str,
    retrieved_case_name: str,
    *,
    session: MelleaSession | None = None,
) -> bool:
    """Return whether Mellea judges two otherwise-unmatched case names equivalent."""
    resolved_session = session or start_mellea_session_from_env()
    model_options = llm_api_config_from_env(os.environ).mellea_call_options(
        max_tokens=CASE_NAME_VERDICT_MAX_TOKENS
    )
    spec = InstructIvrSpec(
        description=SEMANTIC_MATCH_INSTRUCTION,
        user_variables={
            "extracted_case_name": extracted_case_name,
            "retrieved_case_name": retrieved_case_name,
        },
        requirements=[
            req(
                'Return exactly one JSON object with shape {"verdict":"semantic_match or not_semantic_match"}.',
                validation_fn=_validate_semantic_output_schema,
            ),
        ],
    )
    result = await run_instruct_ivr(
        resolved_session,
        spec,
        strategy=MultiTurnStrategy(loop_budget=VERDICT_MAX_REPAIR_TURNS),
        model_options=model_options,
    )
    if not result.success:
        msg = "Semantic case-name classifier exhausted retries without satisfying requirements."
        raise ValueError(msg)
    return _semantic_output_from_text(result.result.value).verdict == "semantic_match"


def _validate_semantic_output_schema(ctx: Context) -> ValidationResult:
    """Require a valid semantic-verdict JSON response before accepting it."""
    try:
        _semantic_output_from_text(ctx.last_output().value)
    except (TypeError, ValueError) as exc:
        return ValidationResult(result=False, reason=str(exc))
    return ValidationResult(result=True)


def _semantic_output_from_text(output: str | object) -> _SemanticVerdictOutput:
    try:
        return _SemanticVerdictOutput.model_validate_json(output)
    except ValidationError as exc:
        msg = f"LLM semantic verdict output was invalid: {exc}"
        raise ValueError(msg) from exc
