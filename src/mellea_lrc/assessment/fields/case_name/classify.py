"""Mellea case-name classification functions."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Literal, TypeVar

from mellea.core import ValidationResult
from mellea.core.base import Context
from mellea.core.requirement import Requirement
from mellea.stdlib.requirements import req
from mellea.stdlib.sampling import MultiTurnStrategy
from pydantic import BaseModel, ConfigDict, ValidationError

from mellea_lrc.llm import InstructIvrSpec, run_instruct_ivr

if TYPE_CHECKING:
    from mellea import MelleaSession

SemanticMatchVerdict = Literal["semantic_match", "not_semantic_match"]
CASE_NAME_VERDICT_MAX_TOKENS = 128
VERDICT_MAX_REPAIR_TURNS = 2
ModelT = TypeVar("ModelT", bound=BaseModel)

SEMANTIC_MATCH_INSTRUCTION = """
Classify whether extracted_case_name is a normal citation of retrieved_case_name.

Consider only the case names and local_context. A semantic match denotes the
same case using normal legal-citation abbreviation or party shortening, with
both sides of "v." represented. Missing, garbled, materially incomplete, or
different-case names are not semantic matches.

extracted_case_name:
{{extracted_case_name}}

retrieved_case_name:
{{retrieved_case_name}}
""".strip()


class _SemanticVerdictOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verdict: SemanticMatchVerdict


async def semantic_match_case_name(
    session: MelleaSession,
    *,
    local_context: str,
    extracted_case_name: str,
    retrieved_case_name: str,
    model_options: dict[str, object],
) -> SemanticMatchVerdict:
    """Classify whether an extracted name semantically matches the retrieved case."""
    spec = _verdict_spec(
        instruction=SEMANTIC_MATCH_INSTRUCTION,
        local_context=local_context,
        extracted_case_name=extracted_case_name,
        retrieved_case_name=retrieved_case_name,
        requirements=_semantic_requirements(),
    )
    result = await run_instruct_ivr(
        session,
        spec,
        strategy=MultiTurnStrategy(loop_budget=VERDICT_MAX_REPAIR_TURNS),
        model_options=model_options,
    )
    return _semantic_output_from_text(result.result.value).verdict


def _verdict_spec(
    *,
    instruction: str,
    local_context: str,
    extracted_case_name: str,
    retrieved_case_name: str,
    requirements: list[Requirement],
) -> InstructIvrSpec:
    return InstructIvrSpec(
        description=instruction,
        grounding_context={"local_context": local_context},
        user_variables={
            "extracted_case_name": extracted_case_name,
            "retrieved_case_name": retrieved_case_name,
        },
        requirements=requirements,
    )


def _semantic_requirements() -> list[Requirement]:
    return [
        req(
            'Return exactly one JSON object with shape {"verdict":"semantic_match or not_semantic_match"}.',
            validation_fn=_validate_semantic_output_schema,
        ),
    ]


def _semantic_output_from_text(output: str | object) -> _SemanticVerdictOutput:
    return _model_from_output(output, _SemanticVerdictOutput, "semantic verdict")


def _validate_semantic_output_schema(ctx: Context) -> ValidationResult:
    try:
        _semantic_output_from_text(ctx.last_output().value)
    except ValueError as exc:
        return ValidationResult(result=False, reason=str(exc))
    return ValidationResult(result=True)


def _model_from_output(
    output: str | object,
    model_type: type[ModelT],
    label: str,
) -> ModelT:
    if not isinstance(output, str):
        msg = f"LLM {label} output was not text: {type(output).__name__}"
        raise ValueError(msg)
    try:
        payload = json.loads(output)
    except json.JSONDecodeError as exc:
        msg = f"LLM {label} output was not valid JSON: {exc}"
        raise ValueError(msg) from exc
    if not isinstance(payload, dict):
        msg = f"LLM {label} output JSON was not an object"
        raise ValueError(msg)
    try:
        return model_type.model_validate(payload)
    except ValidationError as exc:
        msg = f"LLM {label} output did not match schema: {exc}"
        raise ValueError(msg) from exc
