"""Project-owned Mellea instruct/validate/repair helpers."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from mellea.backends import ModelOption
from mellea.stdlib import functional as mfuncs
from mellea.stdlib.context import ChatContext

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mellea import MelleaSession
    from mellea.core.requirement import Requirement
    from mellea.core.sampling import SamplingResult
    from mellea.stdlib.sampling import MultiTurnStrategy
    from pydantic import BaseModel


@dataclass(frozen=True, slots=True)
class InstructIvrSpec:
    """Complete project-level specification for one Mellea IVR instruction."""

    description: str
    prefix: str | None = None
    """A long text several calls share, sent as the system message ahead of the instruction.

    Measured against the configured model through OpenRouter, the provider served a repeated
    opening from its prompt cache only as a system message: the same opinion sent inside the
    instruction's message, or as a user message before it, was billed in full on every call, and
    as the system message it was billed at the cached rate from the second call on. It is also sent as written, where the instruction's texts pass through
    template substitution."""
    grounding_context: Mapping[str, str] = field(default_factory=dict)
    user_variables: Mapping[str, str] = field(default_factory=dict)
    requirements: Sequence[Requirement] = field(default_factory=tuple)
    output_format: type[BaseModel] | None = None


@dataclass(frozen=True, slots=True)
class IvrRequirementAttempt:
    """One requirement result Mellea recorded for one generated answer."""

    description: str | None
    passed: bool
    reason: str | None
    score: float | None


@dataclass(frozen=True, slots=True)
class IvrAttempt:
    """One model answer and Mellea's complete validation record for it."""

    output: str
    requirements: tuple[IvrRequirementAttempt, ...]


@dataclass(frozen=True, slots=True)
class IvrRun:
    """Serializable account of an instruct/validate/repair run.

    Mellea already retains every generated answer and every requirement result.
    This project-owned projection keeps that information after the live session
    is gone, without retaining backend objects or credentials.
    """

    success: bool
    selected_attempt: int
    attempts: tuple[IvrAttempt, ...]
    backend: str
    model: str | None
    model_options: Mapping[str, object]
    instruction: str
    prefix: str | None
    grounding_context: Mapping[str, str]
    user_variables: Mapping[str, str]
    output_schema: Mapping[str, object] | None

    def __post_init__(self) -> None:
        if not self.attempts:
            msg = "An IVR run must retain at least one model attempt"
            raise ValueError(msg)
        if not -len(self.attempts) <= self.selected_attempt < len(self.attempts):
            msg = "The selected IVR attempt must be one of the retained attempts"
            raise ValueError(msg)

    @property
    def output(self) -> str:
        """The selected final output, including a failed run's last answer."""
        return self.attempts[self.selected_attempt].output

    @property
    def failure_reason(self) -> str | None:
        """The selected attempt's first failed requirement reason, when any."""
        return next(
            (
                requirement.reason
                for requirement in self.attempts[self.selected_attempt].requirements
                if not requirement.passed and requirement.reason
            ),
            None,
        )


async def run_instruct_ivr(
    session: MelleaSession,
    spec: InstructIvrSpec,
    *,
    strategy: MultiTurnStrategy,
    model_options: dict[str, object],
) -> IvrRun:
    """Run one IVR instruction and retain Mellea's inspectable run history."""
    sampled = await asyncio.to_thread(
        mfuncs.instruct,
        spec.description,
        context=ChatContext(),
        backend=session.backend,
        grounding_context=dict(spec.grounding_context),
        user_variables=dict(spec.user_variables),
        requirements=list(spec.requirements),
        strategy=strategy,
        return_sampling_results=True,
        format=spec.output_format,
        model_options=model_options
        if spec.prefix is None
        else {**model_options, ModelOption.SYSTEM_PROMPT: spec.prefix},
    )
    return _to_ivr_run(session, spec, model_options, sampled)


def _to_ivr_run(
    session: MelleaSession,
    spec: InstructIvrSpec,
    model_options: Mapping[object, object],
    sampled: SamplingResult[str] | object,
) -> IvrRun:
    """Project a Mellea ``SamplingResult`` into JSON-safe project data.

    The run retains the complete public history Mellea exposes. It deliberately
    accepts no reduced result shape: a result without attempt history could not
    explain a failed review without repeating the model call.
    """
    generations = sampled.sample_generations
    validations = sampled.sample_validations
    attempts = tuple(
        IvrAttempt(
            output=str(generation.value),
            requirements=tuple(
                IvrRequirementAttempt(
                    description=requirement.description,
                    passed=validation.as_bool(),
                    reason=validation.reason,
                    score=validation.score,
                )
                for requirement, validation in validations[index]
            )
            if index < len(validations)
            else (),
        )
        for index, generation in enumerate(generations)
    )
    success = sampled.success
    selected_attempt = sampled.result_index

    backend = session.backend
    output_format = spec.output_format
    return IvrRun(
        success=success,
        selected_attempt=selected_attempt,
        attempts=attempts,
        backend=type(backend).__qualname__,
        model=_optional_string(getattr(backend, "model_id", None)),
        model_options=_json_mapping(model_options),
        instruction=spec.description,
        prefix=spec.prefix,
        grounding_context=dict(spec.grounding_context),
        user_variables=dict(spec.user_variables),
        output_schema=(output_format.model_json_schema() if output_format is not None else None),
    )


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _json_mapping(values: Mapping[object, object]) -> dict[str, object]:
    """Keep public model options while refusing backend objects and secrets."""
    return {_json_key(key): _json_value(value) for key, value in values.items()}


def _json_key(value: object) -> str:
    if isinstance(value, Enum):
        return str(value.value)
    return str(value)


def _json_value(value: object) -> object:
    if isinstance(value, Enum):
        return _json_value(value.value)
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Mapping):
        return _json_mapping(value)
    if isinstance(value, tuple | list):
        return [_json_value(item) for item in value]
    return str(value)
