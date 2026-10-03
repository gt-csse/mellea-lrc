"""Project-owned Mellea instruct/validate/repair helpers."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from copy import copy
from dataclasses import dataclass, field
from enum import Enum
from threading import Lock
from typing import TYPE_CHECKING

from pydantic import BaseModel, JsonValue, ValidationError

from mellea_lrc.llm.config import DEFAULT_TIMEOUT_SECONDS, OUTPUT_MODE_OPTION, PROFILE_OPTION, LlmOutputMode
from mellea_lrc.model.ivr import IvrAttempt, IvrRequirementAttempt, IvrRun

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mellea import MelleaSession
    from mellea.core import ValidationResult
    from mellea.core.requirement import Requirement
    from mellea.core.sampling import SamplingResult
    from mellea.stdlib.sampling import MultiTurnStrategy


@dataclass(frozen=True, slots=True)
class InstructIvrSpec:
    """Complete project-level specification for one Mellea IVR instruction.

    Passing ``output_format`` installs one wrapper-owned schema requirement in
    every transport mode. ``json_schema`` also constrains native generation;
    ``json_object`` and ``prompt`` supply the schema in the instruction instead.
    Callers provide only domain requirements; they never need to parse Pydantic
    output merely to trigger a repair.
    """

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


async def run_instruct_ivr(
    session: MelleaSession,
    spec: InstructIvrSpec,
    *,
    strategy: MultiTurnStrategy,
    model_options: dict[str, object],
) -> IvrRun:
    """Run one IVR instruction and retain Mellea's inspectable run history."""
    from mellea.backends import ModelOption
    from mellea.stdlib import functional as mfuncs
    from mellea.stdlib.context import ChatContext

    if PROFILE_OPTION in model_options:
        profile = model_options[PROFILE_OPTION]
        if not isinstance(profile, Mapping):
            raise TypeError("IVR profile metadata must be a mapping")
        model_options = {
            **model_options,
            PROFILE_OPTION: {**profile, "max_attempts": strategy.loop_budget},
        }
    requirements = _requirements_for(spec)
    description, variables, wire_format, options = _output_transport(spec, model_options)
    local_strategy = copy(strategy)
    native_repair = local_strategy.repair
    completed: dict[int, tuple[object, IvrAttempt]] = {}
    history_lock = Lock()

    def capture_repair(old_ctx, new_ctx, past_actions, past_results, past_val):
        # The native hook runs after validation and before the next generation.
        # Copy provider data now, before later repair contexts can mutate it.
        with history_lock:
            for generation, attempt in zip(past_results, _attempts(past_results, past_val), strict=True):
                completed.setdefault(id(generation), (generation, attempt))
        return native_repair(old_ctx, new_ctx, past_actions, past_results, past_val)

    local_strategy.repair = capture_repair
    instruct = mfuncs.ainstruct(
        description,
        context=ChatContext(),
        backend=session.backend,
        grounding_context=dict(spec.grounding_context),
        user_variables=variables,
        requirements=requirements,
        strategy=local_strategy,
        return_sampling_results=True,
        await_result=True,
        format=wire_format,
        model_options=options if spec.prefix is None else {**options, ModelOption.SYSTEM_PROMPT: spec.prefix},
    )
    timeout_seconds = _call_timeout_seconds(model_options)
    try:
        # Mellea's stream timeout guards a missing chunk, but some provider
        # failures leave an open request after a stream has begun.  The outer
        # deadline turns that condition into an inspectable failed IVR run so
        # callers can defer safely and persist the reason.
        sampled = await asyncio.wait_for(instruct, timeout=timeout_seconds)
    except TimeoutError:
        with history_lock:
            attempts = tuple(attempt for _, attempt in completed.values())
        return _timed_out_ivr_run(session, spec, model_options, timeout_seconds, attempts)
    return _to_ivr_run(session, spec, model_options, sampled)


def _output_transport(
    spec: InstructIvrSpec, model_options: Mapping[str, object]
) -> tuple[str, dict[str, str], type[BaseModel] | None, dict[str, object]]:
    """Select the explicit wire format without weakening IVR validation.

    Some compatible endpoints implement JSON mode but not constrained schemas.
    The schema and domain requirements are mandatory in every transport mode;
    an endpoint error never silently changes modes or repairs output in code.
    """
    options = dict(model_options)
    options.pop(PROFILE_OPTION, None)  # Retained in IvrRun, never sent to the provider.
    mode = LlmOutputMode(options.pop(OUTPUT_MODE_OPTION, LlmOutputMode.JSON_SCHEMA))
    if "response_format" in options:
        raise ValueError("Configure output_mode instead of supplying response_format")
    if spec.output_format is None:
        if mode is not LlmOutputMode.JSON_SCHEMA:
            raise ValueError("Output transport mode requires a Pydantic output_format")
        return spec.description, dict(spec.user_variables), None, options
    if mode is LlmOutputMode.JSON_SCHEMA:
        return spec.description, dict(spec.user_variables), spec.output_format, options
    variable = "ivr_output_schema"
    if variable in spec.user_variables or variable in spec.grounding_context:
        raise ValueError(f"{variable} is reserved for IVR schema transport")
    variables = {**spec.user_variables, variable: json.dumps(spec.output_format.model_json_schema())}
    description = (
        spec.description + "\nReturn only one JSON object matching this schema:\n{{ivr_output_schema}}"
    )
    if mode is LlmOutputMode.JSON_OBJECT:
        options["response_format"] = {"type": "json_object"}
    return description, variables, None, options


_SCHEMA_REQUIREMENT = "Return exactly one JSON object matching the required output schema."
_CALL_TIMEOUT_REQUIREMENT = "The model response completed within the configured timeout."


def _call_timeout_seconds(model_options: Mapping[object, object]) -> float:
    """Read the project-owned whole-call limit from Mellea call options."""
    from mellea.backends import ModelOption

    value = model_options.get(ModelOption.STREAM_TIMEOUT, DEFAULT_TIMEOUT_SECONDS)
    if isinstance(value, int | float) and value > 0:
        return float(value)
    return DEFAULT_TIMEOUT_SECONDS


def _timed_out_ivr_run(
    session: MelleaSession,
    spec: InstructIvrSpec,
    model_options: Mapping[object, object],
    timeout_seconds: float,
    completed_attempts: tuple[IvrAttempt, ...] = (),
) -> IvrRun:
    """Represent an outer call deadline using the same artifact shape as IVR repair failure."""
    return IvrRun(
        success=False,
        selected_attempt=len(completed_attempts),
        attempts=(
            *completed_attempts,
            IvrAttempt(
                output="",
                requirements=(
                    IvrRequirementAttempt(
                        description=_CALL_TIMEOUT_REQUIREMENT,
                        passed=False,
                        reason=(f"Model call exceeded the configured {timeout_seconds:g}-second timeout."),
                        score=None,
                    ),
                ),
            ),
        ),
        **_run_metadata(session, spec, model_options),
    )


def _requirements_for(spec: InstructIvrSpec) -> list[Requirement]:
    """Install schema validation once and protect domain checks from bad JSON.

    Mellea evaluates all requirements concurrently. Without the guard, a
    domain validator that parses the output can raise beside the schema
    validator, replacing a useful repair with a provider failure or duplicate
    Pydantic diagnostics. The schema requirement is the only failed check
    until the response parses.
    """
    if spec.output_format is None:
        return list(spec.requirements)
    output_format = spec.output_format
    return [
        _schema_requirement(output_format),
        *(_guard_with_schema(requirement, output_format) for requirement in spec.requirements),
    ]


def _schema_requirement(output_format: type[BaseModel]) -> Requirement:
    """Create the one schema validation requirement for an IVR call."""
    from mellea.stdlib.requirements import req

    return req(
        _SCHEMA_REQUIREMENT,
        validation_fn=lambda ctx: _validate_schema(ctx, output_format),
    )


def _guard_with_schema(requirement: Requirement, output_format: type[BaseModel]) -> Requirement:
    """Do not invoke a parser-owning domain validator on malformed output."""
    from mellea.core import ValidationResult
    from mellea.core.requirement import Requirement

    if requirement.validation_fn is None:
        return requirement
    domain_validation = requirement.validation_fn

    def guarded(ctx: object) -> ValidationResult:
        if not _schema_matches(ctx, output_format):
            return ValidationResult(result=True)
        return domain_validation(ctx)

    return Requirement(
        description=requirement.description,
        validation_fn=guarded,
        output_to_bool=requirement.output_to_bool,
        check_only=requirement.check_only,
    )


def _validate_schema(ctx: object, output_format: type[BaseModel]) -> ValidationResult:
    """Validate generated JSON and give a concise repair instruction on failure."""
    from mellea.core import ValidationResult

    try:
        _parse_schema(ctx, output_format)
    except ValidationError as error:
        return ValidationResult(result=False, reason=_schema_error_message(error))
    return ValidationResult(result=True)


def _schema_matches(ctx: object, output_format: type[BaseModel]) -> bool:
    try:
        _parse_schema(ctx, output_format)
    except ValidationError:
        return False
    return True


def _parse_schema(ctx: object, output_format: type[BaseModel]) -> BaseModel:
    last_output = getattr(ctx, "last_output")()
    return output_format.model_validate_json(str(getattr(last_output, "value", last_output)))


def _schema_error_message(error: ValidationError) -> str:
    """Convert Pydantic diagnostics into short, model-actionable feedback."""
    details: list[str] = []
    priority = {"json_invalid": 0, "missing": 1, "extra_forbidden": 2}
    errors = sorted(
        error.errors(include_url=False),
        key=lambda item: priority.get(str(item["type"]), 3),
    )
    for item in errors[:4]:
        kind = str(item["type"])
        location = _schema_location(item.get("loc", ()))
        if kind == "json_invalid":
            details.append("The previous response was incomplete or invalid JSON.")
        elif kind == "missing":
            details.append(f"{location} is required.")
        elif kind == "extra_forbidden":
            details.append(f"Remove unsupported field {location}.")
        elif kind == "string_too_short":
            details.append(f"{location} must not be empty.")
        else:
            details.append(f"{location}: {item['msg']}.")
    suffix = " ".join(details) if details else "The previous response did not match the schema."
    return f"{_SCHEMA_REQUIREMENT} {suffix}"


def _schema_location(location: object) -> str:
    if not isinstance(location, tuple | list) or not location:
        return "The response"
    parts = [f"[{part}]" if isinstance(part, int) else str(part) for part in location]
    return f"`{'.'.join(parts)}`"


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
    attempts = _attempts(sampled.sample_generations, sampled.sample_validations)
    success = sampled.success
    selected_attempt = sampled.result_index

    return IvrRun(
        success=success,
        selected_attempt=selected_attempt,
        attempts=attempts,
        **_run_metadata(session, spec, model_options),
    )


def _run_metadata(
    session: MelleaSession,
    spec: InstructIvrSpec,
    model_options: Mapping[object, object],
) -> dict[str, object]:
    """Retain the same instruction provenance for completed and timed-out runs."""
    backend = session.backend
    output_format = spec.output_format
    return {
        "backend": type(backend).__qualname__,
        "model": _optional_string(getattr(backend, "model_id", None)),
        "model_options": _json_mapping(model_options),
        "instruction": spec.description,
        "prefix": spec.prefix,
        "grounding_context": dict(spec.grounding_context),
        "user_variables": dict(spec.user_variables),
        "output_schema": output_format.model_json_schema() if output_format is not None else None,
    }


def _attempts(
    generations: Sequence[object],
    validations: Sequence[Sequence[tuple[Requirement, ValidationResult]]],
) -> tuple[IvrAttempt, ...]:
    """Freeze completed native generations and their exact validation/provider data."""
    return tuple(
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
            request=_provider_request(generation),
            response=_provider_response(generation),
        )
        for index, generation in enumerate(generations)
    )


def _provider_request(generation: object) -> JsonValue:
    """Project the exact provider messages Mellea sent for one attempt.

    ``GenerateLog.prompt`` is the backend-level request after Mellea has added
    the initial instruction or its multi-turn repair feedback. The fallback is
    deliberately ``None`` for lightweight test doubles and non-logging
    backends; it never invents a request from the final answer.
    """
    log = getattr(generation, "_generate_log", None)
    return _json_value(getattr(log, "prompt", None))


def _provider_response(generation: object) -> JsonValue:
    """Project the provider response that explains completion and token use."""
    log = getattr(generation, "_generate_log", None)
    response = getattr(log, "model_output", None)
    if response is None:
        raw = getattr(generation, "raw", None)
        response = getattr(raw, "response", None)
    return _json_value(response)


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _json_mapping(values: Mapping[object, object]) -> dict[str, JsonValue]:
    """Keep public model options while refusing backend objects and secrets."""
    return {_json_key(key): _json_value(value) for key, value in values.items()}


def _json_key(value: object) -> str:
    if isinstance(value, Enum):
        return str(value.value)
    return str(value)


def _json_value(value: object) -> JsonValue:
    if isinstance(value, Enum):
        return _json_value(value.value)
    if value is None or isinstance(value, str | int | float | bool):
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _json_value(model_dump(mode="json"))
    if isinstance(value, Mapping):
        return _json_mapping(value)
    if isinstance(value, tuple | list):
        return [_json_value(item) for item in value]
    return str(value)
