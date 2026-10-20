"""Project-owned Mellea instruct/validate/repair helpers."""

from __future__ import annotations

import json
from collections.abc import Mapping
from copy import copy
from dataclasses import dataclass, field
from enum import Enum
from threading import Lock
from typing import TYPE_CHECKING

from openai import APITimeoutError
from pydantic import BaseModel, JsonValue, ValidationError

from mellea_lrc.llm.config import OUTPUT_MODE_OPTION, PROFILE_OPTION, LlmOutputMode
from mellea_lrc.model.ivr import IvrAttempt, IvrRequirementAttempt, IvrRun

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mellea import MelleaSession
    from mellea.core import ValidationResult
    from mellea.core.requirement import Requirement
    from mellea.core.sampling import SamplingResult
    from mellea.stdlib.sampling import MultiTurnStrategy


class InvalidJsonResponseError(ValueError):
    """Generated JSON could not be parsed; preserve the parser's specific failure."""

    def __init__(self, parser_error: str) -> None:
        self.parser_error = parser_error
        super().__init__(
            "The previous response was incomplete or invalid JSON. "
            f"Parser error: {parser_error} "
            "PLEASE CHECK IF TOKEN LIMIT VIOLATED. "
            "Return a complete JSON object with all required fields and closing delimiters. "
            "Keep reasons concise, quote only necessary evidence, and avoid repetitive text."
        )


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
    description, variables, wire_format, options = _output_transport(spec, model_options)
    local_strategy = copy(strategy)
    native_repair = local_strategy.repair
    native_iteration = local_strategy._subsample_iteration
    http_timeout: APITimeoutError | None = None
    completed: dict[int, tuple[object, IvrAttempt]] = {}
    history_lock = Lock()
    pending: IvrAttempt | None = None
    pending_generation: object | None = None

    def retain_before_validation(validation, requirement):
        def retained(ctx):
            nonlocal pending, pending_generation
            generation = ctx.last_output()
            if generation is not None:
                # Retain an answer before validation, including a validation
                # that itself requires an HTTP request and might fail.
                with history_lock:
                    if generation is not pending_generation:
                        pending = _attempts((generation,), ())[0]
                        pending_generation = generation
            result = validation(ctx)
            with history_lock:
                if pending is not None and generation is pending_generation:
                    pending = pending.model_copy(
                        update={
                            "requirements": (
                                *pending.requirements,
                                IvrRequirementAttempt(
                                    description=requirement.description,
                                    passed=result.as_bool(),
                                    reason=result.reason,
                                    score=result.score,
                                ),
                            )
                        }
                    )
            return result

        return retained

    requirements = []
    for requirement in _requirements_for(spec):
        observed = copy(requirement)
        if observed.validation_fn is not None:
            observed.validation_fn = retain_before_validation(observed.validation_fn, observed)
        requirements.append(observed)

    def capture_repair(old_ctx, new_ctx, past_actions, past_results, past_val):
        nonlocal pending, pending_generation
        # The native hook runs after validation and before the next generation.
        # Copy provider data now, before later repair contexts can mutate it.
        with history_lock:
            for generation, attempt in zip(past_results, _attempts(past_results, past_val), strict=True):
                completed.setdefault(id(generation), (generation, attempt))
            pending = None
            pending_generation = None
        next_action, repair_context = native_repair(old_ctx, new_ctx, past_actions, past_results, past_val)
        if spec.output_format is not None and past_results:
            try:
                spec.output_format.model_validate_json(str(past_results[-1].value))
            except ValidationError as error:
                if any(detail["type"] == "json_invalid" for detail in error.errors(include_url=False)):
                    # Keep earlier conversation and the instruction that produced
                    # this answer, but do not feed malformed JSON back to the
                    # model. The full answer remains in the captured IVR trace.
                    # Valid JSON with schema or domain errors uses native repair.
                    repair_context = old_ctx.add(past_actions[-1])
        return next_action, repair_context

    local_strategy.repair = capture_repair

    async def capture_http_timeout(*args, **kwargs):
        nonlocal http_timeout
        try:
            async for sample in native_iteration(*args, **kwargs):
                yield sample
        except APITimeoutError as error:
            # Mellea 0.7 returns earlier failed samples when a later request
            # raises. Observe that error on this copied strategy so the trace
            # includes the interrupted request as well as completed repairs.
            http_timeout = error
            raise

    local_strategy._subsample_iteration = capture_http_timeout
    try:
        # Only the backend HTTP client enforces a time limit. Validation and
        # subsequent repairs do not share a review-wide deadline.
        sampled = await mfuncs.ainstruct(
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
            model_options=options
            if spec.prefix is None
            else {**options, ModelOption.SYSTEM_PROMPT: spec.prefix},
        )
    except APITimeoutError as error:
        http_timeout = error
        sampled = None
    if http_timeout is not None and (sampled is None or not sampled.success):
        with history_lock:
            attempts = tuple(attempt for _, attempt in completed.values())
        return _http_timeout_ivr_run(session, spec, model_options, http_timeout, attempts, pending)
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
    if OUTPUT_MODE_OPTION not in options:
        raise ValueError("IVR requires an explicit output_mode from its configured profile")
    mode = LlmOutputMode(options.pop(OUTPUT_MODE_OPTION))
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
_HTTP_TIMEOUT_REQUIREMENT = "The provider HTTP request completed within its configured timeout."


def _http_timeout_ivr_run(
    session: MelleaSession,
    spec: InstructIvrSpec,
    model_options: Mapping[object, object],
    error: APITimeoutError,
    completed_attempts: tuple[IvrAttempt, ...] = (),
    pending_attempt: IvrAttempt | None = None,
) -> IvrRun:
    """Retain an actual HTTP timeout and every preceding generation/repair."""
    return IvrRun(
        success=False,
        selected_attempt=len(completed_attempts),
        attempts=(
            *completed_attempts,
            IvrAttempt(
                output=pending_attempt.output if pending_attempt is not None else "",
                request=pending_attempt.request if pending_attempt is not None else None,
                response=pending_attempt.response if pending_attempt is not None else None,
                requirements=(
                    *(pending_attempt.requirements if pending_attempt is not None else ()),
                    IvrRequirementAttempt(
                        description=_HTTP_TIMEOUT_REQUIREMENT,
                        passed=False,
                        reason=f"{type(error).__name__}: {error}",
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
    except InvalidJsonResponseError as error:
        return ValidationResult(result=False, reason=f"{_SCHEMA_REQUIREMENT} {error}")
    except ValidationError as error:
        return ValidationResult(result=False, reason=_schema_error_message(error))
    return ValidationResult(result=True)


def _schema_matches(ctx: object, output_format: type[BaseModel]) -> bool:
    try:
        _parse_schema(ctx, output_format)
    except (InvalidJsonResponseError, ValidationError):
        return False
    return True


def _parse_schema(ctx: object, output_format: type[BaseModel]) -> BaseModel:
    last_output = getattr(ctx, "last_output")()
    try:
        return output_format.model_validate_json(str(getattr(last_output, "value", last_output)))
    except ValidationError as error:
        for detail in error.errors(include_url=False):
            if detail["type"] == "json_invalid":
                raise InvalidJsonResponseError(detail["msg"]) from error
        raise


def _schema_error_message(error: ValidationError) -> str:
    """Convert Pydantic diagnostics into short, model-actionable feedback."""
    details: list[str] = []
    priority = {"missing": 0, "extra_forbidden": 1}
    errors = sorted(
        error.errors(include_url=False),
        key=lambda item: priority.get(str(item["type"]), 3),
    )
    for item in errors[:4]:
        kind = str(item["type"])
        location = _schema_location(item.get("loc", ()))
        if kind == "missing":
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
    """Retain the response and the provider's separate streaming usage chunk.

    Mellea stores final stream usage in ``generation.usage``; its reconstructed
    response can omit that chunk. Merge those provider counters into the saved
    response without mutating Mellea's log or estimating interrupted usage.
    """
    log = getattr(generation, "_generate_log", None)
    response = getattr(log, "model_output", None)
    if response is None:
        raw = getattr(generation, "raw", None)
        response = getattr(raw, "response", None)
    projected = _json_value(response)
    metadata = getattr(generation, "generation", None)
    usage = _json_value(getattr(metadata, "usage", None))
    if isinstance(usage, dict) and (projected is None or isinstance(projected, dict)):
        return {**(projected or {}), "usage": usage}
    return projected


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
