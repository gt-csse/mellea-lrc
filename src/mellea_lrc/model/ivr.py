"""Durable, JSON-native traces of model review attempts."""

from __future__ import annotations

import json
from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, JsonValue, StrictInt, TypeAdapter, model_validator

_TokenCount = Annotated[StrictInt, Field(ge=0)]
_TOKEN_COUNT = TypeAdapter(_TokenCount)

SCHEMA_REQUIREMENT_DESCRIPTION = "Return exactly one JSON object matching the required output schema."
HTTP_TIMEOUT_REQUIREMENT_DESCRIPTION = "The provider HTTP request completed within its configured timeout."
HTTP_REQUEST_REQUIREMENT_DESCRIPTION = "The provider HTTP request completed successfully."


class IvrFailureKind(StrEnum):
    """Observable failure events; more than one can describe the same attempt."""

    HTTP_TIMEOUT = "http_timeout"
    CONNECTION_ERROR = "connection_error"
    HTTP_STATUS = "http_status"
    PROVIDER_ERROR = "provider_error"
    INVALID_JSON = "invalid_json"
    SCHEMA_VALIDATION = "schema_validation"
    REQUIREMENT_VALIDATION = "requirement_validation"
    OUTPUT_TRUNCATED = "output_truncated"
    REFUSAL = "refusal"
    CONTENT_FILTER = "content_filter"


class IvrTransportError(BaseModel):
    """A real SDK failure, without inventing a model answer or token usage."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: IvrFailureKind
    error_type: str = Field(min_length=1)
    message: str
    status_code: Annotated[StrictInt, Field(ge=100, le=599)] | None = None
    request_id: str | None = None

    @model_validator(mode="after")
    def check_transport_kind(self) -> IvrTransportError:
        if self.kind not in {
            IvrFailureKind.HTTP_TIMEOUT,
            IvrFailureKind.CONNECTION_ERROR,
            IvrFailureKind.HTTP_STATUS,
            IvrFailureKind.PROVIDER_ERROR,
        }:
            raise ValueError("Transport errors require a transport failure kind")
        return self


class TokenUsage(BaseModel):
    """Provider-reported counters; missing counters remain unknown.

    Cached input is included in input tokens. Reasoning output is included in
    output tokens, so neither should be added again when computing totals.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    input_tokens: _TokenCount | None = None
    output_tokens: _TokenCount | None = None
    total_tokens: _TokenCount | None = None
    cached_input_tokens: _TokenCount | None = None
    cache_write_input_tokens: _TokenCount | None = None
    reasoning_output_tokens: _TokenCount | None = None

    @classmethod
    def from_response(cls, response: JsonValue) -> TokenUsage | None:
        """Read standard Chat Completions or Responses usage without estimates.

        Compatible providers can retain both standard naming forms. Equal
        aliases are counted once; conflicting counters raise rather than
        silently choosing a value. Every counter remains provider-reported.
        """
        if not isinstance(response, dict) or response.get("usage") is None:
            return None
        usage = response["usage"]
        if not isinstance(usage, dict):
            raise ValueError("Provider usage must be a JSON object")
        details = {}
        for name in (
            "prompt_tokens_details",
            "completion_tokens_details",
            "input_tokens_details",
            "output_tokens_details",
        ):
            value = usage.get(name)
            if value is not None and not isinstance(value, dict):
                raise ValueError("Provider token details must be JSON objects or null")
            details[name] = value or {}

        def counter(name: str, *values: JsonValue) -> int | None:
            reported = [_TOKEN_COUNT.validate_python(value) for value in values if value is not None]
            if not reported:
                return None
            if any(value != reported[0] for value in reported[1:]):
                raise ValueError(f"Conflicting provider counters for {name}")
            return reported[0]

        prompt, completion = details["prompt_tokens_details"], details["completion_tokens_details"]
        inputs, outputs = details["input_tokens_details"], details["output_tokens_details"]
        values = {
            "input_tokens": counter("input_tokens", usage.get("prompt_tokens"), usage.get("input_tokens")),
            "output_tokens": counter(
                "output_tokens", usage.get("completion_tokens"), usage.get("output_tokens")
            ),
            "total_tokens": counter("total_tokens", usage.get("total_tokens")),
            "cached_input_tokens": counter(
                "cached_input_tokens", prompt.get("cached_tokens"), inputs.get("cached_tokens")
            ),
            "cache_write_input_tokens": counter(
                "cache_write_input_tokens", prompt.get("cache_write_tokens"), inputs.get("cache_write_tokens")
            ),
            "reasoning_output_tokens": counter(
                "reasoning_output_tokens", completion.get("reasoning_tokens"), outputs.get("reasoning_tokens")
            ),
        }
        return cls.model_validate(values) if any(value is not None for value in values.values()) else None


class IvrRequirementAttempt(BaseModel):
    """One requirement result Mellea recorded for one generated answer."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    description: str | None
    passed: bool
    reason: str | None
    score: float | None


class IvrAttempt(BaseModel):
    """One model answer, its provider exchange, and validation record.

    ``request`` is the exact provider message sequence for this attempt. On a
    repair turn it includes Mellea's user feedback containing the failed
    requirement reasons. ``response`` is the provider response Mellea retained,
    including finish reason and usage when the backend provides them. Both are
    projected to JSON-safe data so an artifact can explain a repair without a
    live Mellea session.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    output: str
    requirements: tuple[IvrRequirementAttempt, ...]
    request: JsonValue = None
    response: JsonValue = None
    transport_error: IvrTransportError | None = None

    @property
    def usage(self) -> TokenUsage | None:
        """A typed view of the retained response, including failed repairs."""
        return TokenUsage.from_response(self.response)

    @property
    def failure_kinds(self) -> tuple[IvrFailureKind, ...]:
        """Classify retained events, including failures later repaired successfully."""
        kinds: set[IvrFailureKind] = set()
        if self.transport_error is not None:
            kinds.add(self.transport_error.kind)
        for requirement in self.requirements:
            if requirement.passed:
                continue
            if requirement.description == SCHEMA_REQUIREMENT_DESCRIPTION:
                try:
                    json.loads(self.output)
                except (json.JSONDecodeError, ValueError):
                    kinds.add(IvrFailureKind.INVALID_JSON)
                else:
                    kinds.add(IvrFailureKind.SCHEMA_VALIDATION)
            elif requirement.description == HTTP_TIMEOUT_REQUIREMENT_DESCRIPTION:
                kinds.add(IvrFailureKind.HTTP_TIMEOUT)
            elif requirement.description != HTTP_REQUEST_REQUIREMENT_DESCRIPTION:
                kinds.add(IvrFailureKind.REQUIREMENT_VALIDATION)
        response = self.response
        if isinstance(response, dict):
            choices = response.get("choices")
            for choice in choices if isinstance(choices, list) else ():
                if not isinstance(choice, dict):
                    continue
                finish = choice.get("finish_reason")
                if finish == "length":
                    kinds.add(IvrFailureKind.OUTPUT_TRUNCATED)
                message = choice.get("message")
                if finish == "content_filter":
                    kinds.add(IvrFailureKind.CONTENT_FILTER)
                if isinstance(message, dict) and message.get("refusal"):
                    kinds.add(IvrFailureKind.REFUSAL)
            incomplete = response.get("incomplete_details")
            if isinstance(incomplete, dict) and incomplete.get("reason") == "max_output_tokens":
                kinds.add(IvrFailureKind.OUTPUT_TRUNCATED)
            if isinstance(incomplete, dict) and incomplete.get("reason") == "content_filter":
                kinds.add(IvrFailureKind.CONTENT_FILTER)
            output = response.get("output")
            for item in output if isinstance(output, list) else ():
                if not isinstance(item, dict):
                    continue
                content = item.get("content")
                if isinstance(content, list) and any(
                    isinstance(part, dict) and part.get("type") == "refusal" for part in content
                ):
                    kinds.add(IvrFailureKind.REFUSAL)
        return tuple(kind for kind in IvrFailureKind if kind in kinds)


class IvrRun(BaseModel):
    """Serializable account of an instruct/validate/repair run.

    Mellea already retains every generated answer, provider exchange, and
    requirement result. This project-owned projection keeps that information
    after the live session is gone, without retaining backend objects or
    credentials. In particular, later attempts retain Mellea's exact repair
    feedback rather than only the resulting validation status.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    success: bool
    selected_attempt: int
    attempts: tuple[IvrAttempt, ...]
    backend: str
    model: str | None
    model_options: dict[str, JsonValue]
    instruction: str
    prefix: str | None
    grounding_context: dict[str, str]
    user_variables: dict[str, str]
    output_schema: dict[str, JsonValue] | None
    elapsed_seconds: float | None = Field(default=None, ge=0, allow_inf_nan=False, strict=True)

    @model_validator(mode="after")
    def check_selected_attempt(self) -> IvrRun:
        if not self.attempts:
            msg = "An IVR run must retain at least one model attempt"
            raise ValueError(msg)
        if not -len(self.attempts) <= self.selected_attempt < len(self.attempts):
            msg = "The selected IVR attempt must be one of the retained attempts"
            raise ValueError(msg)
        return self

    @property
    def output(self) -> str:
        """The selected final output, including a failed run's last answer."""
        return self.attempts[self.selected_attempt].output

    @property
    def first_attempt_success(self) -> bool:
        return self.success and len(self.attempts) == 1

    @property
    def repaired(self) -> bool:
        return len(self.attempts) > 1

    @property
    def recovered_after_repair(self) -> bool:
        return self.success and self.repaired

    @property
    def failure_kinds(self) -> tuple[IvrFailureKind, ...]:
        """All observed categories, not a mutually exclusive outcome partition."""
        kinds = {kind for attempt in self.attempts for kind in attempt.failure_kinds}
        return tuple(kind for kind in IvrFailureKind if kind in kinds)

    @property
    def final_failure_kinds(self) -> tuple[IvrFailureKind, ...]:
        return () if self.success else self.attempts[self.selected_attempt].failure_kinds

    @property
    def failure_reason(self) -> str | None:
        """Explain selected-attempt validation failure and provider interruption.

        A provider can return partial JSON with ``content_filter`` or ``length``
        rather than a normal ``stop``. Preserve that diagnostic alongside the
        schema failure, deriving it from the retained response without another
        persisted field. Earlier failed attempts do not describe a recovered run.
        """
        attempt = self.attempts[self.selected_attempt]
        reason = next(
            (
                requirement.reason
                for requirement in attempt.requirements
                if not requirement.passed and requirement.reason
            ),
            None,
        )
        if self.success or not isinstance(attempt.response, dict):
            return reason
        choices = attempt.response.get("choices")
        if not isinstance(choices, list):
            return reason
        finishes = dict.fromkeys(
            choice["finish_reason"]
            for choice in choices
            if isinstance(choice, dict)
            and isinstance(choice.get("finish_reason"), str)
            and choice["finish_reason"] not in {"", "stop"}
        )
        if not finishes:
            return reason
        provider_reason = f"Provider finish reason: {', '.join(finishes)}."
        return f"{provider_reason} {reason}" if reason else provider_reason
