"""Durable, JSON-native traces of model review attempts."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, JsonValue, StrictInt, model_validator

_TokenCount = Annotated[StrictInt, Field(ge=0)]


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
        """Read Chat Completions usage without estimating absent values."""
        if not isinstance(response, dict) or response.get("usage") is None:
            return None
        usage = response["usage"]
        if not isinstance(usage, dict):
            raise ValueError("Provider usage must be a JSON object")
        prompt = usage.get("prompt_tokens_details")
        completion = usage.get("completion_tokens_details")
        prompt = {} if prompt is None else prompt
        completion = {} if completion is None else completion
        if not isinstance(prompt, dict) or not isinstance(completion, dict):
            raise ValueError("Provider token details must be JSON objects or null")
        values = {
            "input_tokens": usage.get("prompt_tokens"),
            "output_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "cached_input_tokens": prompt.get("cached_tokens"),
            "cache_write_input_tokens": prompt.get("cache_write_tokens"),
            "reasoning_output_tokens": completion.get("reasoning_tokens"),
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

    @property
    def usage(self) -> TokenUsage | None:
        """A typed view of the retained response, including failed repairs."""
        return TokenUsage.from_response(self.response)


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
