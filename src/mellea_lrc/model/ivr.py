"""Durable, JSON-native traces of model review attempts."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, JsonValue, model_validator


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
