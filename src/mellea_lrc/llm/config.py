"""OpenAI-compatible API binding for Mellea-backed validation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mellea import MelleaSession

OUTPUT_MODE_OPTION = "@@@mellea_lrc_output_mode@@@"
PROFILE_OPTION = "@@@mellea_lrc_profile@@@"
_REASONING_EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh"}


class LlmOutputMode(StrEnum):
    JSON_SCHEMA = "json_schema"
    JSON_OBJECT = "json_object"
    PROMPT = "prompt"


@dataclass(frozen=True, slots=True)
class LlmApiConfig:
    """Resolved binding to an OpenAI-compatible API."""

    model: str
    api_base: str
    api_key: str = field(repr=False)
    temperature: float
    timeout_seconds: float
    output_mode: LlmOutputMode
    service_tier: str | None = None
    reasoning_effort: str | None = None
    profile_name: str | None = None

    def __post_init__(self) -> None:
        for name in ("model", "api_base", "api_key"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonempty string")
        for name in ("temperature", "timeout_seconds"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite and numeric")
        if self.temperature < 0:
            raise ValueError("temperature must be nonnegative")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.service_tier is not None and self.service_tier not in {"default", "flex", "priority"}:
            raise ValueError("service_tier must be default, flex, priority, or None")
        if not isinstance(self.output_mode, LlmOutputMode):
            raise ValueError("output_mode must be an LlmOutputMode")
        if self.reasoning_effort is not None and (
            not isinstance(self.reasoning_effort, str) or self.reasoning_effort not in _REASONING_EFFORTS
        ):
            raise ValueError("reasoning_effort must be none, minimal, low, medium, high, xhigh, or None")

    def mellea_call_options(self, *, max_tokens: int) -> dict[str, object]:
        """Build bounded per-call Mellea options for structured generation.

        ``OpenAIBackend(timeout=...)`` bounds ordinary OpenAI requests.  Mellea
        applies a separate stream-chunk deadline, so carry the same configured
        timeout into that option as well.  This keeps a provider that opens a
        stream but never sends a chunk from holding an iterative review run.
        """
        from mellea.backends import ModelOption

        options: dict[str, object] = {
            "temperature": self.temperature,
            "max_tokens": max_tokens,
            ModelOption.STREAM_TIMEOUT: self.timeout_seconds,
            OUTPUT_MODE_OPTION: self.output_mode,
        }
        if self.profile_name is not None:
            options[PROFILE_OPTION] = {"name": self.profile_name, "endpoint": self.api_base}
        if self.service_tier is not None:
            options["service_tier"] = self.service_tier
        if self.reasoning_effort is not None:
            options["reasoning_effort"] = self.reasoning_effort
        return options


def start_mellea_session(config: LlmApiConfig) -> MelleaSession:
    """Start a session using the same resolved binding as its call options."""
    from mellea import MelleaSession
    from mellea.backends.openai import OpenAIBackend

    backend_options: dict[str, object] = {"temperature": config.temperature}
    if config.service_tier is not None:
        backend_options["service_tier"] = config.service_tier
    backend = OpenAIBackend(
        model_id=config.model,
        base_url=config.api_base,
        api_key=config.api_key,
        timeout=config.timeout_seconds,
        max_retries=0,
        model_options=backend_options,
    )
    return MelleaSession(backend)
