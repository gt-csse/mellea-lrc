"""OpenAI-compatible API binding for Mellea-backed validation."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

    from mellea import MelleaSession

LLM_MODEL_ENV = "MELLEA_LRC_LLM_MODEL"
LLM_API_BASE_ENV = "MELLEA_LRC_LLM_API_BASE"
LLM_API_KEY_ENV = "MELLEA_LRC_LLM_API_KEY"
LLM_TEMPERATURE_ENV = "MELLEA_LRC_LLM_TEMPERATURE"
LLM_TIMEOUT_SECONDS_ENV = "MELLEA_LRC_LLM_TIMEOUT_SECONDS"
LLM_SERVICE_TIER_ENV = "MELLEA_LRC_LLM_SERVICE_TIER"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TIMEOUT_SECONDS = 120.0
DEFAULT_SERVICE_TIER: str | None = None


@dataclass(frozen=True, slots=True)
class LlmApiConfig:
    """Resolved binding to an OpenAI-compatible API."""

    model: str
    api_base: str
    api_key: str = field(repr=False)
    temperature: float = DEFAULT_TEMPERATURE
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    service_tier: str | None = DEFAULT_SERVICE_TIER

    def mellea_call_options(self, *, max_tokens: int, temperature: float | None = None) -> dict[str, object]:
        """Build bounded per-call Mellea options for structured generation.

        ``OpenAIBackend(timeout=...)`` bounds ordinary OpenAI requests.  Mellea
        applies a separate stream-chunk deadline, so carry the same configured
        timeout into that option as well.  This keeps a provider that opens a
        stream but never sends a chunk from holding an iterative review run.
        """
        from mellea.backends import ModelOption

        options: dict[str, object] = {
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": max_tokens,
            ModelOption.STREAM_TIMEOUT: self.timeout_seconds,
        }
        if self.service_tier is not None:
            options["service_tier"] = self.service_tier
        return options


def llm_api_config_from_env(environ: Mapping[str, str]) -> LlmApiConfig:
    """Resolve an OpenAI-compatible API binding from environment variables."""
    return LlmApiConfig(
        model=_required_env(environ, LLM_MODEL_ENV),
        api_base=_required_env(environ, LLM_API_BASE_ENV).rstrip("/"),
        api_key=_required_env(environ, LLM_API_KEY_ENV),
        temperature=_optional_float_env(environ, LLM_TEMPERATURE_ENV, DEFAULT_TEMPERATURE),
        timeout_seconds=_optional_float_env(environ, LLM_TIMEOUT_SECONDS_ENV, DEFAULT_TIMEOUT_SECONDS),
        service_tier=_optional_choice_env(
            environ,
            LLM_SERVICE_TIER_ENV,
            DEFAULT_SERVICE_TIER,
            choices={"default", "flex", "priority"},
        ),
    )


def start_mellea_session_from_env() -> MelleaSession:
    """Start a Mellea session from the configured API binding."""
    from dotenv import load_dotenv
    from mellea import MelleaSession
    from mellea.backends.openai import OpenAIBackend

    load_dotenv(override=False)
    config = llm_api_config_from_env(os.environ)
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


def _required_env(environ: Mapping[str, str], name: str) -> str:
    value = environ.get(name, "").strip()
    if not value:
        msg = f"Missing required LLM configuration: {name}"
        raise RuntimeError(msg)
    return value


def _optional_float_env(environ: Mapping[str, str], name: str, default: float) -> float:
    value = environ.get(name, "").strip()
    if not value:
        return default
    try:
        return float(value)
    except ValueError as exc:
        msg = f"{name} must be a float-compatible value"
        raise RuntimeError(msg) from exc


def _optional_choice_env(
    environ: Mapping[str, str],
    name: str,
    default: str | None,
    *,
    choices: set[str],
) -> str | None:
    value = environ.get(name, "").strip().lower()
    if not value:
        return default
    if value not in choices:
        allowed = ", ".join(sorted(choices))
        msg = f"{name} must be one of: {allowed}"
        raise RuntimeError(msg)
    return value
