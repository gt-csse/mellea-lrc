"""OpenAI-compatible API binding for Mellea-backed validation."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from mellea import MelleaSession
from mellea.backends.openai import OpenAIBackend

if TYPE_CHECKING:
    from collections.abc import Mapping

LLM_MODEL_ENV = "MELLEA_LRC_LLM_MODEL"
LLM_API_BASE_ENV = "MELLEA_LRC_LLM_API_BASE"
LLM_API_KEY_ENV = "MELLEA_LRC_LLM_API_KEY"
LLM_TEMPERATURE_ENV = "MELLEA_LRC_LLM_TEMPERATURE"
LLM_TIMEOUT_SECONDS_ENV = "MELLEA_LRC_LLM_TIMEOUT_SECONDS"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TIMEOUT_SECONDS = 45.0


@dataclass(frozen=True, slots=True)
class LlmApiConfig:
    """Resolved binding to an OpenAI-compatible API."""

    model: str
    api_base: str
    api_key: str = field(repr=False)
    temperature: float = DEFAULT_TEMPERATURE
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS

    def mellea_call_options(self, *, max_tokens: int, temperature: float = 0) -> dict[str, object]:
        """Build per-call Mellea model options for structured generation."""
        return {"temperature": temperature, "max_tokens": max_tokens}


def llm_api_config_from_env(environ: Mapping[str, str]) -> LlmApiConfig:
    """Resolve an OpenAI-compatible API binding from environment variables."""
    return LlmApiConfig(
        model=_required_env(environ, LLM_MODEL_ENV),
        api_base=_required_env(environ, LLM_API_BASE_ENV).rstrip("/"),
        api_key=_required_env(environ, LLM_API_KEY_ENV),
        temperature=_optional_float_env(environ, LLM_TEMPERATURE_ENV, DEFAULT_TEMPERATURE),
        timeout_seconds=_optional_float_env(
            environ, LLM_TIMEOUT_SECONDS_ENV, DEFAULT_TIMEOUT_SECONDS
        ),
    )


def start_mellea_session_from_env() -> MelleaSession:
    """Start a Mellea session from the configured API binding."""
    config = llm_api_config_from_env(os.environ)
    backend = OpenAIBackend(
        model_id=config.model,
        base_url=config.api_base,
        api_key=config.api_key,
        timeout=config.timeout_seconds,
        max_retries=0,
        model_options={"temperature": config.temperature},
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
