"""OpenAI-compatible LLM provider configuration for Mellea-backed stages."""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

LLM_PROVIDER_ENV = "MELLEA_LRC_LLM_PROVIDER"
LLM_MODEL_ENV = "MELLEA_LRC_LLM_MODEL"
LLM_API_BASE_ENV = "MELLEA_LRC_LLM_API_BASE"
LLM_API_KEY_ENV = "MELLEA_LRC_LLM_API_KEY"
LLM_TEMPERATURE_ENV = "MELLEA_LRC_LLM_TEMPERATURE"
LLM_OPENROUTER_REQUIRE_PARAMETERS_ENV = "MELLEA_LRC_LLM_OPENROUTER_REQUIRE_PARAMETERS"
DEFAULT_MELLEA_BACKEND = "openai"
DEFAULT_TEMPERATURE = 0.0
DEEPSEEK_DEFAULT_MODEL = "deepseek-v4-pro"
DEEPSEEK_DEFAULT_API_BASE = "https://api.deepseek.com"


class LlmProvider(str, Enum):
    """Supported OpenAI-compatible LLM providers."""

    DEEPSEEK = "deepseek"
    DIGITALOCEAN = "digitalocean"
    OPENROUTER = "openrouter"


@dataclass(frozen=True, slots=True)
class LlmProviderConfig:
    """Resolved provider configuration for Mellea LLM calls."""

    provider: LlmProvider
    backend: str
    model: str
    api_base: str
    api_key: str
    temperature: float
    openrouter_require_parameters: bool

    def mellea_call_options(self, *, max_tokens: int, temperature: float = 0) -> dict[str, object]:
        """Build per-call Mellea model options for structured generation."""
        options: dict[str, object] = {"temperature": temperature, "max_tokens": max_tokens}
        if self.provider == LlmProvider.OPENROUTER and self.openrouter_require_parameters:
            options["extra_body"] = {"provider": {"require_parameters": True}}
        return options

    def chat_completions_base_url(self) -> str:
        """Return the provider-specific OpenAI-compatible chat completions base URL."""
        if self.provider == LlmProvider.DEEPSEEK:
            return self.api_base.rstrip("/")
        return chat_completions_base_url(self.api_base)


def llm_provider_config_from_env(environ: Mapping[str, str]) -> LlmProviderConfig:
    """Resolve the active OpenAI-compatible LLM provider from env vars."""
    provider = _required_provider(environ)
    temperature = _optional_float_env(environ, LLM_TEMPERATURE_ENV, DEFAULT_TEMPERATURE)
    if provider == LlmProvider.OPENROUTER:
        return LlmProviderConfig(
            provider=provider,
            backend=DEFAULT_MELLEA_BACKEND,
            model=_required_env(environ, LLM_MODEL_ENV),
            api_base=_required_env(environ, LLM_API_BASE_ENV),
            api_key=_required_env(environ, LLM_API_KEY_ENV),
            temperature=temperature,
            openrouter_require_parameters=_optional_bool_env(
                environ,
                LLM_OPENROUTER_REQUIRE_PARAMETERS_ENV,
                False,
            ),
        )
    if provider == LlmProvider.DEEPSEEK:
        return LlmProviderConfig(
            provider=provider,
            backend=DEFAULT_MELLEA_BACKEND,
            model=_optional_env(environ, LLM_MODEL_ENV, DEEPSEEK_DEFAULT_MODEL),
            api_base=_optional_env(environ, LLM_API_BASE_ENV, DEEPSEEK_DEFAULT_API_BASE),
            api_key=_required_env(environ, LLM_API_KEY_ENV),
            temperature=temperature,
            openrouter_require_parameters=False,
        )
    if provider == LlmProvider.DIGITALOCEAN:
        return LlmProviderConfig(
            provider=provider,
            backend=DEFAULT_MELLEA_BACKEND,
            model=_required_env(environ, LLM_MODEL_ENV),
            api_base=_required_env(environ, LLM_API_BASE_ENV),
            api_key=_required_env(environ, LLM_API_KEY_ENV),
            temperature=temperature,
            openrouter_require_parameters=False,
        )

    msg = f"Unsupported LLM provider: {provider.value}"
    raise RuntimeError(msg)


def start_mellea_session_from_env() -> object:
    """Start a Mellea session from LLM provider environment variables."""
    try:
        from mellea import start_session  # noqa: PLC0415
    except ImportError as exc:
        msg = "Mellea LLM dependencies are not installed. Run with: uv sync --group llm"
        raise RuntimeError(msg) from exc

    config = llm_provider_config_from_env(os.environ)
    return start_session(
        config.backend,
        model_id=config.model,
        base_url=config.chat_completions_base_url(),
        api_key=config.api_key,
        model_options={"temperature": config.temperature},
    )


def chat_completions_base_url(base_url: str) -> str:
    """Normalize an OpenAI-compatible provider base URL to its `/v1` root."""
    normalized = base_url.rstrip("/")
    if normalized.endswith("/v1"):
        return normalized
    return f"{normalized}/v1"


def _required_provider(environ: Mapping[str, str]) -> LlmProvider:
    raw_provider = _required_env(environ, LLM_PROVIDER_ENV)
    try:
        return LlmProvider(raw_provider)
    except ValueError as exc:
        allowed = ", ".join(item.value for item in LlmProvider)
        msg = f"{LLM_PROVIDER_ENV} must be one of: {allowed}"
        raise RuntimeError(msg) from exc


def _required_env(environ: Mapping[str, str], name: str) -> str:
    value = environ.get(name, "").strip()
    if not value:
        msg = f"Missing required LLM configuration: {name}"
        raise RuntimeError(msg)
    return value


def _optional_env(environ: Mapping[str, str], name: str, default: str) -> str:
    value = environ.get(name, "").strip()
    return value or default


def _optional_float_env(environ: Mapping[str, str], name: str, default: float) -> float:
    value = environ.get(name, "").strip()
    if not value:
        return default
    try:
        return float(value)
    except ValueError as exc:
        msg = f"{name} must be a float-compatible value"
        raise RuntimeError(msg) from exc


def _optional_bool_env(environ: Mapping[str, str], name: str, default: bool) -> bool:
    value = environ.get(name, "").strip()
    if not value:
        return default
    if value.lower() in {"1", "true", "yes"}:
        return True
    if value.lower() in {"0", "false", "no"}:
        return False
    msg = f"{name} must be a boolean-compatible value"
    raise RuntimeError(msg)
