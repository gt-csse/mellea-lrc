"""OpenAI-compatible API binding for Mellea-backed validation."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

LLM_MODEL_ENV = "MELLEA_LRC_LLM_MODEL"
LLM_API_BASE_ENV = "MELLEA_LRC_LLM_API_BASE"
LLM_API_KEY_ENV = "MELLEA_LRC_LLM_API_KEY"
LLM_TEMPERATURE_ENV = "MELLEA_LRC_LLM_TEMPERATURE"
LLM_RESPONSE_FORMAT_ENV = "MELLEA_LRC_LLM_RESPONSE_FORMAT"
LLM_CERT_REQUIRED_ENV = "MELLEA_LRC_LLM_CERT_REQUIRED"
DEFAULT_TEMPERATURE = 0.0
JSON_OBJECT_RESPONSE_FORMAT: dict[str, str] = {"type": "json_object"}


class LlmResponseFormat(str, Enum):
    """Structured-output modes supported by the API binding."""

    JSON_SCHEMA = "json_schema"
    JSON_OBJECT = "json_object"


@dataclass(frozen=True, slots=True)
class LlmApiConfig:
    """Resolved binding to an OpenAI-compatible API."""

    model: str
    api_base: str
    api_key: str = field(repr=False)
    temperature: float = DEFAULT_TEMPERATURE
    response_format: LlmResponseFormat = LlmResponseFormat.JSON_SCHEMA
    cert_required: bool = True

    def mellea_call_options(self, *, max_tokens: int, temperature: float = 0) -> dict[str, object]:
        """Build per-call Mellea model options for structured generation."""
        options: dict[str, object] = {"temperature": temperature, "max_tokens": max_tokens}
        if self.response_format == LlmResponseFormat.JSON_OBJECT:
            options["extra_body"] = {"response_format": JSON_OBJECT_RESPONSE_FORMAT}
        return options


def llm_api_config_from_env(environ: Mapping[str, str]) -> LlmApiConfig:
    """Resolve an OpenAI-compatible API binding from environment variables."""
    return LlmApiConfig(
        model=_required_env(environ, LLM_MODEL_ENV),
        api_base=_required_env(environ, LLM_API_BASE_ENV).rstrip("/"),
        api_key=_required_env(environ, LLM_API_KEY_ENV),
        temperature=_optional_float_env(environ, LLM_TEMPERATURE_ENV, DEFAULT_TEMPERATURE),
        response_format=_optional_response_format(environ),
        cert_required=_optional_bool_env(environ, LLM_CERT_REQUIRED_ENV, default=True),
    )


def start_mellea_session_from_env() -> object:
    """Start a Mellea session from the configured API binding."""
    try:
        from mellea import MelleaSession  # noqa: PLC0415
    except ImportError as exc:
        msg = "Mellea LLM dependencies are not installed."
        raise RuntimeError(msg) from exc

    from mellea_lrc.llm.openai_backend import MelleaLRCOpenAIBackend  # noqa: PLC0415

    config = llm_api_config_from_env(os.environ)
    backend = MelleaLRCOpenAIBackend(
        model_id=config.model,
        base_url=config.api_base,
        api_key=config.api_key,
        certificate_verification=config.cert_required,
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


def _optional_response_format(environ: Mapping[str, str]) -> LlmResponseFormat:
    value = environ.get(LLM_RESPONSE_FORMAT_ENV, "").strip()
    if not value:
        return LlmResponseFormat.JSON_SCHEMA
    try:
        return LlmResponseFormat(value)
    except ValueError as exc:
        allowed = ", ".join(item.value for item in LlmResponseFormat)
        msg = f"{LLM_RESPONSE_FORMAT_ENV} must be one of: {allowed}"
        raise RuntimeError(msg) from exc


def _optional_bool_env(
    environ: Mapping[str, str],
    name: str,
    *,
    default: bool,
) -> bool:
    value = environ.get(name, "").strip()
    if not value:
        return default
    if value.lower() in {"1", "true", "yes"}:
        return True
    if value.lower() in {"0", "false", "no"}:
        return False
    msg = f"{name} must be a boolean-compatible value"
    raise RuntimeError(msg)
