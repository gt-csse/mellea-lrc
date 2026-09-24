"""The external model binding is explicit and bounded."""

from __future__ import annotations

import pytest
from mellea.backends import ModelOption

from mellea_lrc.llm.config import (
    DEFAULT_TIMEOUT_SECONDS,
    LLM_SERVICE_TIER_ENV,
    LLM_TIMEOUT_SECONDS_ENV,
    llm_api_config_from_env,
)


def _environment(**extra: str) -> dict[str, str]:
    return {
        "MELLEA_LRC_LLM_MODEL": "z-ai/glm-5.3-flash",
        "MELLEA_LRC_LLM_API_BASE": "https://openrouter.ai/api/v1",
        "MELLEA_LRC_LLM_API_KEY": "test-key",
        **extra,
    }


def test_model_request_timeout_has_a_bounded_default() -> None:
    config = llm_api_config_from_env(_environment())

    assert config.timeout_seconds == DEFAULT_TIMEOUT_SECONDS
    assert config.service_tier is None
    assert "service_tier" not in config.mellea_call_options(max_tokens=32)


def test_model_request_timeout_can_be_set_in_the_environment() -> None:
    config = llm_api_config_from_env(_environment(**{LLM_TIMEOUT_SECONDS_ENV: "12.5"}))

    assert config.timeout_seconds == 12.5
    assert config.mellea_call_options(max_tokens=99)[ModelOption.STREAM_TIMEOUT] == 12.5


def test_model_request_timeout_must_be_numeric() -> None:
    with pytest.raises(RuntimeError, match=LLM_TIMEOUT_SECONDS_ENV):
        llm_api_config_from_env(_environment(**{LLM_TIMEOUT_SECONDS_ENV: "soon"}))


def test_service_tier_can_be_configured_and_rejects_unknown_values() -> None:
    config = llm_api_config_from_env(_environment(**{LLM_SERVICE_TIER_ENV: "priority"}))
    assert config.service_tier == "priority"

    with pytest.raises(RuntimeError, match=LLM_SERVICE_TIER_ENV):
        llm_api_config_from_env(_environment(**{LLM_SERVICE_TIER_ENV: "rush"}))
