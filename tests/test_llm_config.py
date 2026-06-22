"""Tests for shared LLM provider configuration."""

from mellea_lrc.llm import LlmProvider, llm_provider_config_from_env


def test_llm_provider_config_openrouter_mellea_call_options_respect_require_parameters() -> None:
    config = llm_provider_config_from_env(
        {
            "MELLEA_LRC_LLM_PROVIDER": "openrouter",
            "MELLEA_LRC_LLM_BACKEND": "openai",
            "MELLEA_LRC_LLM_MODEL": "openai/gpt-4.1-mini",
            "MELLEA_LRC_LLM_API_BASE": "https://openrouter.ai/api/v1",
            "MELLEA_LRC_LLM_API_KEY": "openrouter-key",
            "MELLEA_LRC_LLM_TEMPERATURE": "0",
            "MELLEA_LRC_LLM_REQUIRE_PARAMETERS": "0",
        }
    )

    assert config.provider == LlmProvider.OPENROUTER
    assert config.openrouter_require_parameters is False
    assert config.mellea_call_options(max_tokens=8) == {
        "temperature": 0,
        "max_tokens": 8,
    }
