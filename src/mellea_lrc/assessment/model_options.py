"""Shared Mellea call configuration."""

import os

from mellea_lrc.llm import llm_provider_config_from_env


def structured_model_options(*, max_tokens: int) -> dict[str, object]:
    """Return provider-specific model options for one Mellea call."""
    return llm_provider_config_from_env(os.environ).mellea_call_options(max_tokens=max_tokens)
