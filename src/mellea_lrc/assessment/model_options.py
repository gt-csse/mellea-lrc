"""Shared Mellea call configuration."""

import os

from mellea_lrc.llm import llm_api_config_from_env


def structured_model_options(*, max_tokens: int) -> dict[str, object]:
    """Return structured-output options for one Mellea call."""
    return llm_api_config_from_env(os.environ).mellea_call_options(max_tokens=max_tokens)
