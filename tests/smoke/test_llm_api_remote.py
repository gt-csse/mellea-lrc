"""LLM remote sanity test for the configured OpenAI-compatible API."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from openai import RateLimitError

from mellea_lrc.core.env import read_env_file
from mellea_lrc.llm import LlmApiConfig, llm_api_config_from_env

pytestmark = [pytest.mark.remote_smoke, pytest.mark.llm_remote_sanity]


def test_llm_api_chat_completion(remote_timeout: float) -> None:
    config = _llm_config_from_env_file()
    try:
        with config.openai_client() as client:
            response = client.chat.completions.create(
                model=config.model,
                messages=[{"role": "user", "content": "Reply with OK only."}],
                temperature=0,
                max_tokens=64,
                timeout=remote_timeout,
            )
    except RateLimitError as exc:
        pytest.xfail(f"LLM API is reachable but rate limited: {exc}")

    assert response.choices
    assert response.choices[0].message.content


def _llm_config_from_env_file() -> LlmApiConfig:
    # Explicit process configuration must override local development defaults.
    values = {**_read_env_file(Path(".env")), **os.environ}
    values = {key: value for key, value in values.items() if not _is_unset(value)}
    try:
        return llm_api_config_from_env(values)
    except RuntimeError as exc:
        pytest.skip(f"{exc} in .env to run the LLM API remote sanity test.")


def _is_unset(value: str | None) -> bool:
    return not value or (value.startswith("<") and value.endswith(">"))


def _read_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        pytest.skip("Create .env to run the LLM API remote sanity test.")
    return read_env_file(path)
