"""Remote smoke test for the OpenAI-compatible LLM endpoint."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import requests
import pytest

from mellea_lrc.llm import (
    LlmProvider,
    LlmProviderConfig,
    llm_provider_config_from_env,
)

pytestmark = pytest.mark.remote_smoke


def test_llm_provider_chat_completion(remote_timeout: float) -> None:
    config = _llm_config_from_env_file()
    response = requests.post(
        f"{config.chat_completions_base_url()}/chat/completions",
        headers={
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        },
        json=_smoke_payload(config),
        timeout=remote_timeout,
    )

    if response.status_code == 429:
        pytest.xfail(f"LLM endpoint is reachable but rate limited: {response.text}")

    assert response.status_code == 200, response.text
    payload = response.json()
    assert _first_message_content(payload)


def _smoke_payload(config: LlmProviderConfig) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": config.model,
        "messages": [{"role": "user", "content": "Reply with OK only."}],
        "temperature": 0,
        "max_tokens": 64,
    }
    if config.provider == LlmProvider.DEEPSEEK:
        payload["thinking"] = {"type": "disabled"}
    return payload


def _llm_config_from_env_file() -> LlmProviderConfig:
    values = {**os.environ, **_read_env_file(Path(".env"))}
    values = {key: value for key, value in values.items() if not _is_unset(value)}
    try:
        return llm_provider_config_from_env(values)
    except RuntimeError as exc:
        pytest.skip(f"{exc} in .env to run LLM provider smoke test.")


def _is_unset(value: str | None) -> bool:
    return not value or value.startswith("<") and value.endswith(">")


def _read_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        pytest.skip("Create .env to run LLM provider smoke test.")

    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        values[key.strip()] = value.strip().strip("\"'")
    return values


def _first_message_content(payload: dict[str, Any]) -> str | None:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    first = choices[0]
    if not isinstance(first, dict):
        return None
    message = first.get("message")
    if not isinstance(message, dict):
        return None
    content = message.get("content")
    return content if isinstance(content, str) and content.strip() else None
