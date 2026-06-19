"""Remote smoke test for the OpenAI-compatible assessment model endpoint."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import requests
import pytest

pytestmark = pytest.mark.remote_smoke

REQUIRED_ENV_KEYS = (
    "MELLEA_LRC_ASSESSMENT_API_BASE",
    "MELLEA_LRC_ASSESSMENT_API_KEY",
    "MELLEA_LRC_ASSESSMENT_MODEL",
)


def test_assessment_provider_chat_completion(remote_timeout: float) -> None:
    config = _assessment_config_from_env_file()
    response = requests.post(
        _chat_completions_url(config["MELLEA_LRC_ASSESSMENT_API_BASE"]),
        headers={
            "Authorization": f"Bearer {config['MELLEA_LRC_ASSESSMENT_API_KEY']}",
            "Content-Type": "application/json",
        },
        json={
            "model": config["MELLEA_LRC_ASSESSMENT_MODEL"],
            "messages": [{"role": "user", "content": "Reply with OK only."}],
            "temperature": 0,
            "max_tokens": 8,
        },
        timeout=remote_timeout,
    )

    if response.status_code == 429:
        pytest.xfail(f"Assessment model endpoint is reachable but rate limited: {response.text}")

    assert response.status_code == 200, response.text
    payload = response.json()
    assert _first_message_content(payload)


def _assessment_config_from_env_file() -> dict[str, str]:
    values = {**os.environ, **_read_env_file(Path(".env"))}
    missing = [key for key in REQUIRED_ENV_KEYS if _is_unset(values.get(key))]
    if missing:
        pytest.skip(f"Set {', '.join(missing)} in .env to run assessment provider smoke test.")
    return {key: values[key] for key in REQUIRED_ENV_KEYS}


def _is_unset(value: str | None) -> bool:
    return not value or value.startswith("<") and value.endswith(">")


def _read_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        pytest.skip("Create .env to run assessment provider smoke test.")

    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        values[key.strip()] = value.strip().strip("\"'")
    return values


def _chat_completions_url(api_base: str) -> str:
    base = api_base.rstrip("/")
    if not base.endswith("/v1"):
        base = f"{base}/v1"
    return f"{base}/chat/completions"


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
