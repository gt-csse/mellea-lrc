"""Tests for the shared OpenAI-compatible API binding."""

import asyncio

import httpx
import openai
import pytest
from mellea.core.base import ModelOutputThunk

from mellea_lrc.llm import (
    LlmResponseFormat,
    llm_api_config_from_env,
    start_mellea_session_from_env,
)
from mellea_lrc.llm.openai_backend import MelleaLRCOpenAIBackend

API_ENV = {
    "MELLEA_LRC_LLM_MODEL": "test-model",
    "MELLEA_LRC_LLM_API_BASE": "https://llm.example/v1",
    "MELLEA_LRC_LLM_API_KEY": "test-key",
}


def test_api_config_defaults_to_json_schema_and_certificate_verification() -> None:
    config = llm_api_config_from_env(API_ENV)

    assert config.model == "test-model"
    assert config.api_base == "https://llm.example/v1"
    assert config.temperature == 0
    assert config.response_format == LlmResponseFormat.JSON_SCHEMA
    assert config.cert_required is True
    assert config.mellea_call_options(max_tokens=8) == {"temperature": 0, "max_tokens": 8}
    assert "test-key" not in repr(config)


def test_api_config_can_request_json_object_output() -> None:
    config = llm_api_config_from_env({**API_ENV, "MELLEA_LRC_LLM_RESPONSE_FORMAT": "json_object"})

    assert config.response_format == LlmResponseFormat.JSON_OBJECT
    assert config.mellea_call_options(max_tokens=8) == {
        "temperature": 0,
        "max_tokens": 8,
        "extra_body": {"response_format": {"type": "json_object"}},
    }


def test_api_config_rejects_unknown_response_format() -> None:
    with pytest.raises(RuntimeError, match="json_schema, json_object"):
        llm_api_config_from_env({**API_ENV, "MELLEA_LRC_LLM_RESPONSE_FORMAT": "xml"})


def test_openai_client_uses_configured_certificate_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeHttpClient:
        def __init__(self, *, verify: bool) -> None:
            captured["verify"] = verify

    class FakeOpenAI:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(httpx, "Client", FakeHttpClient)
    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)
    config = llm_api_config_from_env({**API_ENV, "MELLEA_LRC_LLM_CERT_REQUIRED": "false"})

    config.openai_client()

    assert captured == {
        "api_key": "test-key",
        "base_url": "https://llm.example/v1",
        "http_client": captured["http_client"],
        "verify": False,
    }


@pytest.mark.parametrize("cert_required", [True, False])
def test_mellea_session_builds_distinct_sync_and_async_clients(
    monkeypatch: pytest.MonkeyPatch,
    cert_required: bool,
) -> None:
    for name, value in API_ENV.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("MELLEA_LRC_LLM_CERT_REQUIRED", str(cert_required))

    session = start_mellea_session_from_env()
    backend = session.backend

    assert isinstance(backend, MelleaLRCOpenAIBackend)
    assert backend.certificate_verification is cert_required
    assert isinstance(backend._client._client, httpx.Client)  # noqa: SLF001
    assert isinstance(backend._async_client._client, httpx.AsyncClient)  # noqa: SLF001

    backend._client.close()  # noqa: SLF001
    asyncio.run(backend._async_client.close())  # noqa: SLF001


def test_backend_post_processing_does_not_mask_request_errors() -> None:
    backend = object.__new__(MelleaLRCOpenAIBackend)
    output = ModelOutputThunk(None)

    asyncio.run(
        backend.post_processing(
            output,
            tools={},
            conversation=[],
            thinking=None,
            seed=None,
            _format=None,
        )
    )
