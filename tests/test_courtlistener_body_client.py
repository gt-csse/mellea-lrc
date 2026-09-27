"""Contract tests for CourtListener opinion and RECAP document retrieval."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from mellea_lrc.courtlistener import (
    CourtListenerClient,
    CourtListenerConfig,
    CourtListenerHTTPError,
    CourtListenerPayloadError,
    CourtListenerTransportError,
)


def _client(handler: httpx.MockTransport) -> CourtListenerClient:
    return CourtListenerClient(
        CourtListenerConfig(
            base_url="https://proxy.example/api/rest/v4/",
            token="test-token",
            pool="reserved",
        ),
        http_client=httpx.Client(transport=handler),
    )


@pytest.mark.parametrize(
    ("method", "endpoint", "payload"),
    [
        (
            "get_opinion",
            "opinions",
            {
                "id": 123,
                "plain_text": "The court holds...",
                "html_with_citations": "<p>The court holds...</p>",
                "other_field": {"keep": True},
            },
        ),
        (
            "get_recap_document",
            "recap-documents",
            {
                "id": 123,
                "plain_text": "The motion is granted.",
                "ocr_status": 2,
                "other_field": ["keep"],
            },
        ),
    ],
)
def test_get_body_record_uses_proxy_and_preserves_full_json(
    method: str, endpoint: str, payload: dict[str, Any]
) -> None:
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=payload)

    actual = getattr(_client(httpx.MockTransport(respond)), method)("123")

    assert actual == payload
    assert len(requests) == 1
    assert requests[0].method == "GET"
    assert str(requests[0].url) == f"https://proxy.example/api/rest/v4/{endpoint}/123/"
    assert requests[0].headers["authorization"] == "Token test-token"
    assert requests[0].headers["x-cl-pool"] == "reserved"
    assert requests[0].headers["accept"] == "application/json"


@pytest.mark.parametrize("method", ["get_opinion", "get_recap_document"])
def test_get_body_record_returns_none_for_404(method: str) -> None:
    client = _client(httpx.MockTransport(lambda _request: httpx.Response(404)))

    assert getattr(client, method)("123") is None


@pytest.mark.parametrize("method", ["get_opinion", "get_recap_document"])
@pytest.mark.parametrize("status", [429, 500])
def test_get_body_record_preserves_http_error(method: str, status: int) -> None:
    client = _client(httpx.MockTransport(lambda _request: httpx.Response(status, text="unavailable")))

    with pytest.raises(CourtListenerHTTPError) as raised:
        getattr(client, method)("123")
    assert raised.value.failure_type == "http_error"
    assert raised.value.upstream_status_code == status


@pytest.mark.parametrize("method", ["get_opinion", "get_recap_document"])
def test_get_body_record_preserves_transport_error(method: str) -> None:
    def fail(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("proxy unreachable", request=request)

    with pytest.raises(CourtListenerTransportError) as raised:
        getattr(_client(httpx.MockTransport(fail)), method)("123")
    assert raised.value.failure_type == "transport_error"


@pytest.mark.parametrize("method", ["get_opinion", "get_recap_document"])
def test_get_body_record_rejects_invalid_json(method: str) -> None:
    client = _client(httpx.MockTransport(lambda _request: httpx.Response(200, text="not JSON")))

    with pytest.raises(CourtListenerPayloadError) as raised:
        getattr(client, method)("123")
    assert raised.value.failure_type == "invalid_json"


@pytest.mark.parametrize("method", ["get_opinion", "get_recap_document"])
@pytest.mark.parametrize("payload", [[], "wrong type", None])
def test_get_body_record_rejects_non_object_json(method: str, payload: object) -> None:
    client = _client(
        httpx.MockTransport(lambda _request: httpx.Response(200, content=json.dumps(payload).encode()))
    )

    with pytest.raises(CourtListenerPayloadError) as raised:
        getattr(client, method)("123")
    assert raised.value.failure_type == "invalid_payload"


@pytest.mark.parametrize("method", ["get_opinion", "get_recap_document"])
def test_get_body_record_rejects_non_numeric_id_without_request(method: str) -> None:
    def fail_if_called(_request: httpx.Request) -> httpx.Response:
        pytest.fail("Invalid record ID must not reach HTTP client")

    with pytest.raises(ValueError):
        getattr(_client(httpx.MockTransport(fail_if_called)), method)("../dockets/123")
