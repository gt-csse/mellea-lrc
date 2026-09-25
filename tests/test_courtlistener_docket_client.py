"""Contract tests for the CourtListener docket metadata request."""

from __future__ import annotations

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


def test_get_docket_reads_court_and_preserves_provider_response() -> None:
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "id": 266243,
                "court_id": "ca2",
                "court": "https://www.courtlistener.com/api/rest/v4/courts/ca2/",
                "docket_number": "77-1201",
            },
        )

    docket = _client(httpx.MockTransport(respond)).get_docket("266243")

    assert docket is not None
    assert docket.id == "266243"
    assert docket.court_id == "ca2"
    assert docket.court == "https://www.courtlistener.com/api/rest/v4/courts/ca2/"
    assert docket.raw_json["docket_number"] == "77-1201"
    assert len(requests) == 1
    assert requests[0].method == "GET"
    assert str(requests[0].url) == "https://proxy.example/api/rest/v4/dockets/266243/"
    assert requests[0].headers["authorization"] == "Token test-token"
    assert requests[0].headers["x-cl-pool"] == "reserved"
    assert requests[0].headers["accept"] == "application/json"


def test_get_docket_returns_none_for_404() -> None:
    client = _client(httpx.MockTransport(lambda _request: httpx.Response(404)))

    assert client.get_docket("266243") is None


@pytest.mark.parametrize("status", [429, 500])
def test_get_docket_raises_on_provider_errors(status: int) -> None:
    client = _client(httpx.MockTransport(lambda _request: httpx.Response(status, text="unavailable")))

    with pytest.raises(CourtListenerHTTPError) as raised:
        client.get_docket("266243")
    assert raised.value.upstream_status_code == status


def test_get_docket_rejects_invalid_response() -> None:
    client = _client(httpx.MockTransport(lambda _request: httpx.Response(200, json={"id": 22})))

    with pytest.raises(CourtListenerPayloadError) as raised:
        client.get_docket("266243")
    assert raised.value.failure_type == "invalid_payload"


def test_get_docket_raises_on_transport_failure() -> None:
    def fail(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("proxy unreachable", request=request)

    client = _client(httpx.MockTransport(fail))
    with pytest.raises(CourtListenerTransportError):
        client.get_docket("266243")


def test_get_docket_rejects_non_numeric_id_without_request() -> None:
    def fail_if_called(_request: httpx.Request) -> httpx.Response:
        pytest.fail("Invalid docket ID must not reach HTTP client")

    client = _client(httpx.MockTransport(fail_if_called))
    with pytest.raises(ValueError):
        client.get_docket("../clusters/123")
