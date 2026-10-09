"""Contract tests for the exact CourtListener citation lookup client."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import parse_qs

import httpx
import pytest

from mellea_lrc.providers.courtlistener import (
    CourtListenerClient,
    CourtListenerConfig,
    CourtListenerConfigurationError,
    CourtListenerHTTPError,
    CourtListenerPayloadError,
    CourtListenerTransportError,
)


def _client(handler: httpx.MockTransport, *, pool: str | None = None) -> CourtListenerClient:
    return CourtListenerClient(
        CourtListenerConfig(
            base_url="https://proxy.example/api/rest/v4/", timeout_seconds=45, token="test-token", pool=pool
        ),
        http_client=httpx.Client(transport=handler),
    )


def test_lookup_posts_exact_locator_and_preserves_all_200_clusters() -> None:
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json=[
                {
                    "citation": "347 U.S. 483",
                    "status": 200,
                    "unexpected_result_field": "still available",
                    "clusters": [
                        {
                            "id": 1,
                            "caseName": "Brown v. Board of Education",
                            "caseNameFull": "Oliver Brown v. Board of Education of Topeka",
                            "dateFiled": "1954-05-17",
                            "court": "scotus",
                            "court_id": "scotus",
                            "docketId": 10,
                            "citations": [{"volume": 347, "reporter": "U.S.", "page": "483"}],
                            "unexpected_cluster_field": "still available",
                        },
                        {"id": 2, "case_name": "Another v. Board", "docket_id": "20"},
                    ],
                }
            ],
        )

    lookup = _client(httpx.MockTransport(respond), pool="reserved").lookup_citation("347", "U.S.", "483")

    assert lookup.citation == "347 U.S. 483"
    assert lookup.status == 200
    assert [cluster.id for cluster in lookup.clusters] == ["1", "2"]
    assert lookup.clusters[0].case_name_full == "Oliver Brown v. Board of Education of Topeka"
    assert lookup.clusters[0].date_filed == "1954-05-17"
    assert lookup.clusters[0].docket_id == "10"
    assert lookup.clusters[0].citations[0].reporter == "U.S."
    assert lookup.clusters[0].citations[0].volume == "347"
    assert lookup.clusters[0].raw_json["citations"][0]["volume"] == 347
    assert lookup.raw_json["unexpected_result_field"] == "still available"
    assert lookup.clusters[0].raw_json["unexpected_cluster_field"] == "still available"
    assert lookup.model_dump(mode="json")["clusters"][0]["id"] == "1"

    assert len(requests) == 1
    assert requests[0].method == "POST"
    assert str(requests[0].url) == "https://proxy.example/api/rest/v4/citation-lookup/"
    assert parse_qs(requests[0].content.decode()) == {
        "volume": ["347"],
        "reporter": ["U.S."],
        "page": ["483"],
    }
    assert requests[0].headers["authorization"] == "Token test-token"
    assert requests[0].headers["x-cl-pool"] == "reserved"


@pytest.mark.parametrize("status", [300, 400, 404])
def test_lookup_preserves_per_item_status(status: int) -> None:
    clusters = (
        [
            {"id": 10, "citations": [{"volume": 1, "reporter": "Handy", "page": "150", "type": 2}]},
            {"id": 20, "citations": [{"volume": 1, "reporter": "Haw.", "page": 150, "type": 2}]},
        ]
        if status == 300
        else []
    )
    handler = httpx.MockTransport(
        lambda _request: httpx.Response(
            200,
            json=[{"citation": "1 F.2d 2", "status": status, "clusters": clusters}],
        )
    )

    lookup = _client(handler).lookup_citation("1", "F.2d", "2")

    assert lookup.status == status
    assert len(lookup.clusters) == len(clusters)
    if status == 300:
        assert lookup.clusters[0].citations[0].volume == "1"
        assert lookup.clusters[1].citations[0].page == "150"


def test_http_and_transport_failures_are_typed() -> None:
    http_error = httpx.MockTransport(lambda _request: httpx.Response(429, json={"detail": "rate limited"}))
    with pytest.raises(CourtListenerHTTPError) as raised_http:
        _client(http_error).lookup_citation("1", "U.S.", "1")
    assert raised_http.value.upstream_status_code == 429

    def fail(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("proxy unreachable", request=request)

    with pytest.raises(CourtListenerTransportError):
        _client(httpx.MockTransport(fail)).lookup_citation("1", "U.S.", "1")


@pytest.mark.parametrize(
    "body",
    [
        {"citation": "1 U.S. 1", "status": 200},
        [],
        [{"citation": "1 U.S. 1", "status": 200}, {"citation": "2 U.S. 2", "status": 200}],
        [{"citation": "1 U.S. 1", "status": "200"}],
        [{"citation": "1 U.S. 1", "status": 200, "clusters": "wrong type"}],
    ],
)
def test_invalid_payload_is_not_treated_as_not_found(body: object) -> None:
    handler = httpx.MockTransport(lambda _request: httpx.Response(200, json=body))
    with pytest.raises(CourtListenerPayloadError):
        _client(handler).lookup_citation("1", "U.S.", "1")


def test_non_json_response_is_a_payload_error() -> None:
    handler = httpx.MockTransport(lambda _request: httpx.Response(200, text="not JSON"))
    with pytest.raises(CourtListenerPayloadError) as raised:
        _client(handler).lookup_citation("1", "U.S.", "1")
    assert raised.value.failure_type == "invalid_json"


def test_base_url_must_be_configured_without_public_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COURTLISTENER_BASE_URL", "https://ambient.example/")
    monkeypatch.setattr("mellea_lrc.providers.courtlistener.client.read_env", dict)

    with pytest.raises(CourtListenerConfigurationError, match="COURTLISTENER_BASE_URL") as raised:
        CourtListenerClient()
    assert raised.value.failure_type == "missing_base_url"


def test_default_proxy_config_does_not_send_saved_token_or_pool(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("COURTLISTENER_BASE_URL", "https://ambient.example/")
    (tmp_path / ".env").write_text(
        "COURTLISTENER_BASE_URL=https://proxy.example/api/rest/v4/\n"
        "COURTLISTENER_TIMEOUT_SECONDS=12.5\n"
        "COURTLISTENER_API_TOKEN=saved-token\n"
        "MELLEA_LRC_COURTLISTENER_POOL=reserved\n"
    )
    client = CourtListenerClient()
    try:
        assert client.config.base_url == "https://proxy.example/api/rest/v4/"
        assert client.config.timeout_seconds == 12.5
        assert client.config.token is None
        assert client.config.pool is None
        assert "Authorization" not in client._headers()
        assert "x-cl-pool" not in client._headers()
    finally:
        client.close()


def test_courtlistener_requires_timeout_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "mellea_lrc.providers.courtlistener.client.read_env",
        lambda: {"COURTLISTENER_BASE_URL": "https://proxy.example/"},
    )
    with pytest.raises(RuntimeError, match="COURTLISTENER_TIMEOUT_SECONDS"):
        CourtListenerConfig.from_env()


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf"), True, None, "45"])
def test_courtlistener_rejects_invalid_timeout(timeout) -> None:
    with pytest.raises(ValueError, match="COURTLISTENER_TIMEOUT_SECONDS"):
        CourtListenerConfig(base_url="https://proxy.example/", timeout_seconds=timeout)


@pytest.mark.parametrize(
    "url",
    [
        None,
        17,
        "",
        "proxy.example",
        "ftp://proxy.example/",
        "https://proxy.example/?q=x",
        "https://proxy.example/#x",
        "https://user:secret@proxy.example/",
        "https://proxy.example:70000/",
        "https://bad host/",
    ],
)
def test_courtlistener_rejects_invalid_base_url(url: str) -> None:
    with pytest.raises(CourtListenerConfigurationError, match="COURTLISTENER_BASE_URL") as raised:
        CourtListenerConfig(base_url=url, timeout_seconds=45)
    assert raised.value.failure_type == "invalid_base_url"


def test_courtlistener_requests_use_configured_timeout_and_config_repr_hides_token() -> None:
    requests = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=[{"citation": "1 U.S. 1", "status": 404, "clusters": []}])

    config = CourtListenerConfig(
        base_url="https://proxy.example/", timeout_seconds=9.5, token="private-token"
    )
    client = CourtListenerClient(config, http_client=httpx.Client(transport=httpx.MockTransport(respond)))
    client.lookup_citation("1", "U.S.", "1")
    assert requests[0].extensions["timeout"] == {"connect": 9.5, "read": 9.5, "write": 9.5, "pool": 9.5}
    assert "private-token" not in repr(config)
