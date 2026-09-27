"""Contract tests for CourtListener docket and opinion search requests."""

from __future__ import annotations

from urllib.parse import parse_qs, urlparse

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


def test_docket_search_sends_only_query_and_type_and_preserves_pagination_and_hits() -> None:
    requests: list[httpx.Request] = []
    next_url = (
        "https://www.courtlistener.com/api/rest/v4/search/?q=Smith+v.+Jones&type=d&cursor=cz0xJmQ9ZA%3D%3D"
    )

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "count": 2,
                "next": next_url,
                "previous": None,
                "results": [
                    {
                        "id": 10,
                        "docket_id": 20,
                        "docketNumber": "24-cv-1",
                        "caseName": "Smith v. Jones",
                        "caseNameFull": "Alex Smith v. Pat Jones",
                        "court_id": "nysd",
                        "dateFiled": "2024-01-02",
                        "unknown_field": {"keep": True},
                    },
                    {"docket_id": 21, "caseName": "Another v. Jones"},
                ],
                "unknown_page_field": "preserved",
            },
        )

    page = _client(httpx.MockTransport(respond)).search("Smith v. Jones", "d")

    assert page.count == 2
    assert page.next == next_url
    assert parse_qs(urlparse(page.next).query)["cursor"] == ["cz0xJmQ9ZA=="]
    assert page.previous is None
    assert len(page.results) == 2
    assert page.results[0].id == "10"
    assert page.results[0].docket_id == "20"
    assert page.results[0].docket_number == "24-cv-1"
    assert page.results[0].case_name == "Smith v. Jones"
    assert page.results[0].case_name_full == "Alex Smith v. Pat Jones"
    assert page.results[0].court_id == "nysd"
    assert page.results[0].date_filed == "2024-01-02"
    assert page.results[0].raw_json["unknown_field"] == {"keep": True}
    assert page.results[0].raw_json["docket_id"] == 20
    assert page.results[1].id is None
    assert page.raw_json["unknown_page_field"] == "preserved"

    assert len(requests) == 1
    assert requests[0].method == "GET"
    assert str(requests[0].url.copy_with(query=None)) == "https://proxy.example/api/rest/v4/search/"
    assert dict(requests[0].url.params) == {"q": "Smith v. Jones", "type": "d"}
    assert requests[0].headers["authorization"] == "Token test-token"
    assert requests[0].headers["x-cl-pool"] == "reserved"


def test_opinion_search_passes_cursor_without_other_filters() -> None:
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "count": 1,
                "next": None,
                "previous": "https://www.courtlistener.com/api/rest/v4/search/?q=Brown&type=o",
                "results": [
                    {
                        "cluster_id": 12,
                        "docket_id": 34,
                        "caseName": "Brown v. Board",
                        "court_id": "scotus",
                        "dateFiled": "1954-05-17",
                        "opinions": [{"id": 56}],
                    }
                ],
            },
        )

    page = _client(httpx.MockTransport(respond)).search("Brown", "o", cursor="a+/=")

    assert page.results[0].cluster_id == "12"
    assert page.results[0].docket_id == "34"
    assert page.results[0].raw_json["opinions"] == [{"id": 56}]
    assert page.previous == "https://www.courtlistener.com/api/rest/v4/search/?q=Brown&type=o"
    assert dict(requests[0].url.params) == {"q": "Brown", "type": "o", "cursor": "a+/="}


def test_recap_document_search_preserves_hit_and_passes_cursor() -> None:
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "count": 1,
                "next": None,
                "previous": None,
                "results": [
                    {
                        "id": 87,
                        "docket_id": 34,
                        "caseName": "Brown v. Board",
                        "description": "Memorandum opinion",
                        "plain_text": "The court holds...",
                    }
                ],
            },
        )

    page = _client(httpx.MockTransport(respond)).search("Brown", "rd", cursor="a+/=")

    assert page.results[0].id == "87"
    assert page.results[0].docket_id == "34"
    assert page.results[0].raw_json["description"] == "Memorandum opinion"
    assert page.results[0].raw_json["plain_text"] == "The court holds..."
    assert dict(requests[0].url.params) == {"q": "Brown", "type": "rd", "cursor": "a+/="}


@pytest.mark.parametrize("status", [404, 429, 500])
def test_search_http_errors_are_typed(status: int) -> None:
    client = _client(httpx.MockTransport(lambda _request: httpx.Response(status, text="unavailable")))

    with pytest.raises(CourtListenerHTTPError) as raised:
        client.search("Brown", "o")
    assert raised.value.upstream_status_code == status
    assert raised.value.failure_type == "http_error"


def test_search_transport_error_is_typed() -> None:
    def fail(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("proxy unreachable", request=request)

    with pytest.raises(CourtListenerTransportError):
        _client(httpx.MockTransport(fail)).search("Brown", "o")


@pytest.mark.parametrize(
    "body",
    [
        [],
        {"count": "1", "next": None, "previous": None, "results": []},
        {"count": 1, "next": None, "previous": None, "results": "wrong type"},
        {"count": 1, "next": None, "previous": None, "results": ["wrong type"]},
        {"count": 1, "next": None, "results": []},
    ],
)
def test_search_rejects_malformed_pages(body: object) -> None:
    client = _client(httpx.MockTransport(lambda _request: httpx.Response(200, json=body)))

    with pytest.raises(CourtListenerPayloadError) as raised:
        client.search("Brown", "o")
    assert raised.value.failure_type == "invalid_payload"


def test_search_rejects_invalid_json() -> None:
    client = _client(httpx.MockTransport(lambda _request: httpx.Response(200, text="not JSON")))

    with pytest.raises(CourtListenerPayloadError) as raised:
        client.search("Brown", "o")
    assert raised.value.failure_type == "invalid_json"


def test_search_rejects_unsupported_type_without_request() -> None:
    def fail_if_called(_request: httpx.Request) -> httpx.Response:
        pytest.fail("Unsupported search type must not reach HTTP client")

    with pytest.raises(ValueError):
        _client(httpx.MockTransport(fail_if_called)).search("Brown", "r")  # type: ignore[arg-type]
