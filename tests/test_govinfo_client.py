"""Contract tests for the GovInfo package search client."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest
from dotenv import load_dotenv

from mellea_lrc.providers.govinfo import (
    GovInfoClient,
    GovInfoConfig,
    GovInfoError,
    GovInfoGranulesPage,
    govinfo_uscourts_docket_query,
)


class _Response:
    status_code = 200
    url = "https://api.govinfo.gov/search?api_key=test-secret"
    text = ""

    def __init__(self, payload: object, status_code: int = 200) -> None:
        self.payload = payload
        self.status_code = status_code

    def json(self) -> object:
        return self.payload


class _Session:
    def __init__(self, response: _Response | None = None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def post(self, url: str, **kwargs: object) -> _Response:
        self.calls.append({"url": url, **kwargs})
        if self.error:
            raise self.error
        assert self.response is not None
        return self.response

    def close(self) -> None:
        pass


def test_govinfo_search_sends_package_query_and_returns_raw_page() -> None:
    payload = {"count": 1, "offsetMark": "next-token", "results": [{"packageId": "USCOURTS-nysd-x"}]}
    session = _Session(_Response(payload))
    client = GovInfoClient(GovInfoConfig(api_key="test-secret"), session=session)  # type: ignore[arg-type]

    result = client.search('collection:uscourts casenumber:("1:24-cv-123")')

    assert result.raw_json == payload
    assert result.results == tuple(payload["results"])
    assert result.count == 1
    assert result.next_offset_mark == "next-token"
    assert session.calls == [
        {
            "url": "https://api.govinfo.gov/search",
            "params": {"api_key": "test-secret"},
            "json": {
                "query": 'collection:uscourts casenumber:("1:24-cv-123")',
                "pageSize": 100,
                "offsetMark": "*",
                "resultLevel": "package",
                "sorts": [{"field": "score", "sortOrder": "DESC"}],
            },
            "headers": {
                "Accept": "application/json",
                "User-Agent": "mellea-lrc (+https://github.com/gt-csse/mellea-lrc)",
            },
            "timeout": 45,
        }
    ]


def test_govinfo_search_can_request_default_granule_results() -> None:
    payload = {
        "count": 1,
        "results": [{"packageId": "USCOURTS-nyd-1_24-cv-123", "granuleId": "opinion-1"}],
    }
    session = _Session(_Response(payload))
    client = GovInfoClient(GovInfoConfig(api_key="test-secret"), session=session)  # type: ignore[arg-type]

    result = client.search('collection:uscourts and "550 U.S. 544"', result_level="default")

    assert result.results[0]["granuleId"] == "opinion-1"
    assert session.calls[0]["json"]["resultLevel"] == "default"
    with pytest.raises(ValueError, match="result level"):
        client.search("query", result_level="granule")  # type: ignore[arg-type]
    assert len(session.calls) == 1


def test_govinfo_key_uses_environment_and_redacts_error_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOVINFO_API_KEY", "secret-value")
    session = _Session(_Response({"message": "secret-value"}, status_code=403))
    client = GovInfoClient(session=session)  # type: ignore[arg-type]

    with pytest.raises(GovInfoError) as raised:
        client.search("query")

    error = raised.value
    assert error.failure_type == "http_error"
    assert error.upstream_status_code == 403
    assert "secret-value" not in str(error)
    assert "secret-value" not in (error.url or "")
    assert "secret-value" not in repr(error.upstream_detail)


def test_govinfo_client_uses_demo_key_when_environment_is_empty(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GOVINFO_API_KEY", "")
    session = _Session(_Response({"count": 0, "results": []}))
    GovInfoClient(session=session).search("query")  # type: ignore[arg-type]

    assert session.calls[0]["params"] == {"api_key": "DEMO_KEY"}


def test_govinfo_config_loads_key_from_dotenv_without_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GOVINFO_API_KEY", "")
    (tmp_path / ".env").write_text("GOVINFO_API_KEY=dotenv-secret\n")
    load_dotenv(dotenv_path=tmp_path / ".env", override=True)

    assert GovInfoConfig.from_env().api_key == "dotenv-secret"


@pytest.mark.parametrize("payload", [[], {"count": "one", "results": []}, {"count": 0, "results": ["bad"]}])
def test_govinfo_rejects_invalid_search_pages(payload: object) -> None:
    client = GovInfoClient(GovInfoConfig(api_key="x"), session=_Session(_Response(payload)))  # type: ignore[arg-type]

    with pytest.raises(GovInfoError, match="invalid"):
        client.search("query")


def test_govinfo_transport_error_is_structured_and_redacted() -> None:
    session = _Session(
        error=httpx.ConnectError(
            "request to https://api.govinfo.gov/?api_key=secret failed",
            request=httpx.Request("POST", "https://api.govinfo.gov/search?api_key=secret"),
        )
    )
    client = GovInfoClient(GovInfoConfig(api_key="secret"), session=session)  # type: ignore[arg-type]

    with pytest.raises(GovInfoError) as raised:
        client.search("query")

    assert raised.value.failure_type == "transport_error"
    assert "secret" not in str(raised.value)
    assert "secret" not in repr(raised.value.upstream_detail)


def test_govinfo_docket_query_keeps_literal_number() -> None:
    assert govinfo_uscourts_docket_query("20 Civ. 6835") == (
        'collection:uscourts casenumber:("20 Civ. 6835")'
    )
    assert govinfo_uscourts_docket_query('X " y') == ('collection:uscourts casenumber:("X \\" y")')


def test_govinfo_granules_page_uses_next_page_mark_and_preserves_raw_metadata() -> None:
    requests: list[httpx.Request] = []
    payload = {
        "count": 2,
        "nextPage": "https://api.govinfo.gov/packages/USCOURTS-nyd-1_24-cv-123/granules?offsetMark=next%2Bmark&pageSize=1&api_key=secret",
        "granules": [
            {
                "granuleId": "USCOURTS-nyd-1_24-cv-123-0",
                "title": "Opinion",
                "granuleLink": "https://api.govinfo.gov/packages/USCOURTS-nyd-1_24-cv-123/granules/USCOURTS-nyd-1_24-cv-123-0/summary",
            }
        ],
        "otherMetadata": {"requestKey": "secret"},
    }

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=payload)

    session = httpx.Client(transport=httpx.MockTransport(respond))
    client = GovInfoClient(GovInfoConfig(api_key="secret"), session=session)
    first = client.list_granules("USCOURTS-nyd-1_24-cv-123", page_size=1)
    client.list_granules("USCOURTS-nyd-1_24-cv-123", offset_mark=first.next_offset_mark or "")

    assert isinstance(first, GovInfoGranulesPage)
    assert first.count == 2
    assert first.next_offset_mark == "next+mark"
    assert first.granules == tuple(payload["granules"])
    assert first.raw_json["otherMetadata"] == {"requestKey": "[redacted]"}
    assert "secret" not in first.raw_json["nextPage"]
    assert requests[0].url.path == "/packages/USCOURTS-nyd-1_24-cv-123/granules"
    assert dict(requests[0].url.params) == {"offsetMark": "*", "pageSize": "1", "api_key": "secret"}
    assert requests[1].url.params["offsetMark"] == "next+mark"


def test_govinfo_granule_summary_returns_raw_download_links_and_checks_identity() -> None:
    requests: list[httpx.Request] = []
    summary = {
        "packageId": "USCOURTS-nyd-1_24-cv-123",
        "granuleId": "opinion-1",
        "dateIssued": "2024-06-01",
        "download": {
            "pdfLink": "https://api.govinfo.gov/packages/USCOURTS-nyd-1_24-cv-123/granules/opinion-1/pdf"
        },
        "nested": {"value": 4, "echoed_key": "secret"},
    }

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=summary)

    session = httpx.Client(transport=httpx.MockTransport(respond))
    client = GovInfoClient(GovInfoConfig(api_key="secret"), session=session)

    result = client.get_granule_summary("USCOURTS-nyd-1_24-cv-123", "opinion-1")
    assert result["download"] == summary["download"]
    assert result["nested"] == {"value": 4, "echoed_key": "[redacted]"}
    assert requests[0].url.path.endswith("/granules/opinion-1/summary")
    assert requests[0].url.params["api_key"] == "secret"
    summary["granuleId"] = "different"
    with pytest.raises(GovInfoError) as raised:
        client.get_granule_summary("USCOURTS-nyd-1_24-cv-123", "opinion-1")
    assert raised.value.failure_type == "invalid_payload"


def test_govinfo_download_pdf_uses_configured_key_only_on_valid_api_link() -> None:
    requests: list[httpx.Request] = []
    pdf = b"%PDF-1.7\nbody bytes"

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, content=pdf, headers={"Content-Type": "application/pdf"})

    session = httpx.Client(transport=httpx.MockTransport(respond))
    client = GovInfoClient(GovInfoConfig(api_key="secret"), session=session)
    url = "https://api.govinfo.gov/packages/USCOURTS-nyd-1_24-cv-123/granules/opinion-1/pdf"

    assert client.download_pdf(url) == pdf
    assert dict(requests[0].url.params) == {"api_key": "secret"}
    assert requests[0].headers["Accept"] == "application/pdf"
    for invalid in (
        "https://evil.example/packages/USCOURTS-nyd-1_24-cv-123/granules/opinion-1/pdf",
        "https://api.govinfo.gov.evil.example/packages/USCOURTS-nyd-1_24-cv-123/pdf",
        "http://api.govinfo.gov/packages/USCOURTS-nyd-1_24-cv-123/pdf",
        "https://api.govinfo.gov/packages/USCOURTS-nyd-1_24-cv-123/granules/opinion-1/summary",
        url + "?api_key=attacker-key",
        "https://api.govinfo.gov/packages/USCOURTS-nyd-1_24-cv-123/granules/%2E%2E/pdf",
    ):
        with pytest.raises(ValueError, match="PDF URL"):
            client.download_pdf(invalid)
    assert len(requests) == 1


@pytest.mark.parametrize("method", ["list", "summary", "pdf"])
def test_govinfo_granule_get_errors_are_structured_and_redacted(method: str) -> None:
    def respond(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"message": "secret unavailable"})

    session = httpx.Client(transport=httpx.MockTransport(respond))
    client = GovInfoClient(GovInfoConfig(api_key="secret"), session=session)
    with pytest.raises(GovInfoError) as raised:
        if method == "list":
            client.list_granules("USCOURTS-nyd-1_24-cv-123")
        elif method == "summary":
            client.get_granule_summary("USCOURTS-nyd-1_24-cv-123", "opinion-1")
        else:
            client.download_pdf("https://api.govinfo.gov/packages/USCOURTS-nyd-1_24-cv-123/pdf")
    error = raised.value
    assert error.failure_type == "http_error"
    assert error.upstream_status_code == 503
    assert "secret" not in str(error)
    assert "secret" not in (error.url or "")
    assert "secret" not in repr(error.upstream_detail)


@pytest.mark.parametrize(
    "payload", [[], {"count": 1, "granules": ["bad"]}, {"count": 1, "granules": [], "nextPage": "?page=2"}]
)
def test_govinfo_rejects_invalid_granule_pages(payload: object) -> None:
    session = httpx.Client(transport=httpx.MockTransport(lambda _request: httpx.Response(200, json=payload)))
    client = GovInfoClient(GovInfoConfig(api_key="secret"), session=session)
    with pytest.raises(GovInfoError) as raised:
        client.list_granules("USCOURTS-nyd-1_24-cv-123")
    assert raised.value.failure_type == "invalid_payload"


def test_govinfo_granule_transport_error_and_invalid_json_are_structured() -> None:
    def disconnected(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"secret failed at {request.url}", request=request)

    client = GovInfoClient(
        GovInfoConfig(api_key="secret"), session=httpx.Client(transport=httpx.MockTransport(disconnected))
    )
    with pytest.raises(GovInfoError) as raised:
        client.list_granules("USCOURTS-nyd-1_24-cv-123")
    assert raised.value.failure_type == "transport_error"
    assert "secret" not in str(raised.value)
    assert "secret" not in repr(raised.value.upstream_detail)

    client = GovInfoClient(
        GovInfoConfig(api_key="secret"),
        session=httpx.Client(
            transport=httpx.MockTransport(lambda _request: httpx.Response(200, text="secret invalid"))
        ),
    )
    with pytest.raises(GovInfoError) as raised:
        client.get_granule_summary("USCOURTS-nyd-1_24-cv-123", "opinion-1")
    assert raised.value.failure_type == "invalid_json"
    assert "secret" not in repr(raised.value.upstream_detail)


def test_govinfo_pdf_redirect_is_a_structured_failure() -> None:
    client = GovInfoClient(
        GovInfoConfig(api_key="secret"),
        session=httpx.Client(
            transport=httpx.MockTransport(
                lambda _request: httpx.Response(
                    302, headers={"Location": "https://outside.example/document.pdf"}
                )
            )
        ),
    )
    with pytest.raises(GovInfoError) as raised:
        client.download_pdf("https://api.govinfo.gov/packages/USCOURTS-nyd-1_24-cv-123/pdf")
    assert raised.value.failure_type == "http_error"
    assert raised.value.upstream_status_code == 302
