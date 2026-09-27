"""Contract tests for the GovInfo package search client."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest
from dotenv import load_dotenv

from mellea_lrc.govinfo import (
    GovInfoClient,
    GovInfoConfig,
    GovInfoError,
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
