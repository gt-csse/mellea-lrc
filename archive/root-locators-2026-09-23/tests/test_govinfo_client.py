"""Contract tests for the GovInfo USCOURTS fallback client."""

from __future__ import annotations

from datetime import date
from typing import Any
from unittest.mock import patch

import pytest

from mellea_lrc.govinfo import (
    GovInfoClient,
    GovInfoConfig,
    GovInfoError,
    govinfo_package_candidate,
    govinfo_uscourts_docket_query,
)


class _Response:
    def __init__(
        self,
        payload: dict[str, object],
        *,
        status_code: int = 200,
        text: str = "",
        content: bytes = b"",
        headers: dict[str, str] | None = None,
    ) -> None:
        self.payload = payload
        self.status_code = status_code
        self.url = "https://api.govinfo.gov/search?api_key=test-key"
        self.text = text
        self.content = content
        self.headers = headers or {}

    def json(self) -> dict[str, object]:
        return self.payload


class _Session:
    def __init__(self, response: _Response | list[_Response]) -> None:
        self.responses = response if isinstance(response, list) else [response]
        self.calls: list[dict[str, Any]] = []

    def request(self, method: str, url: str, **kwargs: object) -> _Response:
        self.calls.append({"method": method, "url": url, **kwargs})
        return self.responses.pop(0)


def test_govinfo_docket_lookup_preserves_the_stated_number_and_optional_court() -> None:
    response = _Response(
        {
            "count": 1,
            "offsetMark": None,
            "results": [
                {
                    "packageId": "USCOURTS-nysd-1_24-cv-08760",
                    "title": "Smith v. Jones",
                    "dateIssued": "2024-01-06",
                }
            ],
        }
    )
    session = _Session(response)
    client = GovInfoClient(config=GovInfoConfig(api_key="test-key"), session=session)  # type: ignore[arg-type]

    result = client.search_uscourts_docket("1:24-cv-08760", court_id="nysd", page_size=20)

    assert result.count == 1
    assert session.calls == [
        {
            "method": "POST",
            "url": "https://api.govinfo.gov/search",
            "params": {"api_key": "test-key"},
            "json": {
                "query": 'collection:uscourts casenumber:("1:24-cv-08760") courtCode:nysd',
                "pageSize": 20,
                "offsetMark": "*",
                "resultLevel": "package",
                "sorts": [{"field": "score", "sortOrder": "DESC"}],
            },
            "headers": {"User-Agent": "mellea-lrc (+https://github.com/gt-csse/mellea-lrc)"},
            "timeout": 45,
        }
    ]


def test_govinfo_package_candidate_keeps_upstream_package_identity() -> None:
    candidate = govinfo_package_candidate(
        {
            "packageId": "USCOURTS-nysd-1_24-cv-08760",
            "title": "Smith v. Jones",
            "dateIssued": "2024-01-06",
        }
    )

    assert candidate["docketNumber"] == "1:24-cv-08760"
    assert candidate["court_id"] == "nysd"
    assert candidate["packageDateIssued"] == "2024-01-06"
    assert "decisionDate" not in candidate
    assert govinfo_uscourts_docket_query("1:24-cv-08760", court_id=None) == (
        'collection:uscourts casenumber:("1:24-cv-08760")'
    )


def test_package_text_follows_granule_text_link_and_preserves_proxy_base() -> None:
    package_id = "USCOURTS-nysd-1_24-cv-08760"
    api = "https://api.govinfo.gov/packages/" + package_id
    session = _Session(
        [
            _Response({"granulesLink": api + "/granules?offsetMark=*&pageSize=100"}),
            _Response({"granules": [{"granuleLink": api + "/granules/part-0/summary"}]}),
            _Response({"download": {"txtLink": api + "/granules/part-0/txt"}}),
            _Response({}, text="See Smith v. Jones, 123 F.3d 456."),
        ]
    )
    client = GovInfoClient(
        config=GovInfoConfig(base_url="https://proxy.test/govinfo/", api_key="test-key"),
        session=session,  # type: ignore[arg-type]
    )

    text = client.get_package_text(package_id)

    assert text == "See Smith v. Jones, 123 F.3d 456."
    assert [call["url"] for call in session.calls] == [
        "https://proxy.test/govinfo/packages/USCOURTS-nysd-1_24-cv-08760/summary",
        "https://proxy.test/govinfo/packages/USCOURTS-nysd-1_24-cv-08760/granules",
        "https://proxy.test/govinfo/packages/USCOURTS-nysd-1_24-cv-08760/granules/part-0/summary",
        "https://proxy.test/govinfo/packages/USCOURTS-nysd-1_24-cv-08760/granules/part-0/txt",
    ]
    assert session.calls[1]["params"] == {"api_key": "test-key", "offsetMark": "*", "pageSize": 5}
    assert all(call["params"]["api_key"] == "test-key" for call in session.calls)


def test_package_text_extracts_pdf_and_returns_empty_when_unavailable() -> None:
    api = "https://api.govinfo.gov/packages/USCOURTS-nysd-1_24-cv-08760"
    session = _Session(
        [
            _Response({"granulesLink": api + "/granules"}),
            _Response({"granules": [{"granuleLink": api + "/granules/part-0/summary"}]}),
            _Response({"download": {"pdfLink": api + "/granules/part-0/pdf"}}),
            _Response({}, content=b"%PDF-test"),
        ]
    )
    client = GovInfoClient(config=GovInfoConfig(api_key="test-key"), session=session)  # type: ignore[arg-type]
    with patch("mellea_lrc.govinfo.client._pdf_text", return_value="Opinion text") as extract:
        assert client.get_package_text("USCOURTS-nysd-1_24-cv-08760") == "Opinion text"
    extract.assert_called_once_with(b"%PDF-test")

    absent = GovInfoClient(
        config=GovInfoConfig(api_key="test-key"),
        session=_Session(_Response({"packageId": "USCOURTS-nysd-1_24-cv-08760"})),  # type: ignore[arg-type]
    )
    assert absent.get_package_text("USCOURTS-nysd-1_24-cv-08760") == ""


def test_package_text_retrospective_cutoff_applies_to_each_opinion_granule() -> None:
    package_id = "USCOURTS-nysd-1_24-cv-08760"
    api = "https://api.govinfo.gov/packages/" + package_id
    granule_links = [api + f"/granules/part-{index}" for index in range(5)]
    session = _Session(
        [
            _Response({"granulesLink": api + "/granules", "dateIssued": "2025-01-01"}),
            _Response({"granules": [{"granuleLink": link + "/summary"} for link in granule_links]}),
            _Response(
                {
                    "dateIssued": "2024-01-01",
                    "download": {"txtLink": granule_links[0] + "/txt"},
                }
            ),
            _Response({}, text="Older opinion"),
            _Response(
                {
                    "dateIssued": "2024-02-01",
                    "download": {"txtLink": granule_links[1] + "/txt"},
                }
            ),
            _Response({}, text="Same-day opinion"),
            _Response(
                {
                    "dateIssued": "2024-02-02",
                    "download": {"txtLink": granule_links[2] + "/txt"},
                }
            ),
            _Response({"download": {"txtLink": granule_links[3] + "/txt"}}),
            _Response(
                {
                    "dateIssued": "2024-02-30",
                    "download": {"txtLink": granule_links[4] + "/txt"},
                }
            ),
        ]
    )
    client = GovInfoClient(config=GovInfoConfig(api_key="test-key"), session=session)  # type: ignore[arg-type]

    assert (
        client.get_package_text(package_id, retrospective_date=date(2024, 2, 1))
        == "Older opinion\n\nSame-day opinion"
    )
    requested = [call["url"] for call in session.calls]
    assert granule_links[0] + "/txt" in requested
    assert granule_links[1] + "/txt" in requested
    assert all(link + "/txt" not in requested for link in granule_links[2:])


def test_package_text_reports_upstream_fetch_failure() -> None:
    client = GovInfoClient(
        config=GovInfoConfig(api_key="test-key"),
        session=_Session(_Response({"detail": "unavailable"}, status_code=503)),  # type: ignore[arg-type]
    )

    with pytest.raises(GovInfoError, match="HTTP 503") as raised:
        client.get_package_text("USCOURTS-nysd-1_24-cv-08760")
    assert raised.value.failure_type == "upstream_http_error"
    assert raised.value.retryable
