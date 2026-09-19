"""Contract tests for the GovInfo USCOURTS fallback client."""

from __future__ import annotations

from typing import Any

from mellea_lrc.govinfo import (
    GovInfoClient,
    GovInfoConfig,
    govinfo_package_candidate,
    govinfo_uscourts_docket_query,
)


class _Response:
    status_code = 200
    url = "https://api.govinfo.gov/search?api_key=test-key"
    text = ""

    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def json(self) -> dict[str, object]:
        return self.payload


class _Session:
    def __init__(self, response: _Response) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def request(self, method: str, url: str, **kwargs: object) -> _Response:
        self.calls.append({"method": method, "url": url, **kwargs})
        return self.response


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
    assert candidate["decisionDate"] == "2024-01-06"
    assert govinfo_uscourts_docket_query("1:24-cv-08760", court_id=None) == (
        'collection:uscourts casenumber:("1:24-cv-08760")'
    )
