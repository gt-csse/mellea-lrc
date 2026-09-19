"""Small, typed client for GovInfo's U.S. Courts Opinions search service."""

from __future__ import annotations

import os
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any
from urllib.parse import urljoin

import requests

DEFAULT_BASE_URL = "https://api.govinfo.gov/"
DEFAULT_API_KEY = "DEMO_KEY"
DEFAULT_USER_AGENT = "mellea-lrc (+https://github.com/gt-csse/mellea-lrc)"


@dataclass(frozen=True, slots=True)
class GovInfoConfig:
    """Configuration for the GovInfo API Search Service."""

    base_url: str = DEFAULT_BASE_URL
    api_key: str = DEFAULT_API_KEY

    @classmethod
    def from_env(cls) -> GovInfoConfig:
        """Load the optional GovInfo API key, using its public demo key by default."""
        return cls(
            base_url=os.getenv("GOVINFO_BASE_URL", DEFAULT_BASE_URL),
            api_key=os.getenv("GOVINFO_API_KEY", "").strip() or DEFAULT_API_KEY,
        )


class GovInfoError(RuntimeError):
    """A structured failure returned while querying the GovInfo API."""

    def __init__(
        self,
        message: str,
        *,
        failure_type: str,
        upstream_status_code: int | None = None,
        retryable: bool = False,
        url: str | None = None,
        upstream_detail: Any = None,
    ) -> None:
        super().__init__(message)
        self.failure_type = failure_type
        self.upstream_status_code = upstream_status_code
        self.retryable = retryable
        self.url = url
        self.upstream_detail = upstream_detail


@dataclass(frozen=True, slots=True)
class GovInfoSearchResult:
    """A preserved page of USCOURTS package results."""

    query: str
    count: int
    results: tuple[dict[str, object], ...]
    next_offset_mark: str | None


class GovInfoClient:
    """Search Federal opinions by the case number GovInfo indexes."""

    def __init__(
        self,
        config: GovInfoConfig | None = None,
        session: requests.Session | None = None,
    ) -> None:
        self.config = config or GovInfoConfig.from_env()
        self.session = session or requests.Session()

    def search_uscourts_docket(
        self,
        docket_number: str,
        *,
        court_id: str | None,
        page_size: int,
    ) -> GovInfoSearchResult:
        """Return USCOURTS packages matching the stated docket without alteration.

        This is intentionally an archive lookup, not another docket grammar:
        the query retains the filing's literal docket text and uses an extracted
        court only as an optional narrowing condition.
        """
        query = govinfo_uscourts_docket_query(docket_number, court_id=court_id)
        url = urljoin(self.config.base_url.rstrip("/") + "/", "search")
        body = {
            "query": query,
            "pageSize": page_size,
            "offsetMark": "*",
            "resultLevel": "package",
            "sorts": [{"field": "score", "sortOrder": "DESC"}],
        }
        try:
            response = self.session.request(
                "POST",
                url,
                params={"api_key": self.config.api_key},
                json=body,
                headers={"User-Agent": DEFAULT_USER_AGENT},
                timeout=45,
            )
        except requests.Timeout as exc:
            raise GovInfoError(
                "GovInfo request timed out",
                failure_type="upstream_timeout",
                retryable=True,
                url=url,
            ) from exc
        except requests.RequestException as exc:
            raise GovInfoError(
                "GovInfo request failed before a response was received",
                failure_type="upstream_request_error",
                retryable=True,
                url=url,
                upstream_detail=str(exc),
            ) from exc
        if response.status_code >= 400:
            raise GovInfoError(
                f"GovInfo returned HTTP {response.status_code}",
                failure_type="upstream_http_error",
                upstream_status_code=response.status_code,
                retryable=response.status_code in {429, 500, 502, 503, 504},
                url=response.url,
                upstream_detail=_response_detail(response),
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise GovInfoError(
                "GovInfo returned invalid JSON",
                failure_type="upstream_invalid_response",
                upstream_status_code=response.status_code,
                retryable=False,
                url=response.url,
                upstream_detail=response.text[:1000],
            ) from exc
        return _search_result(payload, query=query, url=response.url)


def govinfo_uscourts_docket_query(docket_number: str, *, court_id: str | None) -> str:
    """Build the documented USCOURTS case-number query from source text."""
    escaped = docket_number.replace("\\", "\\\\").replace('"', '\\"')
    clauses = ["collection:uscourts", f'casenumber:("{escaped}")']
    if court_id:
        clauses.append(f"courtCode:{court_id}")
    return " ".join(clauses)


def govinfo_package_url(package_id: str | None) -> str | None:
    """Return the public details page for a USCOURTS package."""
    return f"https://www.govinfo.gov/app/details/{package_id}" if package_id else None


def govinfo_package_candidate(result: dict[str, object]) -> dict[str, object]:
    """Expose a search result using the project-wide candidate field names."""
    package_id = _string(result.get("packageId"))
    court_id, docket_number = _package_identity(package_id)
    return {
        **result,
        "govinfo_package_id": package_id,
        "caseName": _string(result.get("title")),
        "court_id": court_id,
        "docketNumber": docket_number,
        "decisionDate": _string(result.get("dateIssued")),
    }


def _search_result(payload: object, *, query: str, url: str) -> GovInfoSearchResult:
    if not isinstance(payload, dict):
        raise GovInfoError(
            "GovInfo returned a non-object search response",
            failure_type="upstream_invalid_response",
            retryable=False,
            url=url,
            upstream_detail=payload,
        )
    count = payload.get("count")
    results = payload.get("results")
    if not isinstance(count, int) or count < 0 or not isinstance(results, list):
        raise GovInfoError(
            "GovInfo returned an invalid search response shape",
            failure_type="upstream_invalid_response",
            retryable=False,
            url=url,
            upstream_detail=payload,
        )
    normalized: list[dict[str, object]] = []
    for item in results:
        if not isinstance(item, dict):
            raise GovInfoError(
                "GovInfo returned a non-object search result",
                failure_type="upstream_invalid_response",
                retryable=False,
                url=url,
                upstream_detail=item,
            )
        normalized.append(_freeze_json(item))
    offset_mark = payload.get("offsetMark")
    return GovInfoSearchResult(
        query=query,
        count=count,
        results=tuple(normalized),
        next_offset_mark=offset_mark if isinstance(offset_mark, str) else None,
    )


def _freeze_json(value: object) -> object:
    if isinstance(value, dict):
        return MappingProxyType({str(key): _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _package_identity(package_id: str | None) -> tuple[str | None, str | None]:
    """Read Court code and literal stored case number from a USCOURTS package ID."""
    if package_id is None or not package_id.startswith("USCOURTS-"):
        return None, None
    remainder = package_id.removeprefix("USCOURTS-")
    court_id, separator, encoded_docket = remainder.partition("-")
    if not separator or not court_id or not encoded_docket:
        return None, None
    return court_id, encoded_docket.replace("_", ":")


def _response_detail(response: requests.Response) -> object:
    try:
        return response.json()
    except ValueError:
        return response.text[:1000]


def _string(value: object) -> str | None:
    return value if isinstance(value, str) else None
