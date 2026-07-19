"""HTTP client for a deployed CourtListener access service."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import json
import os
from typing import TYPE_CHECKING
from urllib import error, parse, request

from pydantic import ValidationError

from mellea_lrc.courtlistener.client import CourtListenerError
from mellea_lrc.courtlistener.citation_lookup import normalize_citation_lookup_result
from mellea_lrc.courtlistener.search import normalize_search_payload

if TYPE_CHECKING:
    from mellea_lrc.courtlistener.citation_lookup_models import CourtListenerCitationLookup
    from mellea_lrc.courtlistener.search_models import CourtListenerSearchResult

CL_ACCESS_URL_ENV = "CL_ACCESS_URL"

PostJson = Callable[[str, Mapping[str, str]], object]
GetJson = Callable[[str], object]


@dataclass(frozen=True, slots=True)
class CourtListenerAccessConfig:
    """Configuration for a deployed CourtListener access service."""

    base_url: str

    @classmethod
    def from_env(cls) -> CourtListenerAccessConfig:
        """Load service configuration from environment variables."""
        base_url = os.environ.get(CL_ACCESS_URL_ENV, "").strip()
        if not base_url:
            msg = f"{CL_ACCESS_URL_ENV} must be set to the cl-access service URL"
            raise ValueError(msg)
        return cls(base_url=base_url)


class CourtListenerAccessClient:
    """Small HTTP wrapper around a deployed CourtListener access service."""

    def __init__(
        self,
        config: CourtListenerAccessConfig | None = None,
        *,
        post_json: PostJson | None = None,
        get_json: GetJson | None = None,
    ) -> None:
        self.config = config or CourtListenerAccessConfig.from_env()
        self._post_json = post_json or _post_json
        self._get_json = get_json or _get_json

    def lookup_citation(
        self,
        volume: str,
        reporter: str,
        page: str,
    ) -> CourtListenerCitationLookup:
        """Look up a reporter citation through the remote service."""
        url = f"{self.config.base_url.rstrip('/')}/citation-lookup"
        _validate_http_url(url)
        payload = self._post_json(
            url,
            {"volume": volume, "reporter": reporter, "page": page},
        )
        try:
            return normalize_citation_lookup_result(payload)
        except ValidationError as exc:
            message = "CourtListener access service returned an invalid citation result"
            raise CourtListenerError(
                message,
                failure_type="service_invalid_response",
                retryable=False,
                url=url,
                upstream_detail=exc.errors(include_url=False),
            ) from exc

    def get_docket(self, cl_docket_id: int | str) -> Mapping[str, object]:
        """Retrieve one docket through the remote access service."""
        url = f"{self.config.base_url.rstrip('/')}/dockets/{cl_docket_id}"
        _validate_http_url(url)
        payload = self._get_json(url)
        if not isinstance(payload, Mapping):
            return {}
        if "http_status" not in payload and "detail" not in payload:
            return {**payload, "http_status": 200}
        return payload

    def get_cluster(self, cl_cluster_id: int | str) -> Mapping[str, object]:
        """Retrieve one opinion cluster through the remote access service."""
        url = f"{self.config.base_url.rstrip('/')}/clusters/{cl_cluster_id}"
        _validate_http_url(url)
        payload = self._get_json(url)
        if not isinstance(payload, Mapping):
            return {}
        if "http_status" not in payload and "detail" not in payload:
            return {**payload, "http_status": 200}
        return payload

    def search_docket_entries(
        self,
        cl_docket_id: int | str,
        entry_number: int | str | None = None,
        cursor: str | None = None,
        order_by: str | None = None,
    ) -> Mapping[str, object]:
        """Retrieve docket entries through the remote access service."""
        params = {
            "cl_docket_id": str(cl_docket_id),
            "entry_number": str(entry_number) if entry_number is not None else None,
            "cursor": cursor,
            "order_by": order_by,
        }
        query = parse.urlencode({key: value for key, value in params.items() if value is not None})
        url = f"{self.config.base_url.rstrip('/')}/docket-entries/search?{query}"
        _validate_http_url(url)
        payload = self._get_json(url)
        if not isinstance(payload, Mapping):
            return {"http_status": None, "detail": "Docket-entry response was not a JSON object."}
        if "http_status" not in payload and "detail" not in payload:
            return {**payload, "http_status": 200}
        return payload

    def search_opinions(self, q: str) -> CourtListenerSearchResult:
        """Run an opinion (``type=o``) search through the remote access service."""
        return self._search(q, "o")

    def search_recap(self, q: str) -> CourtListenerSearchResult:
        """Run a RECAP (``type=r``) search through the remote access service."""
        return self._search(q, "r")

    def _search(self, q: str, search_type: str) -> CourtListenerSearchResult:
        query = parse.urlencode({"q": q, "type": search_type})
        url = f"{self.config.base_url.rstrip('/')}/search?{query}"
        _validate_http_url(url)
        payload = self._get_json(url)
        if not isinstance(payload, Mapping):
            payload = {"http_status": None, "detail": "Search response was not a JSON object."}
        return normalize_search_payload(payload, query=q, search_type=search_type)


def _get_json(url: str) -> object:
    req = request.Request(  # noqa: S310
        url,
        headers={"Accept": "application/json"},
        method="GET",
    )
    try:
        with request.urlopen(req, timeout=45) as response:  # noqa: S310
            return json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        return {"http_status": exc.code, "detail": _json_detail(detail) or detail}
    except (OSError, json.JSONDecodeError) as exc:
        return {"http_status": None, "detail": str(exc)}


def _post_json(url: str, data: Mapping[str, str]) -> object:
    encoded = parse.urlencode(data).encode("utf-8")
    req = request.Request(  # noqa: S310
        url,
        data=encoded,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=45) as response:  # noqa: S310
            return json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        raw_detail = exc.read().decode("utf-8", errors="replace")
        detail = _json_detail(raw_detail) or {}
        raise CourtListenerError(
            _error_message(detail, raw_detail),
            failure_type=str(detail.get("failure_type", "service_error")),
            upstream_status_code=_optional_int(detail.get("upstream_status_code")),
            retryable=detail.get("retryable") is True,
            url=url,
            cache_key=_optional_str(detail.get("key")),
            retry_after_seconds=_optional_float(detail.get("retry_after_seconds")),
            upstream_detail=detail.get("upstream_detail"),
        ) from exc
    except (OSError, json.JSONDecodeError) as exc:
        message = "CourtListener access service request failed"
        raise CourtListenerError(
            message,
            failure_type="service_request_error",
            retryable=True,
            url=url,
            upstream_detail=str(exc),
        ) from exc


def _validate_http_url(url: str) -> None:
    scheme = parse.urlparse(url).scheme
    if scheme not in {"http", "https"}:
        msg = f"cl-access URL must be http(s), got: {scheme or '<empty>'}"
        raise ValueError(msg)


def _json_detail(raw: str) -> dict[str, object] | None:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    detail = payload.get("detail", payload)
    return detail if isinstance(detail, dict) else payload


def _error_message(detail: dict[str, object] | None, fallback: str) -> str:
    if detail is None:
        return fallback
    message = detail.get("message")
    if isinstance(message, str) and message:
        return message
    failure_type = detail.get("failure_type")
    if isinstance(failure_type, str) and failure_type:
        return failure_type
    return fallback


def _optional_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _optional_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _optional_float(value: object) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None
