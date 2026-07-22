"""Direct CourtListener API client."""

# ruff: noqa: ANN401, EM101, PLR2004, TRY003

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import urljoin

import requests
from pydantic import ValidationError

from mellea_lrc.courtlistener.citation_lookup import normalize_citation_lookup_payload
from mellea_lrc.courtlistener.docket import normalize_docket_payload
from mellea_lrc.courtlistener.protocols import CourtListenerServiceClient
from mellea_lrc.courtlistener.search import normalize_search_payload

if TYPE_CHECKING:
    from mellea_lrc.courtlistener.citation_lookup_models import CourtListenerCitationLookup
    from mellea_lrc.courtlistener.docket_models import CourtListenerDocket
    from mellea_lrc.courtlistener.search_models import CourtListenerSearchResult


DEFAULT_BASE_URL = "https://www.courtlistener.com/api/rest/v4/"
DEFAULT_USER_AGENT = "mellea-lrc (+https://github.com/gt-csse/mellea-lrc)"


@dataclass(frozen=True, slots=True)
class CourtListenerConfig:
    """Configuration for direct CourtListener API access."""

    base_url: str = DEFAULT_BASE_URL
    token: str | None = None

    @classmethod
    def from_env(cls) -> CourtListenerConfig:
        """Load the API base URL and single API token from the environment."""
        token = os.getenv("COURTLISTENER_API_TOKEN", "").strip() or None
        return cls(base_url=os.getenv("COURTLISTENER_BASE_URL", DEFAULT_BASE_URL), token=token)


class CourtListenerError(RuntimeError):
    """Structured failure raised by direct CourtListener requests."""

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
        self.message = message
        self.failure_type = failure_type
        self.upstream_status_code = upstream_status_code
        self.retryable = retryable
        self.url = url
        self.upstream_detail = upstream_detail


class CourtListenerClient(CourtListenerServiceClient):
    """Direct client for the CourtListener API."""

    def __init__(
        self,
        config: CourtListenerConfig | None = None,
        session: requests.Session | None = None,
    ) -> None:
        self.config = config or CourtListenerConfig.from_env()
        self.session = session or requests.Session()

    def lookup_citation(
        self,
        volume: str,
        reporter: str,
        page: str,
    ) -> CourtListenerCitationLookup:
        """Look up one exact reporter citation by volume, reporter, and page."""
        response = self._send_citation_lookup({"volume": volume, "reporter": reporter, "page": page})
        payload = self._response_payload(response)
        try:
            return normalize_citation_lookup_payload(payload)
        except ValidationError as exc:
            raise CourtListenerError(
                "CourtListener returned an invalid citation-lookup response",
                failure_type="upstream_invalid_response",
                upstream_status_code=response.status_code,
                retryable=False,
                url=response.url,
                upstream_detail=exc.errors(include_url=False),
            ) from exc

    def search(
        self,
        query: str,
        search_type: Literal["r", "rd", "d", "o"],
        cursor: str | None = None,
        *,
        semantic: bool = False,
    ) -> CourtListenerSearchResult:
        """Search CourtListener's opinions, RECAP, docket, or document corpus."""
        if search_type not in {"r", "rd", "d", "o"}:
            raise ValueError("search_type must be one of: r, rd, d, o")
        params: dict[str, str] = {"q": query, "type": search_type}
        if cursor:
            params["cursor"] = cursor
        if semantic:
            params["semantic"] = "true"
        response = self._send_search(params)
        payload = self._response_payload(response)
        try:
            return normalize_search_payload(
                payload,
                query=query,
                search_type=search_type,
                semantic=semantic,
            )
        except ValidationError as exc:
            raise CourtListenerError(
                "CourtListener returned an invalid search response",
                failure_type="upstream_invalid_response",
                upstream_status_code=response.status_code,
                retryable=False,
                url=response.url,
                upstream_detail=exc.errors(include_url=False),
            ) from exc

    def get_docket(self, docket_id: str) -> CourtListenerDocket:
        """Retrieve one docket by its CourtListener identifier."""
        response = self._send_docket(docket_id)
        payload = self._response_payload(response)
        try:
            return normalize_docket_payload(payload)
        except ValidationError as exc:
            raise CourtListenerError(
                "CourtListener returned an invalid docket response",
                failure_type="upstream_invalid_response",
                upstream_status_code=response.status_code,
                retryable=False,
                url=response.url,
                upstream_detail=exc.errors(include_url=False),
            ) from exc

    def _send_citation_lookup(self, data: dict[str, str]) -> requests.Response:
        """POST one exact citation lookup without altering its established contract."""
        url = urljoin(self.config.base_url.rstrip("/") + "/", "citation-lookup/")
        try:
            response = self.session.request(
                "POST",
                url,
                data=data,
                headers=self._headers(),
                timeout=45,
            )
        except requests.Timeout as exc:
            raise CourtListenerError(
                "CourtListener request timed out",
                failure_type="upstream_timeout",
                retryable=True,
                url=url,
            ) from exc
        except requests.RequestException as exc:
            raise CourtListenerError(
                "CourtListener request failed before a response was received",
                failure_type="upstream_request_error",
                retryable=True,
                url=url,
                upstream_detail=str(exc),
            ) from exc
        if response.status_code >= 400:
            raise _courtlistener_http_error(response)
        return response

    def _send_search(self, params: dict[str, str]) -> requests.Response:
        """GET CourtListener search without coupling it to citation lookup."""
        url = urljoin(self.config.base_url.rstrip("/") + "/", "search/")
        try:
            response = self.session.request(
                "GET",
                url,
                params=params,
                headers=self._headers(),
                timeout=45,
            )
        except requests.Timeout as exc:
            raise CourtListenerError(
                "CourtListener request timed out",
                failure_type="upstream_timeout",
                retryable=True,
                url=url,
            ) from exc
        except requests.RequestException as exc:
            raise CourtListenerError(
                "CourtListener request failed before a response was received",
                failure_type="upstream_request_error",
                retryable=True,
                url=url,
                upstream_detail=str(exc),
            ) from exc
        if response.status_code >= 400:
            raise _courtlistener_http_error(response)
        return response

    def _send_docket(self, docket_id: str) -> requests.Response:
        """GET one docket without coupling it to citation lookup."""
        url = urljoin(self.config.base_url.rstrip("/") + "/", f"dockets/{docket_id}/")
        try:
            response = self.session.request("GET", url, headers=self._headers(), timeout=45)
        except requests.Timeout as exc:
            raise CourtListenerError(
                "CourtListener request timed out", failure_type="upstream_timeout", retryable=True, url=url
            ) from exc
        except requests.RequestException as exc:
            raise CourtListenerError(
                "CourtListener request failed before a response was received",
                failure_type="upstream_request_error",
                retryable=True,
                url=url,
                upstream_detail=str(exc),
            ) from exc
        if response.status_code >= 400:
            raise _courtlistener_http_error(response)
        return response

    def _response_payload(self, response: requests.Response) -> object:
        try:
            return response.json()
        except ValueError as exc:
            raise CourtListenerError(
                "CourtListener returned a non-JSON response",
                failure_type="upstream_invalid_json",
                upstream_status_code=response.status_code,
                retryable=True,
                url=response.url,
                upstream_detail=response.text[:500],
            ) from exc

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json", "User-Agent": DEFAULT_USER_AGENT}
        if self.config.token:
            headers["Authorization"] = f"Token {self.config.token}"
        return headers


def _courtlistener_http_error(response: requests.Response) -> CourtListenerError:
    failure_type = _failure_type_for_status(response.status_code)
    return CourtListenerError(
        f"CourtListener citation lookup failed with {response.status_code}",
        failure_type=failure_type,
        upstream_status_code=response.status_code,
        retryable=failure_type in {"api_limit", "upstream_error"},
        url=response.url,
        upstream_detail=_response_detail(response),
    )


def _failure_type_for_status(status_code: int) -> str:
    if status_code == 429:
        return "api_limit"
    if status_code in {401, 403}:
        return "upstream_auth"
    if status_code == 404:
        return "upstream_not_found"
    if 400 <= status_code < 500:
        return "upstream_bad_request"
    return "upstream_error"


def _response_detail(response: requests.Response) -> Any:
    try:
        return response.json()
    except ValueError:
        return response.text[:500]
