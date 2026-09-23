"""Direct CourtListener API client."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from math import isfinite
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import urljoin, urlparse

import requests
from pydantic import ValidationError

from mellea_lrc.courtlistener.citation_lookup import normalize_citation_lookup_payload
from mellea_lrc.courtlistener.docket import normalize_docket_payload
from mellea_lrc.courtlistener.opinion import normalize_opinion_payload
from mellea_lrc.courtlistener.protocols import CourtListenerServiceClient
from mellea_lrc.courtlistener.recap_document_transport import normalize_recap_document_payload
from mellea_lrc.courtlistener.search import normalize_search_payload

if TYPE_CHECKING:
    from mellea_lrc.courtlistener.citation_lookup_models import CourtListenerCitationLookup
    from mellea_lrc.courtlistener.docket_models import CourtListenerDocket
    from mellea_lrc.courtlistener.opinion_models import CourtListenerOpinion
    from mellea_lrc.courtlistener.recap_document_models import CourtListenerRecapDocument
    from mellea_lrc.courtlistener.search_models import CourtListenerSearchResult


DEFAULT_BASE_URL = "https://www.courtlistener.com/api/rest/v4/"
DEFAULT_USER_AGENT = "mellea-lrc (+https://github.com/gt-csse/mellea-lrc)"


@dataclass(frozen=True, slots=True)
class CourtListenerConfig:
    """Configuration for direct CourtListener API access."""

    base_url: str = DEFAULT_BASE_URL
    token: str | None = None
    pool: str | None = None
    retry_short_rate_limits: bool = False

    @classmethod
    def from_env(cls) -> CourtListenerConfig:
        """Load the API base URL, token, and optional proxy request pool.

        ``MELLEA_LRC_COURTLISTENER_POOL`` selects an allowance offered by the
        CourtListener caching proxy. It is sent only when explicitly configured,
        so direct CourtListener callers retain their established contract.
        """
        token = os.getenv("COURTLISTENER_API_TOKEN", "").strip() or None
        pool = os.getenv("MELLEA_LRC_COURTLISTENER_POOL", "").strip() or None
        return cls(
            base_url=os.getenv("COURTLISTENER_BASE_URL", DEFAULT_BASE_URL),
            token=token,
            pool=pool,
        )


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
        retry_after_seconds: float | None = None,
    ) -> None:
        """Initialize the error with its message and structured failure details."""
        super().__init__(message)
        self.message = message
        self.failure_type = failure_type
        self.upstream_status_code = upstream_status_code
        self.retryable = retryable
        self.url = url
        self.upstream_detail = upstream_detail
        self.retry_after_seconds = retry_after_seconds


class CourtListenerClient(CourtListenerServiceClient):
    """Direct client for the CourtListener API."""

    def __init__(
        self,
        config: CourtListenerConfig | None = None,
        session: requests.Session | None = None,
    ) -> None:
        """Initialize the client with its config and HTTP session."""
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

    def get_opinion(self, opinion_id: str) -> CourtListenerOpinion:
        """Retrieve one sub-opinion by its CourtListener identifier."""
        response = self._send_resource("opinions", opinion_id)
        payload = self._response_payload(response)
        try:
            return normalize_opinion_payload(payload)
        except ValidationError as exc:
            raise CourtListenerError(
                "CourtListener returned an invalid opinion response",
                failure_type="upstream_invalid_response",
                upstream_status_code=response.status_code,
                retryable=False,
                url=response.url,
                upstream_detail=exc.errors(include_url=False),
            ) from exc

    def get_recap_document(self, document_id: str) -> CourtListenerRecapDocument:
        """Retrieve the full indexed text of one RECAP filing or attachment."""
        response = self._send_resource("recap-documents", document_id)
        payload = self._response_payload(response)
        try:
            return normalize_recap_document_payload(payload)
        except ValidationError as exc:
            raise CourtListenerError(
                "CourtListener returned an invalid RECAP document response",
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
            response = self._request(
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
            response = self._request(
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
            response = self._request("GET", url, headers=self._headers(), timeout=45)
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

    def _send_resource(self, resource: str, resource_id: str) -> requests.Response:
        """GET one identifier-addressed CourtListener resource."""
        url = urljoin(
            self.config.base_url.rstrip("/") + "/",
            f"{resource}/{resource_id}/",
        )
        try:
            response = self._request("GET", url, headers=self._headers(), timeout=45)
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

    def _request(self, method: str, url: str, **kwargs: Any) -> requests.Response:
        """Optionally wait out one short proxy quota window for every route.

        This sits below search and body retrieval, so secondary quoted queries
        and opinion/RECAP fetches receive the same Retry-After treatment as raw
        searches. Long windows are returned immediately to the calling stage.
        """
        response = self.session.request(method, url, **kwargs)
        if not self.config.retry_short_rate_limits or response.status_code != 429:
            return response
        delay = courtlistener_retry_delay(_courtlistener_http_error(response), retry_number=1)
        if delay is None:
            return response
        time.sleep(delay)
        return self.session.request(method, url, **kwargs)

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
        if self.config.pool:
            headers["x-cl-pool"] = self.config.pool
        return headers


def body_text_client(client: CourtListenerServiceClient) -> CourtListenerServiceClient:
    """Fetch bodies through the configured Modal proxy pool without a direct token.

    An explicit pool must also apply to opinion and RECAP fetches: silently
    dropping it sends those requests to the proxy's default quota, which may
    be exhausted even while the selected pool is available. With no selected
    pool, the proxy still chooses its default key. Other CourtListener hosts
    keep their configured authentication.
    """
    host = urlparse(client.config.base_url).hostname if isinstance(client, CourtListenerClient) else None
    if host is not None and host.endswith(".modal.run"):
        return CourtListenerClient(config=replace(client.config, token=None))
    return client


def _courtlistener_http_error(response: requests.Response) -> CourtListenerError:
    failure_type = _failure_type_for_status(response.status_code)
    return CourtListenerError(
        f"CourtListener request failed with {response.status_code}",
        failure_type=failure_type,
        upstream_status_code=response.status_code,
        retryable=failure_type in {"api_limit", "upstream_error"},
        url=response.url,
        upstream_detail=_response_detail(response),
        retry_after_seconds=_retry_after_seconds(response.headers.get("Retry-After")),
    )


def courtlistener_retry_delay(
    error: CourtListenerError, *, retry_number: int, waited_seconds: float = 0.0
) -> float | None:
    """Give a bounded delay for a short quota window or transient request failure.

    At most two retries and 90 seconds of total waiting are allowed. A proxy's
    longer Retry-After is treated as daily exhaustion and returned to the caller.
    """
    if not error.retryable or retry_number not in (1, 2) or waited_seconds >= 90:
        return None
    if error.failure_type == "api_limit":
        fallback = (2.0, 4.0)[retry_number - 1]
    elif error.failure_type in {"upstream_timeout", "upstream_request_error"} or (
        error.failure_type == "upstream_error"
        and error.upstream_status_code is not None
        and 500 <= error.upstream_status_code < 600
    ):
        fallback = (1.0, 2.0)[retry_number - 1]
    else:
        return None
    delay = error.retry_after_seconds if error.retry_after_seconds is not None else fallback
    if not isfinite(delay) or delay < 0:
        return None
    return delay if delay <= 90 - waited_seconds else None


def _retry_after_seconds(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        seconds = float(value)
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(value)
        except (TypeError, ValueError, IndexError):
            return None
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=UTC)
        seconds = (retry_at - datetime.now(UTC)).total_seconds()
    return max(0.0, seconds) if isfinite(seconds) else None


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
