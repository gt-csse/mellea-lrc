"""Narrow HTTP client for CourtListener citation lookup and docket metadata."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal
from urllib.parse import urlparse

import httpx
from pydantic import ValidationError

from mellea_lrc.configuration import read_env, required_setting
from mellea_lrc.providers.courtlistener.models import (
    CourtListenerCitationLookup,
    CourtListenerDocket,
    CourtListenerSearchPage,
)

_USER_AGENT = "mellea-lrc (+https://github.com/gt-csse/mellea-lrc)"


class CourtListenerError(RuntimeError):
    """Base class for CourtListener configuration and request failures."""

    def __init__(
        self,
        message: str,
        *,
        failure_type: str,
        upstream_status_code: int | None = None,
        url: str | None = None,
        upstream_detail: Any = None,
    ) -> None:
        super().__init__(message)
        self.failure_type = failure_type
        self.upstream_status_code = upstream_status_code
        self.url = url
        self.upstream_detail = upstream_detail


class CourtListenerConfigurationError(CourtListenerError):
    """The configured proxy URL is absent or invalid."""


class CourtListenerTransportError(CourtListenerError):
    """The request failed before an HTTP response arrived."""


class CourtListenerHTTPError(CourtListenerError):
    """A CourtListener endpoint returned an HTTP error."""


class CourtListenerPayloadError(CourtListenerError):
    """The endpoint returned JSON that does not satisfy the lookup contract."""


@dataclass(frozen=True, slots=True)
class CourtListenerConfig:
    """Configuration for the CourtListener proxy endpoint."""

    base_url: str
    timeout_seconds: float
    token: str | None = field(default=None, repr=False)
    pool: str | None = None

    def __post_init__(self) -> None:
        try:
            if not isinstance(self.base_url, str):
                raise ValueError("Base URL must be a string")
            parsed = urlparse(self.base_url)
            valid_url = (
                parsed.scheme in {"http", "https"}
                and bool(parsed.hostname)
                and not parsed.query
                and not parsed.fragment
                and parsed.username is None
                and parsed.password is None
                and not any(character.isspace() for character in self.base_url)
            )
            parsed.port
        except (TypeError, ValueError):
            valid_url = False
        if not valid_url:
            raise CourtListenerConfigurationError(
                "COURTLISTENER_BASE_URL must be an HTTP(S) base URL without credentials, a query, or fragment",
                failure_type="invalid_base_url",
            )
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or not math.isfinite(self.timeout_seconds)
            or self.timeout_seconds <= 0
        ):
            raise ValueError("COURTLISTENER_TIMEOUT_SECONDS must be a positive finite number")

    @classmethod
    def from_env(cls) -> CourtListenerConfig:
        """Use the configured proxy with its own rotating CourtListener tokens.

        Credentials kept in .env are not sent by default. Callers
        must construct an explicit config to select a token or proxy pool.
        """
        values = read_env()
        try:
            base_url = required_setting(values, "COURTLISTENER_BASE_URL")
        except RuntimeError as error:
            raise CourtListenerConfigurationError(
                "COURTLISTENER_BASE_URL must be configured for citation lookup",
                failure_type="missing_base_url",
            ) from error
        raw_timeout = required_setting(values, "COURTLISTENER_TIMEOUT_SECONDS")
        try:
            timeout = float(raw_timeout)
        except ValueError as error:
            raise ValueError("COURTLISTENER_TIMEOUT_SECONDS must be a positive finite number") from error
        return cls(base_url=base_url, timeout_seconds=timeout)


class CourtListenerClient:
    """Look up citations, search hits, and records through the configured proxy."""

    def __init__(
        self,
        config: CourtListenerConfig | None = None,
        *,
        http_client: httpx.Client | None = None,
    ) -> None:
        self.config = config if config is not None else CourtListenerConfig.from_env()
        self._http_client = http_client if http_client is not None else httpx.Client()
        self._owns_http_client = http_client is None

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json", "User-Agent": _USER_AGENT}
        if self.config.token:
            headers["Authorization"] = f"Token {self.config.token}"
        if self.config.pool:
            headers["x-cl-pool"] = self.config.pool
        return headers

    def _request(
        self,
        method: Literal["get", "post"],
        url: str,
        *,
        operation: str,
        missing_ok: bool = False,
        params: dict[str, str] | None = None,
        data: dict[str, str] | None = None,
    ) -> httpx.Response | None:
        options: dict[str, Any] = {"headers": self._headers(), "timeout": self.config.timeout_seconds}
        if params is not None:
            options["params"] = params
        if data is not None:
            options["data"] = data
        try:
            response = getattr(self._http_client, method)(url, **options)
        except httpx.TransportError as exc:
            raise CourtListenerTransportError(
                f"CourtListener {operation} failed before a response was received",
                failure_type="transport_error",
                url=url,
                upstream_detail=str(exc),
            ) from exc
        if missing_ok and response.status_code == 404:
            return None
        if response.status_code >= 400:
            raise CourtListenerHTTPError(
                f"CourtListener {operation} returned HTTP {response.status_code}",
                failure_type="http_error",
                upstream_status_code=response.status_code,
                url=str(response.url),
                upstream_detail=response.text[:500],
            )
        return response

    @staticmethod
    def _json(response: httpx.Response, operation: str) -> Any:
        try:
            return response.json()
        except ValueError as exc:
            raise CourtListenerPayloadError(
                f"CourtListener {operation} returned invalid JSON",
                failure_type="invalid_json",
                upstream_status_code=response.status_code,
                url=str(response.url),
                upstream_detail=response.text[:500],
            ) from exc

    def lookup_citation(self, volume: str, reporter: str, page: str) -> CourtListenerCitationLookup:
        """Return CourtListener's one result, including every candidate cluster."""
        response = self._request(
            "post",
            self.config.base_url.rstrip("/") + "/citation-lookup/",
            operation="citation lookup",
            data={"volume": volume, "reporter": reporter, "page": page},
        )
        assert response is not None
        payload = self._json(response, "citation lookup")
        if not isinstance(payload, list) or len(payload) != 1:
            raise CourtListenerPayloadError(
                "CourtListener citation lookup must return a list with exactly one result",
                failure_type="invalid_payload",
                upstream_status_code=response.status_code,
                url=str(response.url),
            )
        try:
            return CourtListenerCitationLookup.model_validate(payload[0])
        except ValidationError as exc:
            raise CourtListenerPayloadError(
                "CourtListener citation lookup returned an invalid result",
                failure_type="invalid_payload",
                upstream_status_code=response.status_code,
                url=str(response.url),
                upstream_detail=exc.errors(include_url=False),
            ) from exc

    def get_docket(self, docket_id: str) -> CourtListenerDocket | None:
        """Return a docket's court metadata, or None when it no longer exists."""
        if not docket_id.isdecimal():
            raise ValueError("CourtListener docket ID must contain only decimal digits")
        response = self._request(
            "get",
            self.config.base_url.rstrip("/") + f"/dockets/{docket_id}/",
            operation="docket lookup",
            missing_ok=True,
        )
        if response is None:
            return None
        payload = self._json(response, "docket lookup")
        try:
            docket = CourtListenerDocket.model_validate(payload)
        except ValidationError as exc:
            raise CourtListenerPayloadError(
                "CourtListener docket lookup returned an invalid result",
                failure_type="invalid_payload",
                upstream_status_code=response.status_code,
                url=str(response.url),
                upstream_detail=exc.errors(include_url=False),
            ) from exc
        if docket.id != docket_id:
            raise CourtListenerPayloadError(
                "CourtListener docket lookup returned a different docket ID",
                failure_type="invalid_payload",
                upstream_status_code=response.status_code,
                url=str(response.url),
            )
        return docket

    def _get_raw_record(self, endpoint: str, record_id: str, label: str) -> dict[str, Any] | None:
        if not record_id.isdecimal():
            raise ValueError(f"CourtListener {label} ID must contain only decimal digits")
        response = self._request(
            "get",
            self.config.base_url.rstrip("/") + f"/{endpoint}/{record_id}/",
            operation=f"{label} lookup",
            missing_ok=True,
        )
        if response is None:
            return None
        payload = self._json(response, f"{label} lookup")
        if not isinstance(payload, dict):
            raise CourtListenerPayloadError(
                f"CourtListener {label} lookup returned an invalid result",
                failure_type="invalid_payload",
                upstream_status_code=response.status_code,
                url=str(response.url),
            )
        return payload

    def get_opinion(self, opinion_id: str) -> dict[str, Any] | None:
        """Return an opinion's full upstream JSON, or None when it is missing."""
        return self._get_raw_record("opinions", opinion_id, "opinion")

    def get_recap_document(self, recap_document_id: str) -> dict[str, Any] | None:
        """Return a RECAP document's full upstream JSON, or None when missing."""
        return self._get_raw_record("recap-documents", recap_document_id, "RECAP document")

    def search(
        self, q: str, search_type: Literal["d", "o", "rd"], cursor: str | None = None
    ) -> CourtListenerSearchPage:
        """Search dockets, opinions, or RECAP documents in upstream order."""
        if search_type not in {"d", "o", "rd"}:
            raise ValueError("CourtListener search type must be 'd', 'o', or 'rd'")
        params = {"q": q, "type": search_type}
        if cursor is not None:
            params["cursor"] = cursor
        response = self._request(
            "get",
            self.config.base_url.rstrip("/") + "/search/",
            operation="search",
            params=params,
        )
        assert response is not None
        payload = self._json(response, "search")
        try:
            return CourtListenerSearchPage.model_validate(payload)
        except ValidationError as exc:
            raise CourtListenerPayloadError(
                "CourtListener search returned an invalid result page",
                failure_type="invalid_payload",
                upstream_status_code=response.status_code,
                url=str(response.url),
                upstream_detail=exc.errors(include_url=False),
            ) from exc

    def close(self) -> None:
        """Close the HTTP client when this instance created it."""
        if self._owns_http_client:
            self._http_client.close()

    def __enter__(self) -> CourtListenerClient:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()
