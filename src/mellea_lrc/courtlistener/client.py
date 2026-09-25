"""Narrow HTTP client for CourtListener citation lookup and docket metadata."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import httpx
from dotenv import load_dotenv
from pydantic import ValidationError

from mellea_lrc.courtlistener.models import CourtListenerCitationLookup, CourtListenerDocket

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
    token: str | None = None
    pool: str | None = None

    @classmethod
    def from_env(cls) -> CourtListenerConfig:
        """Read the endpoint and optional headers from the environment or .env."""
        load_dotenv(override=False)
        base_url = os.getenv("COURTLISTENER_BASE_URL", "").strip()
        if not base_url:
            raise CourtListenerConfigurationError(
                "COURTLISTENER_BASE_URL must be configured for citation lookup",
                failure_type="missing_base_url",
            )
        return cls(
            base_url=base_url,
            token=os.getenv("COURTLISTENER_API_TOKEN", "").strip() or None,
            pool=os.getenv("MELLEA_LRC_COURTLISTENER_POOL", "").strip() or None,
        )


class CourtListenerClient:
    """Look up an exact reporter citation and its candidate docket."""

    def __init__(
        self,
        config: CourtListenerConfig | None = None,
        *,
        http_client: httpx.Client | None = None,
    ) -> None:
        self.config = config if config is not None else CourtListenerConfig.from_env()
        parsed = urlparse(self.config.base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.query or parsed.fragment:
            raise CourtListenerConfigurationError(
                "COURTLISTENER_BASE_URL must be an HTTP(S) base URL without a query or fragment",
                failure_type="invalid_base_url",
            )
        self._http_client = http_client if http_client is not None else httpx.Client()
        self._owns_http_client = http_client is None

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json", "User-Agent": _USER_AGENT}
        if self.config.token:
            headers["Authorization"] = f"Token {self.config.token}"
        if self.config.pool:
            headers["x-cl-pool"] = self.config.pool
        return headers

    def lookup_citation(self, volume: str, reporter: str, page: str) -> CourtListenerCitationLookup:
        """Return CourtListener's one result, including every candidate cluster."""
        url = self.config.base_url.rstrip("/") + "/citation-lookup/"
        try:
            response = self._http_client.post(
                url,
                data={"volume": volume, "reporter": reporter, "page": page},
                headers=self._headers(),
                timeout=45,
            )
        except httpx.TransportError as exc:
            raise CourtListenerTransportError(
                "CourtListener citation lookup failed before a response was received",
                failure_type="transport_error",
                url=url,
                upstream_detail=str(exc),
            ) from exc

        if response.status_code >= 400:
            raise CourtListenerHTTPError(
                f"CourtListener citation lookup returned HTTP {response.status_code}",
                failure_type="http_error",
                upstream_status_code=response.status_code,
                url=str(response.url),
                upstream_detail=response.text[:500],
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise CourtListenerPayloadError(
                "CourtListener citation lookup returned invalid JSON",
                failure_type="invalid_json",
                upstream_status_code=response.status_code,
                url=str(response.url),
                upstream_detail=response.text[:500],
            ) from exc

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
        url = self.config.base_url.rstrip("/") + f"/dockets/{docket_id}/"
        try:
            response = self._http_client.get(url, headers=self._headers(), timeout=45)
        except httpx.TransportError as exc:
            raise CourtListenerTransportError(
                "CourtListener docket lookup failed before a response was received",
                failure_type="transport_error",
                url=url,
                upstream_detail=str(exc),
            ) from exc

        if response.status_code == 404:
            return None
        if response.status_code >= 400:
            raise CourtListenerHTTPError(
                f"CourtListener docket lookup returned HTTP {response.status_code}",
                failure_type="http_error",
                upstream_status_code=response.status_code,
                url=str(response.url),
                upstream_detail=response.text[:500],
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise CourtListenerPayloadError(
                "CourtListener docket lookup returned invalid JSON",
                failure_type="invalid_json",
                upstream_status_code=response.status_code,
                url=str(response.url),
                upstream_detail=response.text[:500],
            ) from exc
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

    def close(self) -> None:
        """Close the HTTP client when this instance created it."""
        if self._owns_http_client:
            self._http_client.close()

    def __enter__(self) -> CourtListenerClient:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()
