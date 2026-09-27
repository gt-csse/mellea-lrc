"""Small client for GovInfo's package search API."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qsl, quote, urlencode, urlsplit, urlunsplit

import httpx
from dotenv import load_dotenv

DEFAULT_BASE_URL = "https://api.govinfo.gov/"
DEFAULT_API_KEY = "DEMO_KEY"
_USER_AGENT = "mellea-lrc (+https://github.com/gt-csse/mellea-lrc)"


class GovInfoError(RuntimeError):
    """A GovInfo request or response failure with safe diagnostic metadata."""

    def __init__(
        self,
        message: str,
        *,
        failure_type: str,
        upstream_status_code: int | None = None,
        url: str | None = None,
        upstream_detail: Any = None,
        api_key: str | None = None,
    ) -> None:
        self.failure_type = failure_type
        self.upstream_status_code = upstream_status_code
        self.url = _safe_url(url, api_key)
        self.upstream_detail = _redact(upstream_detail, api_key)
        super().__init__(_redact(message, api_key))


@dataclass(frozen=True, slots=True)
class GovInfoSearchPage:
    """One complete raw API page and its package results."""

    raw_json: dict[str, Any]
    results: tuple[dict[str, Any], ...]
    count: int
    next_offset_mark: str | None


@dataclass(frozen=True, slots=True)
class GovInfoConfig:
    """GovInfo API settings."""

    base_url: str = DEFAULT_BASE_URL
    api_key: str = DEFAULT_API_KEY

    @classmethod
    def from_env(cls) -> GovInfoConfig:
        load_dotenv(override=False)
        return cls(api_key=os.getenv("GOVINFO_API_KEY", "").strip() or DEFAULT_API_KEY)


class GovInfoClient:
    """Search GovInfo packages while preserving each returned page."""

    def __init__(
        self,
        config: GovInfoConfig | None = None,
        *,
        session: httpx.Client | None = None,
    ) -> None:
        self.config = config or GovInfoConfig.from_env()
        self.session = session if session is not None else httpx.Client()
        self._owns_session = session is None

    def search(
        self,
        query: str,
        *,
        offset_mark: str = "*",
        page_size: int = 100,
    ) -> GovInfoSearchPage:
        """POST a package query and validate the response page shape."""
        url = self.config.base_url.rstrip("/") + "/search"
        try:
            response = self.session.post(
                url,
                params={"api_key": self.config.api_key},
                json={
                    "query": query,
                    "pageSize": page_size,
                    "offsetMark": offset_mark,
                    "resultLevel": "package",
                    "sorts": [{"field": "score", "sortOrder": "DESC"}],
                },
                headers={"Accept": "application/json", "User-Agent": _USER_AGENT},
                timeout=45,
            )
        except httpx.TransportError as exc:
            raise GovInfoError(
                "GovInfo search failed before a response was received",
                failure_type="transport_error",
                url=url,
                upstream_detail=str(exc),
                api_key=self.config.api_key,
            ) from None

        response_url = getattr(response, "url", None) or url
        if response.status_code >= 400:
            raise GovInfoError(
                f"GovInfo search returned HTTP {response.status_code}",
                failure_type="http_error",
                upstream_status_code=response.status_code,
                url=response_url,
                upstream_detail=_response_detail(response),
                api_key=self.config.api_key,
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise GovInfoError(
                "GovInfo search returned invalid JSON",
                failure_type="invalid_json",
                upstream_status_code=response.status_code,
                url=response_url,
                upstream_detail=getattr(response, "text", "")[:1000],
                api_key=self.config.api_key,
            ) from exc
        if not isinstance(payload, dict):
            raise self._payload_error(
                "GovInfo search returned an invalid non-object response", response_url, payload
            )
        results = payload.get("results")
        count = payload.get("count")
        if (
            not isinstance(results, list)
            or any(not isinstance(item, dict) for item in results)
            or not isinstance(count, int)
            or isinstance(count, bool)
            or count < 0
        ):
            raise self._payload_error("GovInfo search returned an invalid page", response_url, payload)
        raw_json = _redact(payload, self.config.api_key)
        # A response may echo the request URL in metadata; scrub credentials from
        # any stored strings before exposing the raw page.
        next_mark = payload.get("offsetMark")
        return GovInfoSearchPage(
            raw_json=raw_json,
            results=tuple(raw_json["results"]),
            count=count,
            next_offset_mark=next_mark if isinstance(next_mark, str) and next_mark else None,
        )

    def _payload_error(self, message: str, url: str, detail: object) -> GovInfoError:
        return GovInfoError(
            message,
            failure_type="invalid_payload",
            url=url,
            upstream_detail=detail,
            api_key=self.config.api_key,
        )

    def close(self) -> None:
        """Close the session when this client created it."""
        if self._owns_session:
            self.session.close()

    def __enter__(self) -> GovInfoClient:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()


def govinfo_uscourts_docket_query(number: str) -> str:
    """Build a literal USCOURTS case-number query from the citation text."""
    escaped = number.replace("\\", "\\\\").replace('"', '\\"')
    return f'collection:uscourts casenumber:("{escaped}")'


def _safe_url(value: str | None, secret: str | None) -> str | None:
    if value is None:
        return None
    parsed = urlsplit(value)
    query = [
        (key, "[redacted]" if key.casefold() == "api_key" else item)
        for key, item in parse_qsl(parsed.query, keep_blank_values=True)
    ]
    result = urlunsplit((parsed.scheme, parsed.netloc, parsed.path, urlencode(query), parsed.fragment))
    if secret:
        return result.replace(secret, "[redacted]").replace(quote(secret, safe=""), "[redacted]")
    return result


def _redact(value: Any, secret: str | None) -> Any:
    if isinstance(value, str):
        if not secret:
            return value
        return value.replace(secret, "[redacted]").replace(quote(secret, safe=""), "[redacted]")
    if isinstance(value, dict):
        return {key: _redact(item, secret) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact(item, secret) for item in value]
    return value


def _response_detail(response: httpx.Response) -> object:
    try:
        return response.json()
    except ValueError:
        return getattr(response, "text", "")[:1000]
