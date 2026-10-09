"""Small client for GovInfo package search and granule retrieval."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Literal
from urllib.parse import parse_qs, parse_qsl, quote, urlencode, urlsplit, urlunsplit

import httpx

from mellea_lrc.configuration import read_env, required_setting
from mellea_lrc.providers.govinfo.models import GovInfoGranulesPage, GovInfoSearchPage

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
class GovInfoConfig:
    """GovInfo API settings."""

    base_url: str
    api_key: str = field(repr=False)
    timeout_seconds: float

    def __post_init__(self) -> None:
        try:
            if not isinstance(self.base_url, str):
                raise ValueError("Base URL must be a string")
            parsed = urlsplit(self.base_url)
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
            raise ValueError(
                "GOVINFO_BASE_URL must be an HTTP(S) base URL without credentials, a query, or fragment"
            )
        if not isinstance(self.api_key, str) or not self.api_key.strip():
            raise ValueError("GOVINFO_API_KEY must be a nonempty string")
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or not math.isfinite(self.timeout_seconds)
            or self.timeout_seconds <= 0
        ):
            raise ValueError("GOVINFO_TIMEOUT_SECONDS must be a positive finite number")

    @classmethod
    def from_env(cls) -> GovInfoConfig:
        values = read_env()
        base_url = required_setting(values, "GOVINFO_BASE_URL")
        api_key = required_setting(values, "GOVINFO_API_KEY")
        raw_timeout = required_setting(values, "GOVINFO_TIMEOUT_SECONDS")
        try:
            timeout = float(raw_timeout)
        except ValueError as error:
            raise ValueError("GOVINFO_TIMEOUT_SECONDS must be a positive finite number") from error
        return cls(base_url=base_url, api_key=api_key, timeout_seconds=timeout)


class GovInfoClient:
    """Search packages and retrieve granules while preserving source metadata."""

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
        result_level: Literal["package", "default"] = "package",
    ) -> GovInfoSearchPage:
        """POST a package or mixed-granule query and validate its page shape."""
        if result_level not in {"package", "default"}:
            raise ValueError("GovInfo search result level must be 'package' or 'default'")
        url = self.config.base_url.rstrip("/") + "/search"
        response = self._request(
            "post",
            url,
            operation="search",
            params={"api_key": self.config.api_key},
            json={
                "query": query,
                "pageSize": page_size,
                "offsetMark": offset_mark,
                "resultLevel": result_level,
                "sorts": [{"field": "score", "sortOrder": "DESC"}],
            },
            accept="application/json",
            error_status=400,
        )
        response_url = str(getattr(response, "url", None) or url)
        payload = self._json(response, "search", fallback_url=url)
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

    def list_granules(
        self,
        package_id: str,
        *,
        offset_mark: str = "*",
        page_size: int = 100,
    ) -> GovInfoGranulesPage:
        """GET one granule page; use ``next_offset_mark`` for the next call."""
        package = _path_id(package_id, "package")
        url = self.config.base_url.rstrip("/") + f"/packages/{package}/granules"
        response = self._request(
            "get",
            url,
            operation="granule list",
            params={"offsetMark": offset_mark, "pageSize": page_size, "api_key": self.config.api_key},
            accept="application/json",
        )
        payload = self._json(response, "granule list")
        if not isinstance(payload, dict):
            raise self._payload_error(
                "GovInfo granule list returned an invalid non-object response", str(response.url), payload
            )
        granules = payload.get("granules")
        count = payload.get("count")
        if (
            not isinstance(granules, list)
            or any(not isinstance(item, dict) for item in granules)
            or not isinstance(count, int)
            or isinstance(count, bool)
            or count < 0
        ):
            raise self._payload_error(
                "GovInfo granule list returned an invalid page", str(response.url), payload
            )
        next_page = payload.get("nextPage")
        if next_page is None:
            next_mark = None
        elif isinstance(next_page, str) and next_page:
            marks = parse_qs(urlsplit(next_page).query).get("offsetMark", [])
            if len(marks) != 1 or not marks[0]:
                raise self._payload_error(
                    "GovInfo granule list returned an invalid nextPage", str(response.url), payload
                )
            next_mark = marks[0]
        else:
            raise self._payload_error(
                "GovInfo granule list returned an invalid nextPage", str(response.url), payload
            )
        raw_json = _redact(payload, self.config.api_key)
        return GovInfoGranulesPage(
            raw_json=raw_json,
            granules=tuple(raw_json["granules"]),
            count=count,
            next_offset_mark=next_mark,
        )

    def get_granule_summary(self, package_id: str, granule_id: str) -> dict[str, Any]:
        """GET a granule's full summary, including its download links."""
        package = _path_id(package_id, "package")
        granule = _path_id(granule_id, "granule")
        url = self.config.base_url.rstrip("/") + f"/packages/{package}/granules/{granule}/summary"
        response = self._request(
            "get",
            url,
            operation="granule summary",
            params={"api_key": self.config.api_key},
            accept="application/json",
        )
        payload = self._json(response, "granule summary")
        if (
            not isinstance(payload, dict)
            or payload.get("packageId") != package_id
            or payload.get("granuleId") != granule_id
        ):
            raise self._payload_error(
                "GovInfo granule summary returned an invalid result", str(response.url), payload
            )
        return _redact(payload, self.config.api_key)

    def download_pdf(self, pdf_url: str) -> bytes:
        """GET bytes from a validated GovInfo package or granule PDF link."""
        url = self._validated_pdf_url(pdf_url)
        response = self._request(
            "get",
            url,
            operation="PDF download",
            params={"api_key": self.config.api_key},
            accept="application/pdf",
        )
        return response.content

    def _request(
        self,
        method: Literal["get", "post"],
        url: str,
        *,
        operation: str,
        params: dict[str, str | int],
        accept: str,
        json: dict[str, Any] | None = None,
        error_status: int = 300,
    ) -> httpx.Response:
        """Keep search's HTTP error contract and reject redirects on downloads."""
        options: dict[str, Any] = {
            "params": params,
            "headers": {"Accept": accept, "User-Agent": _USER_AGENT},
            "timeout": self.config.timeout_seconds,
        }
        if json is not None:
            options["json"] = json
        try:
            response = getattr(self.session, method)(url, **options)
        except httpx.TransportError as exc:
            raise GovInfoError(
                f"GovInfo {operation} failed before a response was received",
                failure_type="transport_error",
                url=url,
                upstream_detail=str(exc),
                api_key=self.config.api_key,
            ) from None
        if response.status_code >= error_status:
            raise GovInfoError(
                f"GovInfo {operation} returned HTTP {response.status_code}",
                failure_type="http_error",
                upstream_status_code=response.status_code,
                url=str(getattr(response, "url", None) or url),
                upstream_detail=_response_detail(response),
                api_key=self.config.api_key,
            )
        return response

    def _json(self, response: httpx.Response, operation: str, *, fallback_url: str | None = None) -> object:
        try:
            return response.json()
        except ValueError as exc:
            raise GovInfoError(
                f"GovInfo {operation} returned invalid JSON",
                failure_type="invalid_json",
                upstream_status_code=response.status_code,
                url=str(getattr(response, "url", None) or fallback_url),
                upstream_detail=getattr(response, "text", "")[:1000],
                api_key=self.config.api_key,
            ) from exc

    def _validated_pdf_url(self, pdf_url: str) -> str:
        base = urlsplit(self.config.base_url)
        link = urlsplit(pdf_url)
        prefix = base.path.rstrip("/") + "/packages/"
        if (
            link.scheme != base.scheme
            or link.netloc != base.netloc
            or link.username is not None
            or link.password is not None
            or link.query
            or link.fragment
            or not link.path.startswith(prefix)
        ):
            raise ValueError("PDF URL must be a GovInfo API package or granule PDF link")
        parts = link.path[len(prefix) :].split("/")
        if not (
            (len(parts) == 2 and parts[1] == "pdf" and _is_path_id(parts[0]))
            or (
                len(parts) == 4
                and parts[1] == "granules"
                and parts[3] == "pdf"
                and _is_path_id(parts[0])
                and _is_path_id(parts[2])
            )
        ):
            raise ValueError("PDF URL must be a GovInfo API package or granule PDF link")
        return pdf_url

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


_PATH_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]*\Z")


def _is_path_id(value: str) -> bool:
    return _PATH_ID.fullmatch(value) is not None


def _path_id(value: str, kind: str) -> str:
    if not _is_path_id(value):
        raise ValueError(f"GovInfo {kind} ID has invalid characters")
    return quote(value, safe="")


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
